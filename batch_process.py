import asyncio
import json
import os
import torch
import glob
import re
from dotenv import load_dotenv
from pypdf import PdfReader, PdfWriter

# Load environment variables for Azure OpenAI keys
load_dotenv()

from processor_app.pc_db import PC_Mistral
from langchain_huggingface import HuggingFaceEmbeddings
from utils.parse_to_mistral import _run_marker_sync
from bidnobid.clause_extraction import process_category, smart_deduplicate_pure_token, CATEGORIES, llm

def get_cuda_device():
    """Strictly enforces CUDA usage."""
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available! This script is configured to run ONLY on an NVIDIA GPU.")
    print("GPU Detected: Using strictly NVIDIA CUDA")
    return "cuda"

def _sanitize_collection_name(name: str) -> str:
    """Cleans up the filename to be used as a valid ChromaDB namespace."""
    name = re.sub(r'[^a-zA-Z0-9._-]', '', name)
    name = re.sub(r'^[^a-zA-Z0-9]+', '', name)
    name = re.sub(r'[^a-zA-Z0-9]+$', '', name)
    if len(name) < 3:
        name = f"col_{name}"
    return name

def append_to_json_file(filepath, new_data):
    """Safely reads the existing JSON array, appends the new file's data, and saves it."""
    if os.path.exists(filepath):
        try:
            with open(filepath, "r", encoding="utf-8") as f:
                data = json.load(f)
        except json.JSONDecodeError:
            data = []
    else:
        data = []
    
    data.append(new_data)
    
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=4)

def is_already_processed(master_file, filename):
    """Check if this PDF was already processed and saved in master file."""
    if not os.path.exists(master_file):
        return False
    try:
        with open(master_file, "r", encoding="utf-8") as f:
            data = json.load(f)
        return any(entry.get("filename") == filename for entry in data)
    except Exception:
        return False

def split_pdf_in_half(pdf_path, part1_path, part2_path):
    """Splits a PDF into two equal halves and saves them."""
    reader = PdfReader(pdf_path)
    total_pages = len(reader.pages)
    
    if total_pages < 2:
        raise ValueError("PDF only has 1 page and cannot be split.")
        
    mid = total_pages // 2
    
    writer1 = PdfWriter()
    for i in range(mid):
        writer1.add_page(reader.pages[i])
    with open(part1_path, "wb") as f1:
        writer1.write(f1)
        
    writer2 = PdfWriter()
    for i in range(mid, total_pages):
        writer2.add_page(reader.pages[i])
    with open(part2_path, "wb") as f2:
        writer2.write(f2)

def extract_text_with_fallback(pdf_path, filename):
    """Attempts to extract text. If GPU OOM occurs, splits file in half and retries."""
    try:
        return _run_marker_sync(pdf_path)
        
    except Exception as e:
        error_msg = str(e).lower()
        if "memory" in error_msg or "cuda" in error_msg or "allocate" in error_msg:
            print(f"GPU Out of Memory detected for {filename}! Splitting file into halves...")
            torch.cuda.empty_cache()
            
            part1_path = f"temp_part1_{filename}"
            part2_path = f"temp_part2_{filename}"
            
            try:
                split_pdf_in_half(pdf_path, part1_path, part2_path)
                
                print(f"   ➔ Extracting Part 1...")
                text1 = _run_marker_sync(part1_path)
                torch.cuda.empty_cache()
                
                print(f"   ➔ Extracting Part 2...")
                text2 = _run_marker_sync(part2_path)
                torch.cuda.empty_cache()
                
                print(f"Successfully extracted and merged both halves!")
                return text1 + "\n\n" + text2
                
            finally:
                if os.path.exists(part1_path): os.remove(part1_path)
                if os.path.exists(part2_path): os.remove(part2_path)
        else:
            raise e

async def process_single_pdf(pdf_path, db, master_file, file_number, total_files):
    filename = os.path.basename(pdf_path)
    namespace = _sanitize_collection_name(filename.replace(".pdf", ""))
    temp_json_path = f"temp_{namespace}.json"

    # Skip if already processed (safe resume after crash)
    if is_already_processed(master_file, filename):
        print(f"Already processed: {filename} → skipping.")
        return True

    print(f"\n{'-'*50}\nProcessing: {filename}\n{'-'*50}")

    # 1. Extract Text
    print("Step 1: Extracting text using Marker...")
    try:
        markdown_text = extract_text_with_fallback(pdf_path, filename)
    except Exception as e:
        print(f"Error extracting text from {filename}: {e}")
        return False
        
    if not markdown_text:
        print(f"Marker returned empty text for {filename}. Skipping.")
        return False

    with open(temp_json_path, "w", encoding="utf-8") as f:
        json.dump({"text": markdown_text}, f, ensure_ascii=False, indent=2)

    # 2. Upload to ChromaDB
    print(f"Step 2: Uploading to ChromaDB (Namespace: {namespace})...")
    try:
        await db.chunk_upload_hybrid(
            filenames=[temp_json_path], 
            dense_index=None, 
            sparse_index=None, 
            namespace=namespace
        )
    except Exception as e:
        print(f"Error uploading to ChromaDB: {e}")
        if os.path.exists(temp_json_path): os.remove(temp_json_path)
        return False

    if os.path.exists(temp_json_path):
        os.remove(temp_json_path)

    # 3. GPT Extraction
    print("Step 3: Running GPT Extraction...")
    retriever = db.get_hybrid_retriever(namespace=namespace, k=20)
    
    tasks = [process_category(cat, retriever, db, 20) for cat in CATEGORIES]
    results = await asyncio.gather(*tasks, return_exceptions=True)
    
    all_rules = []
    for res in results:
        if isinstance(res, list):
            all_rules.extend(res)
        elif isinstance(res, Exception):
            print(f"Warning: A category extraction failed: {res}")

    # 4. Deduplication
    print(f"Step 4: Deduplicating {len(all_rules)} extracted rules...")
    if all_rules:
        try:
            final_rules = await smart_deduplicate_pure_token(all_rules, llm)
        except Exception as e:
            print(f"Deduplication error: {e}. Falling back to raw rules.")
            final_rules = all_rules 
    else:
        final_rules = []

    # 5. Format and Append to Master JSON
    print("Step 5: Appending to master_dataset.json...")
    formatted_output = []
    for rule in final_rules:
        if isinstance(rule, dict):
            formatted_output.append({
                "rule": rule.get("rule", ""),
                "description": rule.get("description", ""),
                "categories": rule.get("categories", [])
            })

    dataset_entry = {
        "file_number": file_number,   # REMOVE BEFORE TRAINING
        "filename": filename,         # REMOVE BEFORE TRAINING
        "instruction": "Analyze the provided tender text and extract structured clause information including rules, description, and category.",
        "input": markdown_text,       # REMOVE BEFORE TRAINING
        "output": formatted_output
    }

    append_to_json_file(master_file, dataset_entry)
    print(f"✓ Appended file {file_number} of {total_files} ({filename}) to master_dataset.json")
    print(f"Successfully finished {filename}")
    return True

async def main():
    master_file = "master_dataset.json"

    # ── START CONFIG ──────────────────────────────────────────────
    START_FROM = 264   # Change this number to resume from any file
    TOTAL_FILES = 500  # Total files in the full dataset
    # ─────────────────────────────────────────────────────────────

    device = get_cuda_device()

    # sorted() ensures consistent file numbering across runs
    all_pdf_files = sorted(glob.glob("/teamspace/studios/this_studio/Ai_Dataset/500 Files/*.pdf"))
    if not all_pdf_files:
        print("No PDF files found in the current folder. Please put your PDFs here and run again.")
        return

    # Slice to start from file number START_FROM (e.g. 265 → index 264)
    pdf_files = all_pdf_files[START_FROM - 1:]

    print(f"Total PDFs in folder : {len(all_pdf_files)}")
    print(f"Starting from file   : #{START_FROM}")
    print(f"Files to process now : {len(pdf_files)}  (#{START_FROM} → #{TOTAL_FILES})")
    print("Initializing models into VRAM (this happens only once)...")

    embed_model = HuggingFaceEmbeddings(
        model_name="intfloat/e5-large-v2", 
        model_kwargs={'device': device, 'trust_remote_code': True},
        encode_kwargs={'normalize_embeddings': True}
    )
    db = PC_Mistral(embed_model=embed_model)

    successful, failed = 0, 0

    # enumerate starts at START_FROM so idx = 265, 266, 267 … 500
    for idx, pdf_path in enumerate(pdf_files, START_FROM):
        print(f"\n==================================================")
        print(f"Processing File {idx} of {TOTAL_FILES}")
        print(f"==================================================")
        try:
            success = await process_single_pdf(
                pdf_path, db, master_file,
                file_number=idx,
                total_files=TOTAL_FILES
            )
            if success:
                successful += 1
            else:
                failed += 1
        except Exception as e:
            print(f"Fatal error processing {pdf_path}: {e}")
            failed += 1
            
        torch.cuda.empty_cache()

    print("\nBATCH PROCESSING COMPLETE!")
    print(f"Range processed : #{START_FROM} → #{TOTAL_FILES}")
    print(f"Successful      : {successful}")
    print(f"Failed          : {failed}")
    print(f"All data compiled in '{master_file}'")

if __name__ == "__main__":
    asyncio.run(main())