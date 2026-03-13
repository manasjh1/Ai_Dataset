import asyncio
import json
import os
import torch
import glob
import re
from dotenv import load_dotenv

# Load environment variables for Azure OpenAI keys
load_dotenv()

from processor_app.pc_db import PC_Mistral
from langchain_huggingface import HuggingFaceEmbeddings
from utils.parse_to_mistral import _run_marker_sync
from bidnobid.clause_extraction import process_category, smart_deduplicate_pure_token, CATEGORIES, llm


# ── CONFIG ────────────────────────────────────────────────────────────────────
PDF_FOLDER  = "./downloaded_pdfs"     # Folder containing all PDFs to process
MASTER_FILE = "master_dataset.json"  # Output file — auto-resumes from here
TOTAL_FILES = None                   

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


def get_last_processed_file_number(master_file):
    """
    Reads master_dataset.json and returns the next file number to process.
    Returns None if the file doesn't exist or is empty (fresh start).
    """
    if not os.path.exists(master_file):
        return None
    try:
        with open(master_file, "r", encoding="utf-8") as f:
            data = json.load(f)
        if not data:
            return None
        last_file_number = max(entry.get("file_number", 0) for entry in data)
        print(f"[Auto-Resume] Last completed file: #{last_file_number}. Resuming from #{last_file_number + 1}.")
        return last_file_number + 1
    except Exception as e:
        print(f"[Auto-Resume] Could not read master file ({e}). Starting from file #1.")
        return None


def extract_text(pdf_path, filename):
    """Extracts text from PDF using Marker."""
    try:
        return _run_marker_sync(pdf_path)
    except Exception as e:
        print(f"Error extracting text from {filename}: {e}")
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
        markdown_text = extract_text(pdf_path, filename)
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
        if os.path.exists(temp_json_path):
            os.remove(temp_json_path)
        return False

    if os.path.exists(temp_json_path):
        os.remove(temp_json_path)

    # 3. GPT Extraction
    print("Step 3: Running GPT Extraction...")
    retriever = db.get_hybrid_retriever(namespace=namespace, k=20)

    tasks = [process_category(cat, retriever, db, 20) for cat in CATEGORIES]
    results = await asyncio.gather(*tasks, return_exceptions=True)

    all_rules = []  # ✅ FIXED: Proper indentation
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
    device = get_cuda_device()

    # ── Discover all PDFs in the folder ──────────────────────────────────────
    all_pdf_files = sorted(glob.glob(os.path.join(PDF_FOLDER, "*.pdf")))

    if not all_pdf_files:
        print(f"No PDF files found in '{PDF_FOLDER}'. Please add PDFs and run again.")
        return

    total_files = TOTAL_FILES if TOTAL_FILES is not None else len(all_pdf_files)
    all_pdf_files = all_pdf_files[:total_files]  # cap if TOTAL_FILES is set

    print(f"PDF folder       : {PDF_FOLDER}")
    print(f"Total PDFs found : {len(all_pdf_files)}")

    # ── Auto-detect resume point ──────────────────────────────────────────────
    auto_start  = get_last_processed_file_number(MASTER_FILE)
    start_index = (auto_start - 1) if auto_start is not None else 0  # convert to 0-based index

    if start_index >= len(all_pdf_files):
        print("All files have already been processed. Nothing to do.")
        return

    pdf_files_to_process = all_pdf_files[start_index:]
    start_from_number    = start_index + 1  # human-readable 1-based file number

    print(f"Resuming from    : #{start_from_number}")
    print(f"Files remaining  : {len(pdf_files_to_process)}")
    print("Initializing models into VRAM (this happens only once)...")

    # ── Load models once ──────────────────────────────────────────────────────
    embed_model = HuggingFaceEmbeddings(
        model_name="intfloat/e5-large-v2",
        model_kwargs={'device': device, 'trust_remote_code': True},
        encode_kwargs={'normalize_embeddings': True}
    )
    db = PC_Mistral(embed_model=embed_model)

    successful, failed = 0, 0

    for idx, pdf_path in enumerate(pdf_files_to_process, start=start_from_number):  # ✅ FIXED: Proper indentation
        print(f"\n{'='*50}")
        print(f"Processing File {idx} of {total_files}")
        print(f"{'='*50}")
        try:
            success = await process_single_pdf(
                pdf_path, db, MASTER_FILE,
                file_number=idx,
                total_files=total_files
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
    print(f"Range processed  : #{start_from_number} → #{total_files}")
    print(f"Successful       : {successful}")
    print(f"Failed           : {failed}")
    print(f"Output saved to  : '{MASTER_FILE}'")


if __name__ == "__main__":
    asyncio.run(main())
