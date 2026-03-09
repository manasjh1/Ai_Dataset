import asyncio
import json
import os
import torch
import time
import logging
from dotenv import load_dotenv

# Load environment variables for Azure OpenAI keys
load_dotenv()

from processor_app.pc_db import PC_Mistral
from langchain_huggingface import HuggingFaceEmbeddings
from utils.parse_to_mistral import _run_marker_sync

# --- Import the GPT extraction logic ---
from bidnobid.clause_extraction import process_category, smart_deduplicate_pure_token, CATEGORIES, llm

# Setup basic logging for the script
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- 1. Detect the Best Available GPU ---
def get_device():
    if torch.cuda.is_available():
        print("GPU Detected: Using NVIDIA CUDA")
        return "cuda"
    elif torch.backends.mps.is_available():
        print("Apple Silicon Detected: Using Metal Performance Shaders (MPS)")
        return "mps"
    else:
        print("No GPU Detected: Falling back to CPU")
        return "cpu"

async def main():
    pdf_path = "2024_LSGD_706572_18_NITLIC.pdf"  
    temp_json_path = "temp_doc.json"
    namespace = "test_tender_namespace"
    device = get_device()

    if not os.path.exists(pdf_path):
        print(f"Error: Please put a file named '{pdf_path}' in this folder.")
        return

    print(f"\nStep 1: Extracting text from {pdf_path} using Marker...")
    markdown_text = _run_marker_sync(pdf_path)
    
    if not markdown_text:
        print("Error: Failed to extract text from PDF.")
        return
        
    print(f"Extracted {len(markdown_text)} characters of text!")

    # --- 3. Save to JSON format ---
    print("\nStep 2: Saving text to temporary JSON format...")
    with open(temp_json_path, "w", encoding="utf-8") as f:
        json.dump({"text": markdown_text}, f, ensure_ascii=False, indent=2)

    # --- 4. Initialize Database & Embeddings (GPU) ---
    print("\nStep 3: Initializing HuggingFace Embeddings and ChromaDB...")
    embed_model = HuggingFaceEmbeddings(
        model_name="intfloat/e5-large-v2", 
        model_kwargs={'device': device, 'trust_remote_code': True},
        encode_kwargs={'normalize_embeddings': True}
    )
    db = PC_Mistral(embed_model=embed_model)

    # --- 5. Upload to ChromaDB ---
    print(f"\nStep 4: Uploading and chunking into ChromaDB (Namespace: {namespace})...")
    total_chunks = await db.chunk_upload_hybrid(
        filenames=[temp_json_path], 
        dense_index=None, 
        sparse_index=None, 
        namespace=namespace
    )
    
    print(f"SUCCESS! {total_chunks} chunks were processed and saved to ChromaDB!")

    # Clean up the temporary JSON file
    if os.path.exists(temp_json_path):
        os.remove(temp_json_path)

    
    print(f"\nStep 5: Initializing Retriever for namespace '{namespace}'...")
    retriever = db.get_hybrid_retriever(namespace=namespace, k=20)

    print("\nStep 6: Running GPT-based Clause Extraction (This may take a minute)...")
    print(f"Categories being processed: {CATEGORIES}")
    
    start_time = time.time()
    
    # Process all categories concurrently
    tasks = [process_category(cat, retriever, db, 20) for cat in CATEGORIES]
    results = await asyncio.gather(*tasks)
    
    # Safely flatten the results into a single list (protects against None returns)
    all_rules = []
    for sublist in results:
        if isinstance(sublist, list):
            all_rules.extend(sublist)
            
    print(f"Extracted {len(all_rules)} raw rules across all categories.")

    print("\nStep 7: Deduplicating and merging extracted rules using LLM...")
    if all_rules:
        final_rules = await smart_deduplicate_pure_token(all_rules, llm)
    else:
        final_rules = []

    total_time = time.time() - start_time
    print(f"\nEXTRACTION COMPLETE! Found {len(final_rules)} unique rules in {total_time:.2f} seconds.")

    # --- 8. Save the final JSON Output (Dataset Format) ---
    print("\nStep 8: Formatting and saving to dataset structure...")
    
    # Map the final rules to ensure they contain all rich dataset features
    formatted_output = []
    for rule in final_rules:
        formatted_output.append({
            "rule": rule.get("rule", ""),
            "description": rule.get("description", ""),
            "reasoning": rule.get("reasoning", ""),
            # Safely capture category/categories as LLMs sometimes switch singular/plural
            "categories": rule.get("categories", rule.get("category", [])),
            "source": rule.get("source", "")
        })

    # Create the dataset payload
    dataset_payload = {
        "instruction": "Analyze the provided tender text and extract structured clause information including rules, description, and category.",
        "input": markdown_text,  # This is the raw extracted text from Marker
        "output": formatted_output # This is the LLM result
    }

    output_file = "dataset_formatted_output.json"
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(dataset_payload, f, ensure_ascii=False, indent=4)
        
    print(f"Saved dataset-formatted JSON to '{output_file}'\n")

if __name__ == "__main__":
    asyncio.run(main())