import asyncio
import json
import os
import torch
from processor_app.pc_db import PC_Mistral
from langchain_huggingface import HuggingFaceEmbeddings
from utils.parse_to_mistral import _run_marker_sync

# --- 1. Detect the Best Available GPU ---
def get_device():
    if torch.cuda.is_available():
        print("🚀 GPU Detected: Using NVIDIA CUDA")
        return "cuda"
    elif torch.backends.mps.is_available():
        print("🍏 Apple Silicon Detected: Using Metal Performance Shaders (MPS)")
        return "mps"
    else:
        print("⚠️ No GPU Detected: Falling back to CPU")
        return "cpu"

async def main():
    pdf_path = "1707222063_Vol-1 (2)-31-32 copy.pdf"  # Replace with a real PDF in your folder
    temp_json_path = "temp_doc.json"
    namespace = "test_tender_namespace"
    device = get_device()

    if not os.path.exists(pdf_path):
        print(f"❌ Error: Please put a file named '{pdf_path}' in this folder.")
        return

    # --- 2. Extract Text using Marker (GPU) ---
    print(f"\n📄 Step 1: Extracting text from {pdf_path} using Marker...")
    markdown_text = _run_marker_sync(pdf_path)
    
    if not markdown_text:
        print("❌ Error: Failed to extract text from PDF.")
        return
        
    print(f"✅ Extracted {len(markdown_text)} characters of text!")

    # --- 3. Save to JSON format ---
    print("\n💾 Step 2: Saving text to temporary JSON format...")
    with open(temp_json_path, "w", encoding="utf-8") as f:
        json.dump({"text": markdown_text}, f, ensure_ascii=False, indent=2)

    # --- 4. Initialize Database & Embeddings (GPU) ---
    print("\n🧠 Step 3: Initializing HuggingFace Embeddings and ChromaDB...")
    embed_model = HuggingFaceEmbeddings(
        model_name="intfloat/e5-large-v2", 
        model_kwargs={'device': device, 'trust_remote_code': True},
        encode_kwargs={'normalize_embeddings': True}
    )
    db = PC_Mistral(embed_model=embed_model)

    # --- 5. Upload to ChromaDB ---
    print(f"\n⬆️ Step 4: Uploading and chunking into ChromaDB (Namespace: {namespace})...")
    total_chunks = await db.chunk_upload_hybrid(
        filenames=[temp_json_path], 
        dense_index=None, 
        sparse_index=None, 
        namespace=namespace
    )
    
    print(f"\n🎉 SUCCESS! {total_chunks} chunks were processed and saved to ChromaDB!")

    # Clean up the temporary JSON file
    if os.path.exists(temp_json_path):
        os.remove(temp_json_path)

if __name__ == "__main__":
    asyncio.run(main())