import os
import boto3
import random
from dotenv import load_dotenv

load_dotenv()
BUCKET_NAME = "extracted-documents-orange"
PREFIX = ""

OUTPUT_DIR = "./downloaded_pdfs"
DOWNLOAD_COUNT = 500

RANDOM_SEED = None

def download_random_pdfs():
    """Download and strongly shuffle PDFs from S3 into a single folder."""
    access_key = os.getenv('AWS_ACCESS_KEY_ID')
    secret_key = os.getenv('AWS_SECRET_ACCESS_KEY')
    region = os.getenv('AWS_REGION', 'us-east-1')

    if not access_key or not secret_key:
        print("ERROR: AWS credentials not found in .env file!")
        return

    s3 = boto3.client('s3',
        aws_access_key_id=access_key,
        aws_secret_access_key=secret_key,
        region_name=region
    )

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print(f"Bucket: {BUCKET_NAME}")
    print("Collecting PDF file list...\n")

    paginator = s3.get_paginator('list_objects_v2')
    
    # List comprehension for faster gathering
    all_pdfs = [
        obj['Key'] 
        for page in paginator.paginate(Bucket=BUCKET_NAME, Prefix=PREFIX) 
        if 'Contents' in page 
        for obj in page['Contents'] 
        if obj['Key'].lower().endswith('.pdf')
    ]

    print(f"✓ Total PDFs found: {len(all_pdfs)}\n")

    if not all_pdfs:
        print("No PDF files found in bucket!")
        return

    # 1. SHUFFLE (Using SystemRandom for higher entropy)
    if RANDOM_SEED is not None:
        random.seed(RANDOM_SEED)
        print(f"Using random seed: {RANDOM_SEED} (reproducible)")
        random.shuffle(all_pdfs) 
    else:
        # Cryptographically strong shuffle for maximum randomness
        crypto_random = random.SystemRandom()
        crypto_random.shuffle(all_pdfs)

    # 2. SELECT
    download_count = min(DOWNLOAD_COUNT, len(all_pdfs))
    selected_pdfs = all_pdfs[:download_count]
    
    print(f"Randomly selected {download_count} PDFs to download\n")

    # 3. DOWNLOAD TO SINGLE FOLDER
    downloaded = 0
    skipped = 0
    failed = 0

    for idx, key in enumerate(selected_pdfs, 1):
        filename = os.path.basename(key)
        local_path = os.path.join(OUTPUT_DIR, filename)
        
        if os.path.exists(local_path):
            skipped += 1
            print(f"[{idx}/{download_count}] SKIP (exists): {filename}")
            continue
        
        try:
            s3.download_file(BUCKET_NAME, key, local_path)
            downloaded += 1
            print(f"[{idx}/{download_count}] ✓ Downloaded: {filename}")
        except Exception as e:
            failed += 1
            print(f"[{idx}/{download_count}] ✗ FAILED: {filename} - {e}")

    print(f"\n{'='*60}")
    print("Download Complete!")
    print(f"Total selected: {download_count}")
    print(f"Downloaded: {downloaded}")
    print(f"Skipped (existing): {skipped}")
    print(f"Failed: {failed}")
    print(f"{'='*60}")

if __name__ == "__main__":
    download_random_pdfs()
