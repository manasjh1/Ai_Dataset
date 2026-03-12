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
    """Download random PDFs from S3 - first collects PDF list, then randomly samples"""
    
    access_key = os.getenv('AWS_ACCESS_KEY_ID')
    secret_key = os.getenv('AWS_SECRET_ACCESS_KEY')
    region = os.getenv('AWS_REGION', 'us-east-1')
    
    if not access_key or not secret_key:
        print("ERROR: AWS credentials not found in .env file!")
        print("Please add AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY to .env")
        return
    
    s3 = boto3.client('s3',
        aws_access_key_id=access_key,
        aws_secret_access_key=secret_key,
        region_name=region
    )
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    print(f"Bucket: {BUCKET_NAME}")
    print(f"Collecting PDF file list...\n")
    
    paginator = s3.get_paginator('list_objects_v2')
    pages = paginator.paginate(Bucket=BUCKET_NAME, Prefix=PREFIX)
    
    all_pdfs = []
    for page in pages:
        if 'Contents' not in page:
            continue
            
        for obj in page['Contents']:
            key = obj['Key']
            if key.lower().endswith('.pdf'):
                all_pdfs.append(key)
        
        if len(all_pdfs) % 1000 == 0:
            print(f"Found {len(all_pdfs)} PDFs so far...")
    
    print(f"\n✓ Total PDFs found: {len(all_pdfs)}")
    
    if len(all_pdfs) == 0:
        print("No PDF files found in bucket!")
        return
    
    if RANDOM_SEED is not None:
        random.seed(RANDOM_SEED)
        print(f"Using random seed: {RANDOM_SEED} (reproducible)")
    
    download_count = min(DOWNLOAD_COUNT, len(all_pdfs))
    random_pdfs = random.sample(all_pdfs, download_count)
    
    print(f"Randomly selected {download_count} PDFs to download\n")
    
    downloaded = 0
    skipped = 0
    failed = 0
    
    for idx, key in enumerate(random_pdfs, 1):
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
    print(f"Download Complete!")
    print(f"Total selected: {download_count}")
    print(f"Downloaded: {downloaded}")
    print(f"Skipped (existing): {skipped}")
    print(f"Failed: {failed}")
    print(f"{'='*60}")


if __name__ == "__main__":
    download_random_pdfs()
