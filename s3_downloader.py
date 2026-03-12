import os
import boto3
from dotenv import load_dotenv

load_dotenv()

BUCKET_NAME = "extracted-documents-orange"
PREFIX = ""  
OUTPUT_DIR = "./downloaded_pdfs"

SKIP_FIRST = 700  
DOWNLOAD_COUNT = 500  



def download_pdfs():
    """Download PDFs with skip and count logic - streams through S3 without listing all files"""
    
    # Get credentials from .env
    access_key = os.getenv('AWS_ACCESS_KEY_ID')
    secret_key = os.getenv('AWS_SECRET_ACCESS_KEY')
    region = os.getenv('AWS_REGION', 'us-east-1')
    
    if not access_key or not secret_key:
        print("ERROR: AWS credentials not found in .env file!")
        print("Please add AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY to .env")
        return
    
    # Initialize S3
    s3 = boto3.client('s3',
        aws_access_key_id=access_key,
        aws_secret_access_key=secret_key,
        region_name=region
    )
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    print(f"Bucket: {BUCKET_NAME}")
    print(f"Skipping first {SKIP_FIRST} PDF files...")
    print(f"Then downloading {DOWNLOAD_COUNT} PDF files")
    print(f"Output: {OUTPUT_DIR}\n")
    
    paginator = s3.get_paginator('list_objects_v2')
    pages = paginator.paginate(Bucket=BUCKET_NAME, Prefix=PREFIX)
    
    pdf_count = 0  
    downloaded = 0
    skipped = 0
    
    for page in pages:
        if 'Contents' not in page:
            continue
            
        for obj in page['Contents']:
            key = obj['Key']
            
            # Only process PDF files
            if not key.lower().endswith('.pdf'):
                continue
            
            pdf_count += 1
            
            # Skip first N PDF files
            if pdf_count <= SKIP_FIRST:
                if pdf_count % 100 == 0:
                    print(f"Skipping PDFs... {pdf_count}/{SKIP_FIRST}")
                continue
            
            # Stop after downloading required count
            if downloaded >= DOWNLOAD_COUNT:
                print(f"\n✓ Downloaded {downloaded} PDF files. Done!")
                return
            
            # Download PDF file
            filename = os.path.basename(key)
            local_path = os.path.join(OUTPUT_DIR, filename)
            
            if os.path.exists(local_path):
                skipped += 1
                print(f"[PDF #{pdf_count}] SKIP (exists): {filename}")
                continue
            
            try:
                s3.download_file(BUCKET_NAME, key, local_path)
                downloaded += 1
                print(f"[PDF #{pdf_count}] ✓ Downloaded ({downloaded}/{DOWNLOAD_COUNT}): {filename}")
            except Exception as e:
                print(f"[PDF #{pdf_count}] ✗ FAILED: {filename} - {e}")
    
    print(f"\n✓ Finished! Downloaded: {downloaded}, Skipped: {skipped}")


if __name__ == "__main__":
    download_pdfs()