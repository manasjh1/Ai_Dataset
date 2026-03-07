import os
import gc
import logging
import asyncio
from collections import Counter
import torch

from marker.converters.pdf import PdfConverter
from marker.models import create_model_dict
from marker.output import text_from_rendered

# Add this dummy semaphore so other files don't crash when trying to import it.
_ocr_semaphore = asyncio.Semaphore(4)

# --- Marker Global ---
_CONVERTER = None

def get_converter():
    """Lazy initialization of Marker models to load into memory only once."""
    global _CONVERTER
    if _CONVERTER is None:
        logging.info("Loading Marker models into memory...")
        
        # Optimize for Apple Silicon (M4 Mac) or NVIDIA GPUs
        if torch.cuda.is_available():
            device = "cuda"
            logging.info("Accelerating Marker with NVIDIA CUDA...")
        elif torch.backends.mps.is_available():
            device = "mps"
            logging.info("Apple Silicon detected. Accelerating Marker with MPS...")
        else:
            device = "cpu"
            logging.info("No GPU detected. Running Marker on CPU...")
            
        # Create models and initialize the new v1.x PdfConverter
        artifact_dict = create_model_dict(device=device)
        _CONVERTER = PdfConverter(artifact_dict=artifact_dict)
        
    return _CONVERTER

def _run_marker_sync(pdf_path: str) -> str:
    """High-quality PDF to Markdown execution using Marker v1.x API."""
    logging.info(f"Starting Marker for: {pdf_path}")
    
    try:
        converter = get_converter()
        
        # Render the PDF
        rendered = converter(pdf_path)
        
        # Extract just the markdown text (ignoring images/metadata for now)
        full_text, _, _ = text_from_rendered(rendered)
        
        logging.info(f"Successfully processed {pdf_path} with Marker.")
        return full_text
        
    except Exception as e:
        logging.error(f"Failed to process PDF with Marker: {e}")
        return ""

def free_memory():
    """Forces garbage collection to clear up RAM/VRAM."""
    gc.collect()

def clean_repetitive_text(text, max_total_repeats=50, keep_first=10, max_ngram=10):
    """Cleans up recurring headers/footers in the parsed document."""
    if not text:
        return ""
        
    words = text.split()
    n = len(words)

    phrase_counter = Counter()
    phrase_indices = {}

    for k in range(1, max_ngram + 1):
        for i in range(n - k + 1):
            phrase = ' '.join(words[i:i + k])
            phrase_counter[phrase] += 1
            phrase_indices.setdefault(phrase, []).append(i)

    to_remove = set()
    for phrase, count in phrase_counter.items():
        if count > max_total_repeats:
            for idx in phrase_indices[phrase][keep_first:]:
                to_remove.update(range(idx, idx + len(phrase.split())))

    cleaned_words = [word for i, word in enumerate(words) if i not in to_remove]
    return ' '.join(cleaned_words)