from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from langchain_huggingface import HuggingFaceEmbeddings
from processor_app.pc_db import PC_Mistral
import os
import logging
import sys
from bidnobid.clause_extraction import router as clauses_router

# Configure Logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s")
stream_handler = logging.StreamHandler(sys.stdout)
stream_handler.setFormatter(formatter)
logger.addHandler(stream_handler)

app = FastAPI(title="AI Extraction Microservice")

# Initialize Embeddings & Local ChromaDB
embed_model = HuggingFaceEmbeddings(
    model_name="intfloat/e5-large-v2",
    model_kwargs={"trust_remote_code": True, 'device': 'cpu'},
    encode_kwargs={'normalize_embeddings': True}
)
app.state.pc_mistral = PC_Mistral(embed_model=embed_model)

# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  
    allow_credentials=True,
    allow_methods=["*"], 
    allow_headers=["*"], 
)

# Include only the AI extraction router
app.include_router(clauses_router)

@app.get("/")
def read_root():
    return {"message": "AI Extraction Microservice is running successfully."}