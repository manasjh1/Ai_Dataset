import os
import json
import asyncio
import logging
import re
from typing import List, Dict
from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel
from utils.create_llm import create_azure_chat_openai
from .prompts import PROMPTS_CATEGORIES as PROMPTS  
from .extract_context import get_pinecone_context

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/bidnobid", tags=["Bid/No-Bid"])

# Initialize LLM
llm = create_azure_chat_openai(
    azure_deployment=os.environ.get("AZURE_OPENAI_MODEL_TD"),
    api_version=os.environ.get("OPENAI_API_VERSION_TD"),
    api_key=os.environ.get("AZURE_OPENAI_KEY_TD"),
    temperature=0.1
)

CATEGORIES = list(PROMPTS.keys())
MAX_TOKENS = 118000  

class EligibilityRequest(BaseModel):
    chat_id: str

def clean_json_response(text: str) -> str:
    text = text.strip()
    if text.startswith("```json"):
        text = text[7:]
    if text.endswith("```"):
        text = text[:-3]
    return text.strip()

def _sanitize_collection_name(name: str) -> str:
    name = re.sub(r'[^a-zA-Z0-9._-]', '', name)
    name = re.sub(r'^[^a-zA-Z0-9]+', '', name)
    name = re.sub(r'[^a-zA-Z0-9]+$', '', name)
    if len(name) < 3:
        name = f"col_{name}"
    return name

def estimate_tokens(clauses: List[Dict]) -> int:
    text = json.dumps(clauses, ensure_ascii=False)
    words = len(text.split())
    chars = len(text)
    return int(words * 4.0 + chars / 3.8 + 200)

async def smart_deduplicate_pure_token(rules: List[Dict], llm_instance) -> List[Dict]:
    total_tokens = estimate_tokens(rules)
    if total_tokens <= MAX_TOKENS:
        prompt = f"""
        ### STRICT RULES ###
        - Merge clauses only if meaning/intent are identical.
        - Output JSON array with keys: rule, description, reasoning, categories, source.
        INPUT: {json.dumps(rules, ensure_ascii=False)}
        """
        try:
            resp = await llm_instance.ainvoke([
                ("system", "You are a Tender analysis expert. Merge true duplicates. Output JSON only."),
                ("human", prompt)
            ])
            return json.loads(clean_json_response(resp.content))
        except Exception as e:
            logger.warning(f"Single call failed ({e}) → falling back to split")

    mid = len(rules) // 2
    left_clean = await smart_deduplicate_pure_token(rules[:mid], llm_instance)
    right_clean = await smart_deduplicate_pure_token(rules[mid:], llm_instance)
    return await smart_deduplicate_pure_token(left_clean + right_clean, llm_instance)

async def process_category(cat, retriever, pc_mistral, top_n):
    formatted_ctx = await get_pinecone_context(cat, retriever, pc_mistral, top_n=top_n)
    if not formatted_ctx:
        return []
    
    question = f"Extract all eligibility criteria relevant to {cat} from the context."
    messages = [
        ("system", PROMPTS[cat]),
        ("human", f"Context:\n{formatted_ctx}\n\nQuestion: {question}")
    ]
    try:
        resp = await llm.ainvoke(messages)
        data = json.loads(clean_json_response(resp.content))
        return data if isinstance(data, list) else [data]
    except Exception as e:
        logger.warning(f"Extract failed for {cat}: {e}")
        return []

@router.post("/extract-clauses")
async def extract_clauses(request: EligibilityRequest, req: Request):
    try:
        namespace = f"{request.chat_id}_test"
        pc_mistral = req.app.state.pc_mistral
        
        retriever = pc_mistral.get_hybrid_retriever(
            namespace=_sanitize_collection_name(namespace),
            k=20
        )

        start = asyncio.get_event_loop().time()
        
        # Process all categories in parallel
        tasks = [process_category(cat, retriever, pc_mistral, 20) for cat in CATEGORIES]
        results = await asyncio.gather(*tasks)
        all_rules = [r for sublist in results for r in sublist]
        
        # Deduplicate
        final_rules = await smart_deduplicate_pure_token(all_rules, llm)

        total_time = asyncio.get_event_loop().time() - start
        
        return {   
            "namespace": namespace,
            "processing_time_seconds": round(total_time, 2),
            "final_rules_count": len(final_rules),
            "eligibility_criteria": final_rules
        }

    except Exception as e:
        logger.error(f"Extraction failed: {str(e)}")
        raise HTTPException(500, str(e))