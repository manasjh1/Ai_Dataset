import os
import json
import asyncio
import logging
import json_repair
from typing import List, Dict, Any
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from utils.create_llm import create_azure_chat_openai
from .prompts import PROMPTS_CATEGORIES as PROMPTS  
import re
from .extract_context import get_pinecone_context
from fastapi import Request

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

router = APIRouter(prefix="/bidnobid" , tags=["Bid/No-Bid"])

llm = create_azure_chat_openai(
    azure_deployment=os.environ.get("AZURE_OPENAI_MODEL_TD", ""),
    api_version=os.environ.get("OPENAI_API_VERSION_TD", ""),
    api_key=os.environ.get("AZURE_OPENAI_KEY_TD", ""),
    temperature=0.1
)

# pc_mistral = PC_Mistral()
DENSE_INDEX_NAME = "dense"
SPARSE_INDEX_NAME = "sparse"
CATEGORIES = list(PROMPTS.keys())
MAX_RECURSION_DEPTH = 5


class EligibilityRequest(BaseModel):
    chat_id: str


def clean_json_response(text: str) -> str:
    text = text.strip()
    if text.startswith("```json"):
        text = text[7:]
    if text.endswith("```"):
        text = text[:-3]
    return text.strip()


async def extract_for_category_fast(category: str, context_chunks: str, question: str) -> List[Dict]:
    try:
        print(f"Extracting for category: {category}")
        print(f"Formatted context length for category {category}: {len(context_chunks)} characters")
        if not context_chunks:
            logger.warning(f"No context available for category {category}, skipping extraction.")
            return []
        print(f"for category {category}:Formatted context length: {len(context_chunks)} characters")
        messages = [
            ("system", PROMPTS[category]),
            ("human", f"Context:\n{context_chunks}\n\nQuestion: {question}")
        ]
        resp = await llm.ainvoke(messages)
        json_str = clean_json_response(str(resp.content))

        data = json.loads(json_str)
        rules = data if isinstance(data, list) else [data]
        logger.info(f"{category}: → {len(rules)} rules")
        return rules
    except Exception as e:
        logger.warning(f"Extract failed for {category}: {e}")
        return []


MAX_TOKENS = 118000  # Safe for GPT-4o (real limit ~128k)

def estimate_tokens(clauses: List[Dict]) -> int:
    text = json.dumps(clauses, ensure_ascii=False)
    words = len(text.split())
    chars = len(text)
    return int(words * 4.0 + chars / 3.8 + 200)  # Ultra-accurate

async def smart_deduplicate_pure_token(
    rules: List[Dict],
    llm_instance,
    _depth: int = 0
) -> List[Dict]:
    """
    TRUE token-only deduplication:
    - If total < 118k tokens → single perfect call
    - If > 118k → split into TWO HALVES → recurse → final merge
    NO fixed batch size. Ever.
    """

    if not rules:
        return []

    # Safety: stop infinite recursion
    if _depth >= MAX_RECURSION_DEPTH:
        logger.warning(f"Max recursion depth {MAX_RECURSION_DEPTH} reached, returning rules as-is")
        return rules

    total_tokens = estimate_tokens(rules)
    logger.info(f"Deduplication depth={_depth}: {len(rules)} rules → ~{total_tokens:,} tokens")

    # SINGLE CALL — BEST QUALITY
    if total_tokens <= MAX_TOKENS:
        logger.info("Single-call deduplication (perfect quality)")
        prompt = f"""
### STRICT RULES ###
- Only merge clauses **if their meaning, intent, and purpose are identical**.
- If clauses are same in meaning but differ in wording, combine them into one.
- Keep all unique clauses separate, even if they appear similar but differ in specifics.
- Preserve the exact information, terminology, and context from each clause.
- If multiple duplicates exist (same rule phrased differently), merge them into one unified version.

### OUTPUT FORMAT ###
Return strictly a JSON array.
Each clause must contain:
- "rule": The main clause text (merged if necessary)
- "description": Explanation or supporting details
- "reasoning": The logic or purpose behind the clause
- "categories": List of related categories
- "source": Original source references(if there are multiple, combine them)

INPUT ({len(rules)} rules):
{json.dumps(rules, indent=2, ensure_ascii=False)}

OUTPUT: Clean JSON array only. No markdown, no explanation, no truncation.
"""

        try:
            resp = await llm_instance.ainvoke(
                [
                    ("system", "You are Tender analysis expert. Your task is to merge only true duplicates. Never remove unique rules. Always output complete, valid JSON. Never truncate output."),
                    ("human", prompt)
                ])
            cleaned = clean_json_response(str(resp.content))

            # Try normal parse first
            try:
                result = json.loads(cleaned)
            except json.JSONDecodeError:
                # Attempt repair before giving up
                logger.warning(f"JSON parse failed, attempting repair (depth={_depth})")
                repaired = json_repair.repair_json(cleaned, return_objects=True)
                result = repaired if isinstance(repaired, List) else rules 

            final = result if isinstance(result, list) else rules
            logger.info(f"Single-call depth={_depth}: {len(rules)} → {len(final)} rules")
            return final
        except Exception as e:
            logger.warning(f"Single call failed ({e}) → falling back to split")

    # PURE TOKEN-BASED SPLIT (no fixed size)
    logger.info(f"Too many tokens → splitting by half for processing (depth={_depth})")
    mid = len(rules) // 2
    left = rules[:mid]
    right = rules[mid:]

    logger.info(f"Splitting into {len(left)} + {len(right)} rules")

    # Recursively deduplicate each half
    left_clean = await smart_deduplicate_pure_token(left, llm_instance, _depth + 1)
    right_clean = await smart_deduplicate_pure_token(right, llm_instance, _depth + 1)

    combined = left_clean + right_clean
    logger.info(f"After split+dedupe: {len(combined)} rules → doing final merge")

    # Final global merge (now fits in one call)
    return await smart_deduplicate_pure_token(combined, llm_instance, _depth + 1)

async def process_category(cat,retriever, pc_mistral, top_n):
    # Step 1: Get category-specific context
    formatted_ctx = await get_pinecone_context(
        cat,  
        retriever,
        pc_mistral,
        top_n=top_n
    )
    
    print(f"{cat} context length: {len(formatted_ctx)} characters")
    print(f"{cat} context sample: {formatted_ctx[:500]}")
    # Step 2: Extract rules for this category
    question = f"Extract all eligibility criteria relevant to {cat} from the context."
    
    return await extract_for_category_fast(
        cat,
        formatted_ctx,
        question
    )

def _sanitize_collection_name(name: str) -> str:
    # Keep only allowed characters
    name = re.sub(r'[^a-zA-Z0-9._-]', '', name)

    # Remove leading non-alphanumeric
    name = re.sub(r'^[^a-zA-Z0-9]+', '', name)

    # Remove trailing non-alphanumeric
    name = re.sub(r'[^a-zA-Z0-9]+$', '', name)

    # Ensure minimum length
    if len(name) < 3:
        name = f"col_{name}"

    return name


@router.post("/extract-clauses")
async def extract_clauses(request: EligibilityRequest,
                          req: Request):
    try:
        namespace = f"{request.chat_id}_test"
        pc_mistral = req.app.state.pc_mistral

        retriever = pc_mistral.get_hybrid_retriever(
            namespace=_sanitize_collection_name(namespace),
            k=20
        )

        start = asyncio.get_event_loop().time()
        
        tasks = [process_category(cat, retriever, pc_mistral, 20) for cat in CATEGORIES]
        results = await asyncio.gather(*tasks)
        all_rules = [r for sublist in results for r in sublist]
        print(f"Total time for extraction before deduplication: {asyncio.get_event_loop().time() - start:.2f}s")
        # HYBRID DEDUPLICATION
        final_rules = await smart_deduplicate_pure_token(all_rules, llm)

        total_time = asyncio.get_event_loop().time() - start
        logger.info(f"TOTAL TIME: {total_time:.2f}s | Rules: {len(all_rules)} → {len(final_rules)}")
        print(f"TOTAL TIME: {total_time:.2f}s | Rules: {len(all_rules)} → {len(final_rules)}")

        return {   
            "namespace": namespace,
            "processing_time_seconds": round(total_time, 2),
            "raw_rules_count": len(all_rules),
            "final_rules_count": len(final_rules),
            "eligibility_criteria": final_rules
        }

    except Exception as e:
        logger.error(f"Extraction failed: {str(e)}")
        raise HTTPException(500, str(e))