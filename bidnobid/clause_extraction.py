import os
import json
import asyncio
import logging
import re
from typing import List, Dict, Any
from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel

from utils.create_llm import create_azure_chat_openai
from .prompts import PROMPTS_CATEGORIES as PROMPTS
from .extract_context import get_pinecone_context

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

router = APIRouter(prefix="/bidnobid", tags=["Bid/No-Bid"])

# Linter Fix: Added ' , "" ' to ensure it always returns a string, not None
llm = create_azure_chat_openai(
    azure_deployment=os.environ.get("AZURE_OPENAI_MODEL_TD", ""),
    api_version=os.environ.get("OPENAI_API_VERSION_TD", ""),
    api_key=os.environ.get("AZURE_OPENAI_KEY_TD", ""),
    temperature=0.1
)

CATEGORIES = list(PROMPTS.keys())
MAX_TOKENS = 118000


class EligibilityRequest(BaseModel):
    chat_id: str
    # No user_id — auth removed


# =========================================================================
# HELPERS
# =========================================================================

# Linter Fix: Added 'text: Any' to accept both strings and Langchain message lists
def clean_json_response(text: Any) -> str:
    """Handle str | list[str | dict] returned by resp.content."""
    if isinstance(text, list):
        # Extract text blocks from content list (AIMessage content format)
        text = "".join(
            block.get("text", "") if isinstance(block, dict) else str(block)
            for block in text
        )
    text = str(text).strip()
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


# =========================================================================
# CATEGORY EXTRACTION
# =========================================================================

async def extract_for_category_fast(
    category: str, context_chunks: str, question: str
) -> List[Dict]:
    try:
        print(f"Extracting for category: {category}")

        if not context_chunks:
            logger.warning(f"No context for '{category}', skipping.")
            return []

        print(f"Context length for '{category}': {len(context_chunks)} characters")

        messages = [
            ("system", PROMPTS[category]),
            ("human", f"Context:\n{context_chunks}\n\nQuestion: {question}")
        ]
        resp = await llm.ainvoke(messages)
        
        # Linter error resolved: clean_json_response now accepts Any type
        json_str = clean_json_response(resp.content)
        
        data = json.loads(json_str)
        rules = data if isinstance(data, list) else [data]
        logger.info(f"'{category}': {len(rules)} rules extracted")
        return rules

    except Exception as e:
        logger.warning(f"Extraction failed for '{category}': {e}")
        return []


# =========================================================================
# DEDUPLICATION
# =========================================================================

async def smart_deduplicate_pure_token(
    rules: List[Dict], llm_instance
) -> List[Dict]:
    """
    Token-budget deduplication:
    - Fits in MAX_TOKENS → single perfect LLM call
    - Too large → split into halves recursively → final merge
    """
    
    # BASE CASE: Prevents infinite loop
    if len(rules) <= 1:
        return rules

    total_tokens = estimate_tokens(rules)
    logger.info(f"Deduplication: {len(rules)} rules → ~{total_tokens:,} tokens")

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
- "source": Original source references (if multiple, combine them)

INPUT ({len(rules)} rules):
{json.dumps(rules, indent=2, ensure_ascii=False)}

OUTPUT: Clean JSON array only. Same format.
"""
        try:
            resp = await llm_instance.ainvoke([
                ("system", "You are a Tender analysis expert. Merge only true duplicates. Never remove unique rules. STRICTLY use double quotes for JSON properties and escape internal quotes."),
                ("human", prompt)
            ])
            cleaned = clean_json_response(resp.content)
            result = json.loads(cleaned)
            final = result if isinstance(result, list) else rules
            logger.info(f"Single-call dedupe: {len(rules)} → {len(final)} rules")
            return final
        except Exception as e:
            logger.warning(f"Single-call dedupe failed ({e}) → splitting")

    # Token-based recursive split
    logger.info("Token budget exceeded or parsing failed → splitting into halves")
    mid = len(rules) // 2
    left, right = rules[:mid], rules[mid:]
    logger.info(f"Split: {len(left)} + {len(right)} rules")

    left_clean = await smart_deduplicate_pure_token(left, llm_instance)
    right_clean = await smart_deduplicate_pure_token(right, llm_instance)

    combined = left_clean + right_clean
    logger.info(f"After half-dedupe: {len(combined)} rules → final merge")

    return await smart_deduplicate_pure_token(combined, llm_instance)


# =========================================================================
# PER-CATEGORY PIPELINE
# =========================================================================

# Linter Fix: Added '-> List[Dict]' return type so the linter knows it is iterable
async def process_category(cat: str, retriever, pc_mistral, top_n: int) -> List[Dict]:
    formatted_ctx = await get_pinecone_context(cat, retriever, pc_mistral, top_n=top_n)

    print(f"{cat} context length: {len(formatted_ctx)} characters")
    print(f"{cat} context sample: {formatted_ctx[:500]}")

    question = f"Extract all eligibility criteria relevant to {cat} from the context."

    return await extract_for_category_fast(cat, formatted_ctx, question)


# =========================================================================
# ENDPOINT
# =========================================================================

@router.post("/extract-clauses")
async def extract_clauses(request: EligibilityRequest, req: Request):
    """
    Extract and deduplicate eligibility clauses from tender documents.
    No DB, no auth — pass chat_id only.
    """
    try:
        namespace = f"{request.chat_id}_test"
        pc_mistral = req.app.state.pc_mistral

        retriever = pc_mistral.get_hybrid_retriever(
            namespace=_sanitize_collection_name(namespace),
            k=20
        )

        start = asyncio.get_event_loop().time()

        # Parallel extraction across all categories
        tasks = [process_category(cat, retriever, pc_mistral, 20) for cat in CATEGORIES]
        results = await asyncio.gather(*tasks)
        
        # Because we added -> List[Dict] to process_category, this will no longer throw an iter error
        all_rules = [r for sublist in results for r in sublist]

        print(f"Extraction done in {asyncio.get_event_loop().time() - start:.2f}s | {len(all_rules)} raw rules")

        final_rules = await smart_deduplicate_pure_token(all_rules, llm)

        total_time = asyncio.get_event_loop().time() - start
        logger.info(f"TOTAL: {total_time:.2f}s | {len(all_rules)} → {len(final_rules)} rules")
        print(f"TOTAL: {total_time:.2f}s | {len(all_rules)} → {len(final_rules)} rules")

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