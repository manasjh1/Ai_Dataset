from .prompts import RETRIEVAL_QUERIES
import logging
from .generate_questions import dedup_contexts

async def get_pinecone_context(category_name: str, retriever, pc, top_n=20):
    print(f"Received category: {category_name}", flush=True)
    
    # Map category to specific retrieval intent
    question_map = {
        "technical eligibility evaluation": "What is the eligibility criteria , evaluation criteria for this tender?completed works, emd (Earnest Money Deposit), company certifications and registrations, bidder eligibility criteria, eligibility conditions, qualification requirements, technical eligibility, technical qualification, experience criteria, financial eligibility, financial qualification,annual turnover, net worth, bid capacity, working capital, solvency / bank solvency , legal eligibility, statutory compliance, registration requirements, disqualification conditions, who is eligible to bid, pre-qualification requirements ,Joint Venture (JV) information",
        "financial documents": "Extract all Joint Venture (JV) information from the tender document. Answer 'Yes' or 'No' if JVs are allowed, then list all JV details.",
        "risk assessment policies": "What are the payment terms mentioned in the tender?",
        "company registration documents": "Analyze the provided tender document and extract the complete 'Scope of Work' for the bidder.",
        "keyexperts and technical staff": "What are the key experts and technical staff requirements mentioned in the tender document?"
    }
    
    # Safely get the mapped question, or fallback to the category name if it's new/unmapped
    question = question_map.get(category_name.lower(), category_name)
    
    # Append keywords safely
    keywords = RETRIEVAL_QUERIES.get(category_name, "")
    search_query = f"{question} {keywords}".strip()

    context_texts = []
    try:
        if retriever is None:
            raise ValueError("Retriever is None")

        # Check if it's a hybrid retriever
        if hasattr(retriever, 'dense_index') and hasattr(retriever, 'sparse_index'):
            # It's a hybrid retriever - use async hybrid search
            context_texts = await pc.get_context_hybrid_async(search_query, retriever, top_n=top_n)
            print(f"Retrieved {len(context_texts)} context chunks for category: {category_name}, given top_n: {top_n}", flush=True)
        else:
            logging.warning("Non-hybrid retriever provided, skipping context retrieval")

    except Exception as e:
        logging.error(f"Error retrieving context get answer mistral ans_ret: {str(e)}")
        context_texts = []

    # Log initial snippets for debugging
    for chunk in context_texts:
        if chunk.get("page") is not None and chunk.get("file") is not None:
            print(f"Initial context chunk - File: {chunk['file']}, Page: {chunk['page']}, Content snippet: {chunk['page_content'][:100]}...", flush=True)

    print(f"Total context chunks before deduplication: {len(context_texts)}")
    
    # Deduplicate the single high-quality pass (no follow-up loops)
    unique_context = await dedup_contexts(context_texts) if context_texts else []
    
    print(f"Total unique context chunks after deduplication: {len(unique_context)}")

    # Format into string for the LLM
    formatted_context = "\n\n".join([
        f"{chunk['page_content']}\n<<<Source: File {chunk['file']}, Page {chunk['page']}>>>" 
        for chunk in unique_context if chunk.get("page") is not None and chunk.get("file") is not None
    ])

    print(f"\n{'='*60}")
    print(f"FINAL EXTRACTED CONTEXT FOR: {category_name}")
    print(f"{'='*60}")
    print(formatted_context)
    print(f"{'='*60}\n", flush=True)
    
    return formatted_context