import logging
from .generate_questions import generate_follow_up_questions, dedup_contexts
from .prompts import RETRIEVAL_QUERIES

async def get_pinecone_context(question: str, retriever, pc, top_n=13):
    tempy = question
    print(f"Received question: {question}", flush=True)

    # Added for consistency with downstream dependencies
    q_template = """CONTEXT:\n{context}\n\n\nAnswer the following question, that is given below based on the context provided above with respect to the system instructions:
1. {question}"""

    # =========================================================================
    # RETRIEVAL QUERY MAPPINGS
    # =========================================================================

    if question.lower() == "technical eligibility evaluation":
        question = (
            "What is the eligibility criteria, evaluation criteria for this tender? "
            "completed works, emd (Earnest Money Deposit), company certifications and registrations, "
            "bidder eligibility criteria, eligibility conditions, qualification requirements, "
            "technical eligibility, technical qualification, experience criteria, "
            "financial eligibility, financial qualification, annual turnover, net worth, "
            "bid capacity, working capital, solvency / bank solvency, legal eligibility, "
            "statutory compliance, registration requirements, disqualification conditions, "
            "who is eligible to bid, pre-qualification requirements, Joint Venture (JV) information"
        )

    elif question.lower() == "financial documents":
        question = (
            "Extract all Joint Venture (JV) information from the tender document. "
            "Answer 'Yes' or 'No' if JVs are allowed, then list all JV details."
        )

    elif question.lower() == "risk assessment policies":
        question = "What are the payment terms mentioned in the tender?"

    elif question.lower() == "company registration documents":
        question = (
            "Analyze the provided tender document and extract the complete "
            "Scope of Work for the bidder."
        )

    elif question.lower() == "keyexperts and technical staff":
        question = (
            "What are the key experts and technical staff requirements "
            "mentioned in the tender document?"
        )

    else:
        # Unknown category — return empty so LLM cannot hallucinate from training knowledge
        return ""

    # =========================================================================
    # CRITICAL FIX: Append broad keywords to capture all relevant chunks
    # =========================================================================
    question += RETRIEVAL_QUERIES[tempy]

    # =========================================================================
    # INITIAL RETRIEVAL
    # =========================================================================
    context_texts = []

    try:
        if retriever is None:
            raise ValueError("Retriever is None")

        if hasattr(retriever, "ainvoke"):
            context_texts = await pc.get_context_hybrid_async(
                question, retriever, top_n=top_n
            )
        else:
            logging.warning("Non-hybrid retriever provided, skipping context retrieval")

    except Exception as e:
        logging.error(
            f"Error retrieving context for '{tempy}': {str(e)}", exc_info=True
        )
        context_texts = []

    # Format initial context safely
    formatted_context = "\n\n".join(
        f"{chunk['page_content']}\n<<<Source: File {chunk['file']}, Page {chunk['page']}>>>"
        for chunk in context_texts
        if chunk.get("page") is not None and chunk.get("file") is not None
    ) if context_texts else "No relevant context found."

    # Second retrieval pass (matches company production code structure)
    try:
        if hasattr(retriever, 'dense_index') and hasattr(retriever, 'sparse_index'):
            context_texts = await pc.get_context_hybrid_async(
                question, retriever, top_n=top_n
            )
            print(
                f"Retrieved {len(context_texts)} context chunks for '{tempy}', top_n={top_n}",
                flush=True
            )
        else:
            logging.info("Hybrid retriever check: dense_index/sparse_index not found, using first-pass results")
    except Exception as e:
        logging.error(f"Error in second retrieval pass for '{tempy}': {str(e)}")
        # Keep context_texts from first pass

    formatted_context = "\n\n".join(
        f"{chunk['page_content']}\n<<<Source: File {chunk['file']},Page {chunk['page']}>>>"
        for chunk in context_texts
        if chunk.get("page") is not None and chunk.get("file") is not None
    )

    print(len(context_texts), flush=True)
    for chunk in context_texts:
        if chunk.get("page") is not None and chunk.get("file") is not None:
            print(
                f"Initial context chunk - File: {chunk['file']}, Page: {chunk['page']}, "
                f"Content snippet: {chunk['page_content'][:100]}...",
                flush=True
            )

    # =========================================================================
    # FOLLOW-UP QUESTION ENRICHMENT
    # =========================================================================
    all_context = []

    questions = await generate_follow_up_questions(tempy, formatted_context)
    print(f"Generated follow-up questions: {questions}", flush=True)

    for q in questions:
        c = await pc.get_context_hybrid_async(q, retriever, top_n=top_n // 2)
        all_context.extend(c)

    all_context.extend(context_texts)
    print(f"Total context chunks before deduplication: {len(all_context)}")

    all_context = await dedup_contexts(all_context)
    print(f"Total unique context chunks after deduplication: {len(all_context)}")

    formatted_context = "\n\n".join(
        f"{chunk['page_content']}\n<<<Source: File {chunk['file']},Page {chunk['page']}>>>"
        for chunk in all_context
        if chunk.get("page") is not None and chunk.get("file") is not None
    )

    return formatted_context