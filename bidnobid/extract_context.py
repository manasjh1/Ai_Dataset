from langchain.prompts import PromptTemplate
from .prompts import RETRIEVAL_QUERIES
import logging
from .generate_questions import generate_follow_up_questions, dedup_contexts
async def get_pinecone_context(question: str, retriever, pc, top_n=13):
    tempy = question
    print(f"Received question: {question}", flush=True)
    q_template = """CONTEXT:\n{context}\n\n\nAnswer the following question, that is given below based on the context provided above with respect to the system instructions:
1. {question}"""
    
    if question.lower() == "technical eligibility evaluation":
      question = "What is the eligibility criteria , evaluation criteria for this tender?completed works, emd (Earnest Money Deposit), company certifications and registrations, bidder eligibility criteria, eligibility conditions, qualification requirements, technical eligibility, technical qualification, experience criteria, financial eligibility, financial qualification,annual turnover, net worth, bid capacity, working capital, solvency / bank solvency , legal eligibility, statutory compliance, registration requirements, disqualification conditions, who is eligible to bid, pre-qualification requirements ,Joint Venture (JV) information"
    elif question.lower() == "financial documents":
      question = "Extract all Joint Venture (JV) information from the tender document. Answer 'Yes' or 'No' if JVs are allowed, then list all JV details."
    elif question.lower() == "risk assessment policies":
      question = "What are the payment terms mentioned in the tender?"
    elif question.lower() == "company registration documents":
      question = "Analyze the provided tender document and extract the complete 'Scope of Work' for the bidder."
    elif question.lower() == "keyexperts and technical staff":
      question = "What are the key experts and technical staff requirements mentioned in the tender document?"
       
    else:
      return ""   # For other questions, we won't fetch context here
    question += RETRIEVAL_QUERIES[tempy]
    try:
        if retriever is None:
            raise ValueError("Retriever is None")

        # Hybrid retriever
        if hasattr(retriever, "ainvoke"):
            context_texts = await pc.get_context_hybrid_async(
                question, retriever, top_n=top_n
            )
        else:
            logging.warning("Non-hybrid retriever provided, skipping context retrieval")

    except Exception as e:
        logging.error(
            f"Error retrieving context in get_answer_mistral: {str(e)}",
            exc_info=True
        )
        context_texts = []

    # --------------------------------------------------
    # Format context safely
    # --------------------------------------------------
    formatted_context = "\n\n".join(
        f"{chunk['page_content']}\n<<<Source: File {chunk['file']}, Page {chunk['page']}>>>"
        for chunk in context_texts
        if chunk.get("page") is not None and chunk.get("file") is not None
    ) if context_texts else "No relevant context found."

    try:
        # Check if it's a hybrid retriever
        if hasattr(retriever, 'dense_index') and hasattr(retriever, 'sparse_index'):
            # It's a hybrid retriever - use async hybrid search
            context_texts = await pc.get_context_hybrid_async(question, retriever, top_n=top_n)
            print(f"Retrieved {len(context_texts)} context chunks for question: {question}, give topn {top_n}", flush=True)
        else:
           logging.info("hybrid retriever failed")
    except Exception as e:
        logging.error(f"Error retrieving context get answer mistral ans_ret: {str(e)}")
        # Fallback to empty context if retrieval fails
        context_texts = []

    formatted_context = "\n\n".join([
        f"{chunk['page_content']}\n<<<Source: File {chunk['file']},Page {chunk['page']}>>>" 
        for chunk in context_texts if chunk.get("page") is not None and chunk.get("file") is not None
    ])
    print(len(context_texts), flush=True)
    for chunk in context_texts:
        if chunk.get("page") is not None and chunk.get("file") is not None:
            print(f"Initial context chunk - File: {chunk['file']}, Page: {chunk['page']}, Content snippet: {chunk['page_content'][:100]}...", flush=True)
    all_context=[]
    # if tempy.lower() in ["eligibility criteria", "evaluation criteria"]:
    questions=await generate_follow_up_questions(tempy, formatted_context)
    print(f"Generated follow-up questions: {questions}", flush=True)
    for q in questions:
            c=await pc.get_context_hybrid_async(q, retriever, top_n=top_n//2)
            all_context.extend(c)
    all_context.extend(context_texts)
    print(f"Total context chunks before deduplication: {len(all_context)}")
    all_context=await dedup_contexts(all_context)
    print(f"Total unique context chunks after deduplication: {len(all_context)}")
    formatted_context = "\n\n".join([
            f"{chunk['page_content']}\n<<<Source: File {chunk['file']},Page {chunk['page']}>>>" 
            for chunk in all_context if chunk.get("page") is not None and chunk.get("file") is not None
        ])
    # for chunk in all_context:
    #     if chunk.get("page") is not None and chunk.get("file") is not None:
    #         print(f"Context chunk - File: {chunk['file']}, Page: {chunk['page']}, Content snippet: {chunk['page_content'][:100]}...", flush=True)
    return formatted_context