import logging
from langchain.prompts import PromptTemplate
from .prompts import RETRIEVAL_QUERIES
from .generate_questions import generate_follow_up_questions, dedup_contexts

logger = logging.getLogger(__name__)

async def get_pinecone_context(question: str, retriever, pc, top_n=13):
    # Note: We kept the function name 'get_pinecone_context' so it acts as a drop-in 
    # replacement for clause_extraction.py, but it strictly uses local ChromaDB.
    
    tempy = question
    print(f"Received question: {question}", flush=True)
    
    # Map the category to a specific search query
    if question.lower() == "technical eligibility evaluation":
        search_query = "What is the eligibility criteria , evaluation criteria for this tender?completed works, emd (Earnest Money Deposit), company certifications and registrations, bidder eligibility criteria, eligibility conditions, qualification requirements, technical eligibility, technical qualification, experience criteria, financial eligibility, financial qualification,annual turnover, net worth, bid capacity, working capital, solvency / bank solvency , legal eligibility, statutory compliance, registration requirements, disqualification conditions, who is eligible to bid, pre-qualification requirements ,Joint Venture (JV) information"
    elif question.lower() == "financial documents":
        search_query = "Extract all Joint Venture (JV) information from the tender document. Answer 'Yes' or 'No' if JVs are allowed, then list all JV details."
    elif question.lower() == "risk assessment policies":
        search_query = "What are the payment terms mentioned in the tender?"
    elif question.lower() == "company registration documents":
        search_query = "Analyze the provided tender document and extract the complete 'Scope of Work' for the bidder."
    elif question.lower() == "keyexperts and technical staff":
        search_query = "What are the key experts and technical staff requirements mentioned in the tender document?"
    else:
        return ""   # For other questions, we won't fetch context here
        
    search_query += " " + RETRIEVAL_QUERIES.get(tempy, "")
    
    # 1. Fetch initial context using ChromaDB Hybrid Retriever
    try:
        if retriever is None:
            raise ValueError("Retriever is None")

        # Calls the async hybrid search method in your pc_db.py (PC_Mistral)
        context_texts = await pc.get_context_hybrid_async(search_query, retriever, top_n=top_n)
        print(f"Retrieved {len(context_texts)} context chunks for question.", flush=True)
        
    except Exception as e:
        logger.error(f"Error retrieving context from ChromaDB: {str(e)}", exc_info=True)
        context_texts = []

    # 2. Format context for follow-up generation
    formatted_context = "\n\n".join([
        f"{chunk['page_content']}\n<<<Source: File {chunk['file']}, Page {chunk['page']}>>>" 
        for chunk in context_texts if chunk.get("page") is not None and chunk.get("file") is not None
    ])
    
    all_context = list(context_texts)
    
    # 3. Generate Follow-up Questions and get additional context
    try:
        questions = await generate_follow_up_questions(tempy, formatted_context)
        print(f"Generated follow-up questions: {questions}", flush=True)
        
        for q in questions:
            # Fetch additional context based on follow-up questions
            c = await pc.get_context_hybrid_async(q, retriever, top_n=max(1, top_n // 2))
            all_context.extend(c)
            
    except Exception as e:
        logger.error(f"Error during follow-up context retrieval: {str(e)}")

    print(f"Total context chunks before deduplication: {len(all_context)}")
    
    # 4. Deduplicate contexts
    if all_context:
        all_context = await dedup_contexts(all_context)
        
    print(f"Total unique context chunks after deduplication: {len(all_context)}")
    
    # 5. Format the final output context
    final_formatted_context = "\n\n".join([
        f"{chunk['page_content']}\n<<<Source: File {chunk.get('file', 'Unknown')}, Page {chunk.get('page', 'Unknown')}>>>" 
        for chunk in all_context if chunk.get("page") is not None and chunk.get("file") is not None
    ])
    
    return final_formatted_context