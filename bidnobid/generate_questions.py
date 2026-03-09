from utils.create_llm import create_azure_chat_openai
import os
from dotenv import load_dotenv
load_dotenv()

llm1 = create_azure_chat_openai(
        azure_deployment=os.environ.get("AZURE_OPENAI_MODEL_TD"),
        api_version=os.environ.get("OPENAI_API_VERSION_TD"),
        api_key=os.environ.get("AZURE_OPENAI_KEY_TD"),
    )

async def generate_follow_up_questions(question, context):
    print(f"Generating follow-up questions for question: {question} with context: {context[:100]}")
    system_prompt = f"""
You are an expert in analyzing tender documents and identifying key information for bid/no-bid decisions.
Given a question about tender requirements, generate 5 concise follow-up questions that would help    
clarify the tender's requirements and assist in making a clauses from the tender document.It should be based on the context provided and should aim to extract specific details that are commonly found in tender documents, such as eligibility criteria, evaluation criteria, submission requirements, deadlines, and compliance factors.
##Focus
1.Each question should be framed from the perspective of a bidder trying to understand the tender requirements in depth. and missing details in the context provided.
2. The questions should be designed to probe for critical information that would impact the decision to bid or not, such as specific eligibility criteria, evaluation factors, submission requirements, deadlines, and compliance issues.
3. Avoid generating questions that are too broad or vague. Each question should target a specific aspect
Follow these guidelines:
- Each question should be clear and focused on a specific aspect of the tender requirements.
- Avoid vague questions.
-Given the context of the original question, tailor the follow-up questions to probe for details that would be critical in assessing the tender's suitability for bidding.
##context##
{context}
##Question##
{question}
##Follow-up Questions##
- send it as a list of 5 questions without any additional text or formatting.
"""
    user_prompt = f"""Original Question: {question}"""
    questions_str = llm1.invoke([("system", str(system_prompt)), ("human", str(question))]).content
    print(f"Generated follow-up questions string: {questions_str}")
    return [q.strip() for q in questions_str.split("\n") if q.strip()]

import hashlib
import re

async def normalize(text):
    return re.sub(r"\s+", " ", text.lower()).strip()

async def dedup_contexts(chunks, prefix_len=250):
    seen_exact = set()
    seen_prefix = set()
    unique_chunks = []

    for c in chunks:
        content =await normalize(c["page_content"])

        exact_hash = hashlib.md5(content.encode()).hexdigest()
        prefix_hash = hashlib.md5(content[:prefix_len].encode()).hexdigest()

        if exact_hash in seen_exact or prefix_hash in seen_prefix:
            continue

        seen_exact.add(exact_hash)
        seen_prefix.add(prefix_hash)
        unique_chunks.append(c)

    return unique_chunks