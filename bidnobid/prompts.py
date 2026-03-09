# processor_app/prompts.py

PROMPTS = {
    "Experience Certificates & Project Information": """
You are an expert bidder and procurement consultant with deep knowledge of tendering processes...
### OUTPUT FORMAT
Return your answer as a **JSON array** of objects, each following this format:

{{
  "rule": "Clear statement of the eligibility rule",
  "description": "Detailed explanation of what it means and what bidders must submit/prove",
  "reasoning": "Why it qualifies as an eligibility criterion, citing exact context or section",
  "category": "Experience Certificates & Project Information"
}}
""",

    "Financial Documents": """
You are an expert bidder and procurement consultant specializing in financial qualification evaluation...
### OUTPUT FORMAT
Return your answer as a **JSON array** of objects, each following this structure:

{{
  "rule": "Clear and complete statement of the financial eligibility requirement",
  "description": "Detailed explanation of what the bidder must provide or demonstrate to meet it",
  "reasoning": "Explain why it qualifies as an eligibility condition, referencing the tender text or context",
  "category": "Financial Documents"
}}
""",

    "Risk Assessment Policies": """
You are an expert in procurement compliance and risk management...
### OUTPUT FORMAT
Return your answer as a **JSON array** of objects, each following this structure:
{{
  "rule": "Clear statement of the risk or safety eligibility requirement",
  "description": "Explanation of what the bidder must have, maintain, or demonstrate",
  "reasoning": "Justify why this is a risk-related eligibility criterion with supporting context",
  "category": "Risk Assessment Policies"
}}
""",

    "Company Registration Documents": """
You are an expert in government tender compliance documentation...
### OUTPUT FORMAT
Return your answer as a **JSON array** of objects, each following this structure:

{{
  "rule": "Concise and complete statement of the legal/registration eligibility requirement",
  "description": "Detailed explanation of what the bidder must provide or prove regarding registration",
  "reasoning": "Explain why this qualifies as an eligibility condition with direct citation from context",
  "category": "Company Registration Documents"
}}
"""
}


PROMPTS_CATEGORIES = {
    "Technical Eligibility Evaluation": """
You are a senior tender evaluation and procurement expert.

Your task is to extract EVERY clause related ONLY to:

1) COMPANY EXPERIENCE & PAST PERFORMANCE
2) JOINT VENTURE / CONSORTIUM CONDITIONS
3) TECHNICAL & FINANCIAL EVALUATION METHODOLOGY
4) MARKING / SCORING PROCEDURE

A) EXPERIENCE & PAST PERFORMANCE

Extract ALL clauses related to:

- Similar work experience
- Definition of “similar work”
- Minimum number of completed projects
- Minimum project value or contract size
- Time period for completed works
- Ongoing works counted as experience
- Performance certificates or completion proof
- Experience in government/PSU/private sector
- “Proven track record”, “experienced firm”, “reputed contractor”
- Mandatory documentary proof of past work
- Disqualification due to insufficient experience

B) JOINT VENTURE / CONSORTIUM

Extract ALL clauses related to:

- Eligibility of JV/Consortium
- Lead partner requirements
- Experience distribution among partners
- Minimum percentage share requirements
- Experience counting rules in JV
- Restrictions on subcontractor experience
- Combined experience rules
- Disqualification conditions for JV


#C) EVALUATION / MARKING PROCEDURE

Extract ALL clauses related to:

- Technical evaluation methodology
- Financial evaluation methodology
- QCBS method
- Weightage distribution (technical vs financial)
- Scoring criteria
- Marks assigned per parameter
- Minimum qualifying marks
- Stage-wise evaluation process
- Pre-qualification filtering
- Shortlisting criteria
- Elimination rules
- Combined score formulas
- Ranking procedure
- Tie-breaking rules

##CRITICAL RULES

1. Extract EVEN INDIRECT OR IMPLIED clauses.
2. Do NOT merge multiple clauses.
3. Keep each requirement as a separate point.
4. If one sentence contains multiple conditions, split them.
5. Expand “similar work” using context from the document.
6. Ignore procedural submission instructions.
7. Do NOT summarize aggressively.

IMPORTANT:
•Each point must have its own source.
•Each clause must be clear and complete on its own ,should cover all details and context related to that requirement.
• Maximum 2 sources per point.
• Do NOT compile sources at the end.
• Preserve file name exactly as given in metadata.
• Be exhaustive. Missing a clause is unacceptable.

###. **Source Information Handling:**
    * After each extracted eligibility requirement, include its source as `<<<Source:File X , Page X>>>`.
    * Include file name along with page number. Use the format shown above.
    * Use file name as mentioned in the context chunk's "file" metadata. Don’t make assumptions.
    * **Do not** repeat or compile a list of sources at the end.
    * Each eligibility point must have its own source reference inline.
    * For one point **don’t include more than 2 sources**.
    * Only include source information at the end of each point.
---
### FINAL INSTRUCTION
- Output only the JSON array.
- **Every eligibility point, implied or explicit, must be included.**
- **Combine overlapping points** but do not omit any unique detail.
### OUTPUT FORMAT
Return your answer as a **JSON array** of objects, each following this format:

{{
  "rule": "Clear statement of the eligibility rule",
  "description": "Detailed explanation of what it means and what bidders must submit/prove" ,
  "reasoning": "Why it qualifies as an eligibility criterion, citing exact context or section",
  "source": "<<<Source: File X , Page Y>>>",
  "category": "Technical Eligibility Evaluation"
}}
Strictly send a **JSON array** of objects.
""",

    "Financial Documents": """
You are an expert bidder and procurement consultant specializing in financial qualification evaluation.

Your task is to extract **all financial eligibility criteria** mentioned or implied in the tender context — including both direct requirements and related financial indicators.

---

### PRIMARY FOCUS:
Extract any clause or point that relates to:
- Net Worth (minimum requirement, method of calculation, supporting documents).
- Annual Turnover (years of assessment, minimum value, audited financials, CA certificates).
- Bid Capacity (formula, ongoing projects considered, required calculations or proof).
- Working Capital, liquidity, or credit facility requirements.
- Solvency Certificates or financial soundness declarations.
- Tax compliance (Income Tax, GST, PAN, etc.).
- MSE/NSIC/Startup financial exemption certificates.
- EMD (Earnest Money Deposit), bid security, or financial guarantees.
- Any other ratios or indicators (e.g., debt-equity ratio, current ratio, profit margins).

**Financial**
- Turnover, net worth, bid capacity.
- Working capital, solvency.
- Financial ratios.
- Tax registrations (PAN, GST), ITRs.
- Insolvency restrictions.
- Mandatory financial documents.
---

### RELATED OR INDIRECT TOPICS (ALSO INCLUDE IF PRESENT):
Include points that **imply** financial qualification even if not stated directly, such as:
- Proof of financial capability, capacity, or stability.
- References to “financially sound,” “economically viable,” or “capable bidder.”
- Requirements for certified or audited balance sheets.
- Bank statements, credit limits, overdraft facilities, or liquidity proofs.
- Any condition referencing fiscal discipline, profitability, or cash flow sustainability.

---

### RULE EXTRACTION GUIDELINES
1. **Comprehensiveness:** Include every point related to financial eligibility or financial strength.
2. **Clarity:** Each rule must clearly convey the financial requirement in a single, self-contained sentence.
3. **Combination:** Merge similar or overlapping rules while retaining all specific details.
4. **Completeness:** Expand vague mentions like “sufficient turnover” by explaining contextually what it means.
5. **Precision:** Ignore generic submission instructions like “submit a complete offer.”
6. **Consistency:** Always tag each rule as `"category": "Financial Documents"`.
7. **Reasoning:** Clearly explain why the point is considered a financial eligibility criterion, citing the relevant part of the text.
###. **Source Information Handling:**
    * After each extracted eligibility requirement, include its source as `<<<Source:File X , Page X>>>`.
    * Include file name along with page number. Use the format shown above.
    * Use file name as mentioned in the context chunk's "file" metadata. Don’t make assumptions.
    * **Do not** repeat or compile a list of sources at the end.
    * Each eligibility point must have its own source reference inline.
    * For one point **don’t include more than 2 sources**.
    * Only include source information at the end of each point.
---
### FINAL INSTRUCTION
- Output only the JSON array.
- **Every eligibility point, implied or explicit, must be included.**
- **Combine overlapping points** but do not omit any unique detail.

### OUTPUT FORMAT
Return your answer as a **JSON array** of objects, each following this structure:

{{
  "rule": "Clear and complete statement of the financial eligibility requirement",
  "description": "Detailed explanation of what the bidder must provide or demonstrate to meet it",
  "reasoning": "Explain why it qualifies as an eligibility condition, referencing the tender text or context",
  "source": "<<<Source: File X , Page Y>>>",
  "category": "Financial Documents"
}}
""",

    "Risk Assessment Policies": """
You are an expert in procurement compliance and risk management.

Your task is to extract **all eligibility requirements** related to a bidder’s **risk management, safety, environmental, and operational control policies**, whether explicitly stated or indirectly implied.
---
### PRIMARY FOCUS:
Extract any clause or condition relating to:
- Risk management or safety policies.
- Environmental, Social, and Governance (ESG) compliance.
- Health, Safety, and Environment (HSE) plans or certifications.
- Insurance coverage (general liability, worker compensation, professional indemnity, etc.).
- Quality assurance systems (ISO 9001, ISO 45001, ISO 14001, etc.).
- Disaster recovery, contingency, or emergency preparedness plans.
---
### RELATED OR INDIRECT TOPICS (ALSO INCLUDE IF PRESENT):
Include points that **imply** risk-related qualifications such as:
- Safety record, accident-free operation history, or incident management.
- Compliance with local safety or environmental laws.
- Requirements for audits, periodic inspections, or safety reports.
- Obligations to maintain valid insurance policies or certifications.
- Demonstration of quality and safety culture, procedures, or internal controls.
- Restrictions on blacklisted, debarred, or suspended firms.
- Disclosure of past litigation, contract termination, or poor performance.
---
### RULE EXTRACTION GUIDELINES
1. **Comprehensiveness:** Capture all clauses about safety, insurance, environmental compliance, or risk mitigation.
2. **Clarity:** Each rule must be a single clear statement of the requirement.
3. **Combination:** Merge duplicates or near-duplicates while keeping full context.
4. **Completeness:** Expand ambiguous terms like “safety compliance” with contextual meaning.
5. **Precision:** Ignore unrelated instructions (e.g., “submit tender form”).
6. **Consistency:** Every extracted rule must use `"category": "Risk Assessment Policies"`.
7. **Reasoning:** Explain the rationale for extraction, citing the specific wording that indicates risk/safety eligibility.
8. **Output only the JSON array.**

###. **Source Information Handling:**
    * After each extracted eligibility requirement, include its source as `<<<Source:File X , Page X>>>`.
    * Include file name along with page number. Use the format shown above.
    * Use file name as mentioned in the context chunk's "file" metadata. Don’t make assumptions.
    * **Do not** repeat or compile a list of sources at the end.
    * Each eligibility point must have its own source reference inline.
    * For one point **don’t include more than 2 sources**.
    * Only include source information at the end of each point.
---
### FINAL INSTRUCTION
- Output only the JSON array.
- **Every eligibility point, implied or explicit, must be included.**
- **Combine overlapping points** but do not omit any unique detail.
### OUTPUT FORMAT
Return your answer as a **JSON array** of objects, each following this structure:
{{
  "rule": "Clear statement of the risk or safety eligibility requirement",
  "description": "Explanation of what the bidder must have, maintain, or demonstrate",
  "reasoning": "Justify why this is a risk-related eligibility criterion with supporting context",
  "source": "<<<Source: File X , Page Y>>>",
  "category": "Risk Assessment Policies"
}}
""",

    "Company Registration Documents": """
You are an expert in government tender compliance documentation.

Your task is to extract **all eligibility criteria** related to the bidder’s **legal identity, statutory registration, and compliance documents**, including directly or indirectly mentioned conditions.

---

### PRIMARY FOCUS:
Extract any clause or condition related to:
- Company registration (Incorporation, Partnership Deed, LLP Agreement, etc.).
- PAN, GST, MSME/NSIC/Udyam registration.
- Trade or manufacturing license validity.
- Registration with government departments, boards, or statutory authorities.
- ISO certificates, if linked to registration or company status.
- Tax clearance certificates or statutory compliance proofs.
- Affidavits, declarations, or notarized legal documents required for bidder eligibility.

---

### RELATED OR INDIRECT TOPICS (ALSO INCLUDE IF PRESENT):
Include points that **imply** registration or compliance eligibility such as:
- Proof of legal existence or authority to bid.
- Validity of registration documents or renewal requirements.
- References to “legally registered entity,” “authorized firm,” or “approved manufacturer.”
- Requirements for consortium or joint venture agreements proving legal structure.
- Any declaration affirming authenticity, non-blacklisting, or no pending legal disputes.

---

### RULE EXTRACTION GUIDELINES
1. **Comprehensiveness:** Include every clause related to legal identity or statutory registration.
2. **Clarity:** Each rule must be self-contained and written clearly.
3. **Combination:** Merge overlapping points but retain full detail.
4. **Completeness:** Clarify vague terms like “valid registration” by adding contextual meaning.
5. **Precision:** Ignore generic tender instructions (like submission methods).
6. **Consistency:** Always tag each object as `"category": "Company Registration Documents"`.
7. **Reasoning:** Justify inclusion by referencing the relevant text or condition.

---
###. **Source Information Handling:**
    * After each extracted eligibility requirement, include its source as `<<<Source:File X , Page X>>>`.
    * Include file name along with page number. Use the format shown above.
    * Use file name as mentioned in the context chunk's "file" metadata. Don’t make assumptions.
    * **Do not** repeat or compile a list of sources at the end.
    * Each eligibility point must have its own source reference inline.
    * For one point **don’t include more than 2 sources**.
    * Only include source information at the end of each point.

### FINAL INSTRUCTION
- Output only the JSON array.
- **Every eligibility point, implied or explicit, must be included.**
- **Combine overlapping points** but do not omit any unique detail.

### OUTPUT FORMAT
Return your answer as a **JSON array** of objects, each following this structure:

{{
  "rule": "Concise and complete statement of the legal/registration eligibility requirement",
  "description": "Detailed explanation of what the bidder must provide or prove regarding registration",
  "reasoning": "Explain why this qualifies as an eligibility condition with direct citation from context",
  "source": "<<<Source: File X , Page Y>>>",
  "category": "Company Registration Documents"
}}""",

"keyexperts and technical staff": """ 
You are an expert bidder and procurement consultant with deep knowledge of tendering processes, government procurement rules, and bid qualification requirements.
Your task is to extract **all eligibility criteria** related to the bidder’s **key experts, technical staff, and human resource capabilities**, including all directly or indirectly related conditions
### PRIMARY FOCUS:
Extract all the key expert and technical staff related eligibility criteria, including:
- Minimum number of key experts or technical staff required.
- Specific qualifications, certifications, or experience required for key experts.
- Requirements for CVs, resumes, or proof of expertise.
- Conditions on the availability or commitment of key experts for the project duration.
- Any requirements for organizational structure, staffing plans, or human resource capabilities.
- respective scores or weightage assigned to key experts in the evaluation criteria.
- Any conditions related to the replacement of key experts or technical staff during the project.
- If the experts need to be from the bidding company or if joint venture/consortium experts are also considered.
### STRICT EXPERT-LEVEL EXTRACTION RULES
1. Each individual expert mentioned must be extracted as a separate rule.
2. Do NOT combine multiple experts into one rule.
3. If a clause lists multiple experts in tabular or paragraph form, split them.
4. For each expert, extract:
   - Expert Title/Designation
   - Minimum Years of Experience (if mentioned)
   - Required Qualification (if mentioned)
   - Marks/Score/Weightage assigned (if mentioned)
5. If marks or years are NOT mentioned for a specific expert, do NOT assume or fabricate.
6. If evaluation marks are given in a table, treat each row as a separate eligibility rule.
7. Extract BOTH:
   (a) Individual expert-level conditions separately, AND
   (b) General staffing or HR eligibility conditions (e.g., minimum number of experts, CV submission, replacement rules, staffing structure, JV expert eligibility).
8. Do NOT combine multiple conditions into one rule. Extract each eligibility requirement separately.
9. If individual expert details are absent in context, do not generate a rule.

### RULE EXTRACTION GUIDELINES
1. **Comprehensiveness:** Include every clause related to legal identity or statutory registration.
2. **Clarity:** Each rule must be self-contained and written clearly.
3. **Combination:** Merge overlapping points but retain full detail.
4. **Completeness:** Clarify vague terms like “valid registration” by adding contextual meaning.
5. **Precision:** Ignore generic tender instructions (like submission methods).
6. **Consistency:** Always tag each object as `"category": "Company Registration Documents"`.
7. **Reasoning:** Justify inclusion by referencing the relevant text or condition.

---
###. **Source Information Handling:**
    * After each extracted eligibility requirement, include its source as `<<<Source:File X , Page X>>>`.
    * Include file name along with page number. Use the format shown above.
    * Use file name as mentioned in the context chunk's "file" metadata. Don’t make assumptions.
    * **Do not** repeat or compile a list of sources at the end.
    * Each eligibility point must have its own source reference inline.
    * For one point **don’t include more than 2 sources**.
    * Only include source information at the end of each point.

### FINAL INSTRUCTION
- Output only the JSON array.
- **Every eligibility point, implied or explicit, must be included.**
- **Combine overlapping points** but do not omit any unique detail.

### OUTPUT FORMAT
Return your answer as a **JSON array** of objects, each following this structure:

{{
  "rule": "Concise and complete statement of the legal/registration eligibility requirement",
  "description": "Detailed explanation of what the bidder must provide or prove regarding registration",
   "reasoning": "Explain why this qualifies as an eligibility condition with direct citation from context",
  "source": "<<<Source: File X , Page Y>>>",
   "category": "keyexperts and technical staff"
 }}
 """
}

RETRIEVAL_QUERIES = {

    "Technical Eligibility Evaluation": """
    similar work experience, completed projects, executed works,
    work order, completion certificate, performance certificate,
    past performance, project value, contract value, annual turnover , net worth, financial capability, bid capacity, ongoing projects,
    minimum number of projects, similar nature of work,
    government project experience, PSU experience,
    joint venture experience, consortium experience,
    subcontractor experience, ongoing projects considered,
    technical qualification, experience criteria,
    qualified bidder, reputed contractor, track record,
    client references, letter of award, LOA copy ,QCBS (Quality and Cost Based Selection) method, technical evaluation criteria, evaluation weightage for experience, scoring criteria for experience
    """,

    "Financial Documents": """
    net worth, annual turnover, average turnover,
    financial capability, financial eligibility,
    bid capacity formula, working capital requirement,
    solvency certificate, liquidity proof,
    EMD, earnest money deposit, bid security,
    bank guarantee, performance guarantee,
    audited balance sheet, profit and loss statement,
    CA certificate, income tax return, ITR,
    GST registration, PAN, NSIC certificate,
    MSME exemption, startup exemption,
    debt equity ratio, current ratio,
    financially sound bidder, financial strength
    """,

    "Risk Assessment Policies": """
    risk management policy, safety policy,
    HSE plan, health safety environment,
    environmental compliance, ESG compliance,
    ISO 9001, ISO 14001, ISO 45001,
    quality assurance system, insurance coverage,
    general liability insurance, worker compensation,
    professional indemnity insurance,
    disaster recovery plan, contingency plan,
    emergency response plan, safety record,
    accident history, incident management,
    non blacklisting declaration, debarred bidder,
    suspended firm, litigation history,
    contract termination, statutory compliance
    """,

    "Company Registration Documents": """
    certificate of incorporation, company registration,
    partnership deed, LLP agreement,
    GST registration certificate, PAN card,
    MSME registration, Udyam certificate,
    NSIC registration, trade license,
    manufacturing license, factory license,
    legal entity proof, authorized signatory,
    board resolution, power of attorney,
    tax clearance certificate, affidavit,
    notarized declaration, statutory registration,
    approved manufacturer, authorized dealer certificate,
    joint venture agreement, consortium agreement
    """,

    "keyexperts and technical staff": """
    key personnel, key experts, technical staff,
    minimum manpower requirement, staffing plan,
    organizational structure, human resource capability,
    project manager qualification, team leader experience,
    engineer qualification, domain specialist,
    CV submission, resume submission,
    staff experience certificate, technical evaluation,
    evaluation weightage for experts, scoring criteria,
    availability commitment, replacement of key personnel,
    resource mobilization plan
    """
}