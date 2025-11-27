import os
import sys
import logging
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain.retrievers.multi_query import MultiQueryRetriever # <--- Re-adding this

# --- PATH FIX ---
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


from src.config import GOOGLE_API_KEY, LLM_MODEL_NAME
from src.vector_store import get_retriever

# Suppress logs
logging.getLogger("langchain.retrievers.multi_query").setLevel(logging.INFO)

def get_qa_chain(target_pdf=None):
    if not GOOGLE_API_KEY:
        print("❌ ERROR: Missing API Key")
        return None

    llm = ChatGoogleGenerativeAI(
        model=LLM_MODEL_NAME,
        google_api_key=GOOGLE_API_KEY,
        temperature=0.5
    )

    # 1. Get the FILTERED retriever (Locks to the correct PDF)
    base_retriever = get_retriever(target_pdf_name=target_pdf)

    # 2. MULTI-QUERY RETRIEVER
    advanced_retriever = MultiQueryRetriever.from_llm(
        retriever=base_retriever,
        llm=llm
    )

    # 3. The Prompt (Slightly relaxed to allow acronym inference)
    custom_prompt_template = """You are a technical assistant for Bose Professional products.
    
    Context: {context}
    
    User Question: {question}
    
    INSTRUCTIONS:
    1. Answer based on the context provided.
    2. You may expand acronyms ONLY if the expansion text exists in the provided context. 
      Do NOT add external definitions (e.g., IP ratings, safety standards, electrical codes). 
      You must ONLY answer using wording found inside the retrieved context.
    3. DATA CLEANING: The source text may have merged words due to PDF formatting.
    4. If the answer is not in the context, say: 
       "This query is not related to the currently selected product."
    
    Helpful Answer:"""
    
    PROMPT = PromptTemplate(
        template=custom_prompt_template, input_variables=["context", "question"]
    )

    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=advanced_retriever, # Use the advanced one!
        return_source_documents=True,
        chain_type_kwargs={"prompt": PROMPT}
    )
    
    return qa_chain