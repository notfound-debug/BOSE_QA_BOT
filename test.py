import sys
import os

# Add 'src' to python path so we can import config
sys.path.append(os.path.join(os.getcwd(), 'src'))

print("--- DIAGNOSTIC TEST START ---")

try:
    # 1. Test Config
    from src.config import DATA_DIR, GOOGLE_API_KEY
    print(f"✅ Config: Loaded. Data dir is {DATA_DIR}")
except ImportError as e:
    print(f"❌ Config Error: {e}")

try:
    # 2. Test LangChain Imports (The ones you said failed)
    from langchain.chains import RetrievalQA
    from langchain.chains import create_retrieval_chain
    from langchain.chains.combine_documents import create_stuff_documents_chain
    print("✅ LangChain Chains: IMPORT SUCCESS")
except ImportError as e:
    print(f"❌ LangChain Chain Error: {e}")
    print("👉 HINT: Run 'pip install -r requirements.txt' again.")

try:
    # 3. Test Vector Store & Embeddings
    from langchain_chroma import Chroma
    from langchain_community.embeddings import HuggingFaceEmbeddings
    print("✅ Chroma & Embeddings: IMPORT SUCCESS")
except ImportError as e:
    print(f"❌ Vector Store Error: {e}")

try:
    # 4. Test Google Gemini Import
    from langchain_google_genai import ChatGoogleGenerativeAI
    print("✅ Google Gemini: IMPORT SUCCESS")
except ImportError as e:
    print(f"❌ Gemini Error: {e}")

print("--- DIAGNOSTIC TEST END ---")