import os
import sys
import shutil
from langchain_chroma import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings

# --- PATH FIX ---
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


from src.config import DB_DIR, EMBEDDING_MODEL_NAME, DATA_DIR 
from src.splitter import split_documents
from src.loader import load_documents

def create_vector_db():
   
    if os.path.exists(DB_DIR):
        print(f" Clearing existing database at {DB_DIR}...")
        try:
            shutil.rmtree(DB_DIR)
        except Exception as e:
            print(f" Could not delete folder (might be open): {e}")

    docs = load_documents()
    chunks = split_documents(docs)
    
    if not chunks:
        print(" No chunks created. Check your data folder.")
        return None

    print(f" Loading embedding model ({EMBEDDING_MODEL_NAME})...")
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)

    print(" Creating Chroma database...")
    vector_store = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=DB_DIR
    )
    
    print(f" Database created successfully at {DB_DIR}")
    return vector_store

def _get_actual_source_path(vector_store, keyword):
    """
    INTERNAL HELPER: Looks inside the DB to find the EXACT path string 
    that matches our keyword. This fixes Windows path issues.
    """
    
    # Get all files in the directory
    import glob
    candidates = glob.glob(os.path.join(DATA_DIR, "*"))
    
    for path in candidates:
        if keyword in os.path.basename(path):
            return os.path.abspath(path)
            
    return None

def get_retriever(target_pdf_name=None):
    """
    Returns a retriever with robust filtering.
    """
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)
    vector_store = Chroma(persist_directory=DB_DIR, embedding_function=embeddings)
    
    search_kwargs = {"k": 7}

    if target_pdf_name:
        # Smart Lookup: Find the file path that actually exists
        actual_path = _get_actual_source_path(vector_store, target_pdf_name)
        
        if actual_path:
            print(f" Locking search to: {os.path.basename(actual_path)}")
            search_kwargs["filter"] = {"source": actual_path}
        else:
            print(f" Warning: Could not find any file matching '{target_pdf_name}'")

    return vector_store.as_retriever(search_kwargs=search_kwargs)

# --- DEBUG TEST ---
if __name__ == "__main__":
    #To check if filtering works
    print("--- DEBUGGING FILTER ---")
    retriever = get_retriever("DM8SE")
    docs = retriever.invoke("IP Rating")
    print(f"Querying 'DM8SE' for 'IP Rating'. Found {len(docs)} docs.")
    if len(docs) > 0:
        print(f"Source of first doc: {docs[0].metadata['source']}")
    else:
        print(" FILTER FAILED: No docs found. Path mismatch likely.")