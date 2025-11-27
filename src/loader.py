import os
import sys
import glob
from langchain_community.document_loaders import PyPDFLoader

# --- THE FIX: Add parent directory to path so 'src' imports work ---
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
# -------------------------------------------------------------------

from src.config import DATA_DIR

def load_documents():
    """
    Scans the DATA_DIR for PDF files and loads them.
    Returns: A list of LangChain Document objects.
    """
    # Verify directory exists first
    if not os.path.exists(DATA_DIR):
        print(f" Error: Directory not found at {DATA_DIR}")
        return []

    pdf_files = glob.glob(os.path.join(DATA_DIR, "*.pdf"))
    
    if not pdf_files:
        print(f" No PDFs found in {DATA_DIR}")
        return []

    print(f" Found {len(pdf_files)} PDF(s) in {DATA_DIR}...")
    
    all_documents = []
    
    for pdf_path in pdf_files:
        try:
            print(f"   - Loading: {os.path.basename(pdf_path)}")
            loader = PyPDFLoader(pdf_path)
            docs = loader.load()
            all_documents.extend(docs)
            print(f"     Loaded {len(docs)} pages.")
            
        except Exception as e:
            print(f"  Error loading {os.path.basename(pdf_path)}: {e}")

    print(f"Total Loaded Pages: {len(all_documents)}")
    return all_documents

# --- UNIT TEST ---
if __name__ == "__main__":
    docs = load_documents()
    if len(docs) > 0:
        print("\n--- CONTENT PREVIEW (First 500 chars) ---")
        print(docs[0].page_content[:500])
        print("-----------------------------------------")