import os
import sys
from langchain.text_splitter import RecursiveCharacterTextSplitter

# --- PATH FIX (Standard for all our files) ---
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
# ---------------------------------------------

from src.loader import load_documents

def split_documents(documents):
    """
    Takes a list of documents and splits them into smaller chunks.
    Crucial for RAG to find specific technical details.
    """
    print(f" Splitting {len(documents)} documents...")
    

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,      # Size of each piece
        chunk_overlap=200,    # Overlap to keep context between chunks
        length_function=len,
        separators=["\n\n", "\n", " ", ""] # Try to split by paragraphs first
    )
    
    chunks = text_splitter.split_documents(documents)
    
    print(f"Created {len(chunks)} chunks from {len(documents)} pages.")
    return chunks

# --- UNIT TEST ---
if __name__ == "__main__":
    # 1. Load first
    docs = load_documents()
    
    if docs:
        # 2. Then Split
        my_chunks = split_documents(docs)
        
        # 3. Verify
        if len(my_chunks) > 0:
            print("\n--- CHUNK PREVIEW (Chunk #1) ---")
            print(my_chunks[0].page_content)
            print("--------------------------------")
            print(f"Metadata: {my_chunks[0].metadata}")