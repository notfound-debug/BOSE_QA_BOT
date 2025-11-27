import os
import sys
import shutil
from langchain_chroma import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.retrievers import BM25Retriever
from langchain.retrievers import EnsembleRetriever
from langchain_core.documents import Document

# --- PATH FIX ---
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.config import DB_DIR, EMBEDDING_MODEL_NAME, DATA_DIR
from src.splitter import split_documents
from src.loader import load_documents


def create_vector_db():
    """Creates Chroma vector DB from PDFs."""
    
    # Reset DB
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


def _find_matching_path(keyword: str):
    """Find real matching PDF path inside the data folder."""
    import glob
    candidates = glob.glob(os.path.join(DATA_DIR, "*"))

    for path in candidates:
        if keyword in os.path.basename(path):
            return os.path.abspath(path)

    return None


def get_retriever(target_pdf_name=None):
    """
    Returns a HYBRID retriever (Dense + BM25) with optional PDF filtering.
    Ensures both UI and eval.py retrieve IDENTICAL chunks.
    """

    # Dense retriever through ChromaDB
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)
    vector_store = Chroma(
        persist_directory=DB_DIR,
        embedding_function=embeddings
    )

    dense_kwargs = {"k": 4}
    dense_retriever = vector_store.as_retriever(search_kwargs=dense_kwargs)

    # Pull ALL documents from Chroma
    raw = vector_store.get()  # dict: {documents: [...], metadatas: [...]}

    all_docs = [
        Document(page_content=text, metadata=meta)
        for text, meta in zip(raw["documents"], raw["metadatas"])
    ]

    # If product filter requested → pre-filter docs for BM25
    bm25_docs = all_docs
    if target_pdf_name:
        real_path = _find_matching_path(target_pdf_name)

        if real_path:
            bm25_docs = [
                d for d in all_docs if d.metadata.get("source") == real_path
            ]
            dense_kwargs["filter"] = {"source": real_path}
            print(f"Locking search to: {os.path.basename(real_path)}")
        else:
            print(f" Warning: Could not match any file for '{target_pdf_name}'")

    # BM25 retriever on filtered Documents
    bm25_retriever = BM25Retriever.from_documents(bm25_docs)
    bm25_retriever.k = 4

    # Hybrid = Dense + BM25 (50/50 weight)
    hybrid = EnsembleRetriever(
        retrievers=[dense_retriever, bm25_retriever],
        weights=[0.5, 0.5]
    )

    return hybrid


# --- DEBUG TEST ---
if __name__ == "__main__":
    print("--- DEBUGGING FILTER ---")
    r = get_retriever("DM8SE")
    docs = r.invoke("IP Rating")

    print(f"Found {len(docs)} docs.")
    if docs:
        print("First doc source:", docs[0].metadata.get("source"))
