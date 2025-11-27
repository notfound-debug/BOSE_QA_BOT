# Technical Documentation Assistant (Hybrid RAG)

## Overview
A hybrid RAG system that answers factual questions from Bose manuals (DM8SE and EX-1280C).  
Uses Hybrid Search (BM25 + Embeddings), Multi-Query Expansion, and Gemini Flash 2.0 for accurate, citation-backed answers.

## Architecture
- PDF loading with PyPDFLoader
- Chunking with RecursiveCharacterTextSplitter
- Embeddings: all-MiniLM-L6-v2
- Vector DB: ChromaDB
- Sparse search: BM25
- Hybrid retrieval: 50/50 dense+sparse fusion
- Multi-query expansion for acronyms
- Product-locked retrieval to avoid cross-product hallucination
- LLM: Gemini Flash 2.0
- UI: Streamlit

## Setup
git clone https://github.com/notfound-debug/BOSE_QA_BOT.git

cd BOSE_QA_BOT

pip install -r requirements.txt

Create a .env file:
GOOGLE_API_KEY=YOUR_KEY

Run the app:
streamlit run app.py

## Evaluation
python src/evaluate.py

Results:
- Accuracy: 100%
- Avg latency: <2s

## Project Structure
app.py
src/
  bot.py
  vector_store.py
  loader.py
  splitter.py
  evaluate.py
data/
assets/

## Future Work
- Add VLM for diagrams
- Offline SLM using ONNX
- Multi-manual retrieval


