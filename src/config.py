import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# FOLDER PATHS
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
DB_DIR = os.path.join(BASE_DIR, "chroma_db")

# API KEYS (Securely loaded)
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

if not GOOGLE_API_KEY:
    raise ValueError("❌ GOOGLE_API_KEY not found. Please set it in your .env file.")

# MODEL SETTINGS
# We chose this because it's the fastest and has the best free-tier limits
LLM_MODEL_NAME = "gemini-2.0-flash" 
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"