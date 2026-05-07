import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# --- PROJECT PATHS ---
ROOT_DIR = Path(__file__).parent.parent.parent
DATA_DIR = ROOT_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
INTERMEDIATE_DATA_DIR = DATA_DIR / "intermediate"
FINAL_DATA_DIR = DATA_DIR / "final"

MODELS_DIR = ROOT_DIR / "models"
BM25_PATH = MODELS_DIR / "bm25.pkl"

# --- API KEYS ---
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
VOYAGE_API_KEY = os.getenv("VOYAGE_API_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
DEEPINFRA_TOKEN = os.getenv("DEEPINFRA_TOKEN")
DOCUMENTINTELLIGENCE_ENDPOINT = os.getenv("DOCUMENTINTELLIGENCE_ENDPOINT")
DOCUMENTINTELLIGENCE_API_KEY = os.getenv("DOCUMENTINTELLIGENCE_API_KEY")

# --- MODEL SETTINGS ---
INDEX_NAME = "hybrid-index"
EMBED_MODEL = "voyage-finance-2"
RERANK_MODEL = "Qwen/Qwen3-Reranker-8B"
LLM_MODEL = "gemini-3-flash-preview"

# Ensure directories exist
for path in [RAW_DATA_DIR, PROCESSED_DATA_DIR, INTERMEDIATE_DATA_DIR, FINAL_DATA_DIR, MODELS_DIR]:
    path.mkdir(parents=True, exist_ok=True)
