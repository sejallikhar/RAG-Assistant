import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
INDEX_DIR = ROOT / "index"
LOG_DIR = ROOT / "logs"

# "ollama" = local open-weight model served by Ollama. "extractive" = no LLM
# (rows are quoted directly); used for CPU-only checks and CI.
LLM_BACKEND = os.getenv("LLM_BACKEND", "ollama")
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434")
LLM_MODEL = os.getenv("LLM_MODEL", "qwen2.5:7b-instruct")

TOP_K = int(os.getenv("TOP_K", "8"))
MAX_ROWS = int(os.getenv("MAX_ROWS", "12"))      # structured lookups returning more rows ask the user to narrow down
MIN_BM25 = float(os.getenv("MIN_BM25", "1.0"))   # hybrid-retrieval floor
USE_DENSE = os.getenv("USE_DENSE", "0") == "1"   # optional embeddings (needs sentence-transformers)
EMBED_MODEL = os.getenv("EMBED_MODEL", "BAAI/bge-small-en-v1.5")
