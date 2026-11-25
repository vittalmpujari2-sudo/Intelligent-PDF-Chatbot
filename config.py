"""
config.py - Central settings for the whole project.

All other files import their settings from here, so if you want to change
something (chunk size, number of retrieved chunks, model name) you only
need to edit this one file or your .env file.
"""

import os
from pathlib import Path

# --- Small environment tweaks that avoid common Windows warnings/errors ---
# (must be set BEFORE faiss / torch / sentence-transformers are imported)
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")          # faiss + torch OpenMP clash
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")       # silences tokenizer warning
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")  # Windows symlink warning

from dotenv import load_dotenv  # noqa: E402

BASE_DIR = Path(__file__).resolve().parent
load_dotenv(BASE_DIR / ".env")

# --- Groq settings (read from the .env file, never hard-coded) ---
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "").strip()
GROQ_MODEL = os.getenv("GROQ_MODEL", "").strip() or "llama-3.3-70b-versatile"

# --- Embedding model (small, fast, runs on CPU) ---
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"

# --- Chunking settings (measured in characters) ---
CHUNK_SIZE = 800      # max characters in one chunk
CHUNK_OVERLAP = 150   # characters shared between neighbouring chunks

# --- Retrieval / generation settings ---
TOP_K = 3             # number of chunks sent to the LLM
TEMPERATURE = 0.0     # 0 = most factual, least creative
MAX_TOKENS = 700      # max length of the generated answer

# --- Fixed reply used when the PDF has no answer ---
FALLBACK_ANSWER = "I could not find this information in the uploaded PDF."


def is_api_key_configured() -> bool:
    """True only if the user replaced the placeholder with a real key."""
    return bool(GROQ_API_KEY) and GROQ_API_KEY != "your_api_key"
