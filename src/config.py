import os
from dotenv import load_dotenv

load_dotenv()

# =========================
# API Keys
# =========================

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")

# =========================
# Model Configuration
# =========================

LLM_MODEL = os.getenv(
    "LLM_MODEL",
    "gpt-4o-mini"
)

EMBEDDING_MODEL = os.getenv(
    "EMBEDDING_MODEL",
    "text-embedding-3-small"
)

# =========================
# Pinecone Configuration
# =========================

PINECONE_INDEX_NAME = os.getenv(
    "PINECONE_INDEX_NAME",
    "agentic-ai-index"
)

# =========================
# RAG Configuration
# =========================

CHUNK_SIZE = int(
    os.getenv("CHUNK_SIZE", "1000")
)

CHUNK_OVERLAP = int(
    os.getenv("CHUNK_OVERLAP", "200")
)

TOP_K = int(
    os.getenv("TOP_K", "5")
)

# =========================
# Data Paths
# =========================

DATA_DIR = os.path.join(
    os.path.dirname(os.path.dirname(__file__)),
    "data"
)

PDF_PATH = os.path.join(
    DATA_DIR,
    "Ebook-Agentic-AI.pdf"
)

# =========================
# Validation
# =========================

if not OPENAI_API_KEY:
    print("Warning: OPENAI_API_KEY is not set in .env")

if not PINECONE_API_KEY:
    print("Warning: PINECONE_API_KEY is not set in .env")