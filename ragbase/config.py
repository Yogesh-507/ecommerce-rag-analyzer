import os
from pathlib import Path

class Config:
    class Path:
        APP_HOME = Path(os.getenv("APP_HOME", Path(__file__).parent.parent))
        DATABASE_DIR = APP_HOME / "docs-db"
        DOCUMENTS_DIR = APP_HOME / "tmp"
        IMAGES_DIR = APP_HOME / "images"

    class Database:
        DOCUMENTS_COLLECTION = "documents"

    class Model:
        EMBEDDINGS = "BAAI/bge-base-en-v1.5"
        RERANKER = "ms-marco-MiniLM-L-12-v2"
        LOCAL_LLM = "llama3.2:3b"          # ← Changed from gemma2:9b
        REMOTE_LLM = "llama-3.1-70b-versatile"
        TEMPERATURE = 0.1                   # ← Changed from 0.0 for better responses
        MAX_TOKENS = 4096                   # ← Reduced from 8000 for efficiency
        USE_LOCAL = True                    # ← Changed from False

    class Retriever:
        USE_RERANKER = True
        USE_CHAIN_FILTER = False

    DEBUG = False
    CONVERSATION_MESSAGES_LIMIT = 6
