"""
Centralized configuration settings for the Financial Advisor Agent.
Loads environment variables and provides all configuration values.
"""

import os
from dotenv import load_dotenv

load_dotenv()

OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY")
if OPENROUTER_API_KEY:
    OPENROUTER_API_KEY = OPENROUTER_API_KEY.strip()

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
DEFAULT_TIMEOUT = 30
ROUTING_ENABLE_LLM_FALLBACK = os.environ.get("ROUTING_ENABLE_LLM_FALLBACK", "false").strip().lower() == "true"

REASONING_MODEL = "deepseek/deepseek-r1"
GENERAL_MODEL = "qwen/qwen3-30b-a3b"
# EMBEDDING_MODEL removed - now using local SentenceTransformer (all-MiniLM-L6-v2)
# See src/retrieval/embedding_model.py for embedding configuration

RETRIEVAL_TOP_K = 3
RETRIEVAL_CANDIDATE_K = 15

if not OPENROUTER_API_KEY:
    print("WARNING: OPENROUTER_API_KEY not set")