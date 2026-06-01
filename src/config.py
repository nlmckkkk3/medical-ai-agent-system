"""Configuration management. Loads settings from .env file."""

import os
from pathlib import Path
from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent.parent

load_dotenv(BASE_DIR / ".env")


# LLM provider selection
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "deepseek")

# DeepSeek
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY", "")
DEEPSEEK_BASE_URL = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com")
DEEPSEEK_MODEL = os.getenv("DEEPSEEK_MODEL", "deepseek-chat")

# Zhipu
ZHIPU_API_KEY = os.getenv("ZHIPU_API_KEY", "")
ZHIPU_BASE_URL = os.getenv("ZHIPU_BASE_URL", "https://open.bigmodel.cn/api/paas/v4")
ZHIPU_MODEL = os.getenv("ZHIPU_MODEL", "glm-4.6v")

# Embedding
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "BAAI/bge-small-zh-v1.5")

# Vector database
CHROMA_DB_PATH = os.getenv("CHROMA_DB_PATH", str(BASE_DIR / "data" / "chroma_db"))

# User profiles
USER_DATA_PATH = os.getenv("USER_DATA_PATH", str(BASE_DIR / "data" / "user_profiles"))


def get_llm_config() -> dict:
    """Return active LLM configuration based on LLM_PROVIDER."""
    if LLM_PROVIDER == "zhipu":
        return {
            "api_key": ZHIPU_API_KEY,
            "base_url": ZHIPU_BASE_URL,
            "model": ZHIPU_MODEL,
        }
    return {
        "api_key": DEEPSEEK_API_KEY,
        "base_url": DEEPSEEK_BASE_URL,
        "model": DEEPSEEK_MODEL,
    }
