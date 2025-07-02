import logging
import os
from typing import Any, Dict

def setup_logging(level: str = "INFO"):
    """Setup logging configuration"""
    logging.basicConfig(
        level=getattr(logging, level),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    return logging.getLogger(__name__)

def validate_api_keys():
    """Validate that required API keys are set"""
    from config.settings import OPENAI_API_KEY, SERPAPI_API_KEY
    
    if not OPENAI_API_KEY:
        raise ValueError("OPENAI_API_KEY not found in environment variables")
    if not SERPAPI_API_KEY:
        raise ValueError("SERPAPI_API_KEY not found in environment variables")
    
    # Set environment variables for langchain
    os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY
    os.environ["SERPAPI_API_KEY"] = SERPAPI_API_KEY

def ensure_directories():
    """Ensure required directories exist"""
    from config.settings import DATA_DIR, CHROMA_DB_DIR
    
    os.makedirs(DATA_DIR, exist_ok=True)
    os.makedirs(CHROMA_DB_DIR, exist_ok=True)
