import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# API Keys
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
SERPAPI_API_KEY = os.getenv("SERPAPI_API_KEY")

# Langfuse Observability Keys
LANGFUSE_PUBLIC_KEY = os.getenv("LANGFUSE_PUBLIC_KEY")
LANGFUSE_SECRET_KEY = os.getenv("LANGFUSE_SECRET_KEY")
LANGFUSE_HOST = os.getenv("LANGFUSE_HOST", "https://cloud.langfuse.com")

# Model configurations
LLM_MODEL = "gpt-3.5-turbo"
EMBEDDING_MODEL = "text-embedding-ada-002"
LLM_TEMPERATURE = 0

# Document processing
CHUNK_SIZE = 1024
CHUNK_OVERLAP = 200
VECTOR_SEARCH_K = 4

# Paths
DATA_DIR = "data"
CHROMA_DB_DIR = "./chroma_db_langchain"
PDF_FILE_PATH = os.path.join(DATA_DIR, "state.pdf")

# Logging
LOG_LEVEL = "INFO"

# Observability settings
ENABLE_LANGFUSE = os.getenv("ENABLE_LANGFUSE", "true").lower() == "true"