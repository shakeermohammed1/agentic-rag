from typing import List, Tuple
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from config.settings import CHUNK_SIZE, CHUNK_OVERLAP
from src.utils import setup_logging
from src.vector_store import load_existing_vector_store, create_vector_query_engine
from config.settings import CHROMA_DB_DIR
logger = setup_logging()

def load_and_process_documents(pdf_path: str) -> Tuple[List[Document], List[Document]]:
    """Load and process PDF documents - equivalent to LlamaIndex SimpleDirectoryReader"""
    logger.info(f"Loading document from {pdf_path}...")
    
    # Load document
    loader = PyPDFLoader(pdf_path)
    documents = loader.load()
    logger.info(f"Loaded {len(documents)} document(s).")
    
    # Split into chunks - equivalent to LlamaIndex SentenceSplitter
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        length_function=len,
    )
    chunks = splitter.split_documents(documents)
    logger.info(f"Created {len(chunks)} chunks.")
    
    return documents, chunks

def split_documents(documents: List[Document], 
                   chunk_size: int = CHUNK_SIZE, 
                   chunk_overlap: int = CHUNK_OVERLAP) -> List[Document]:
    """Split documents into chunks"""
    if not documents:
        logger.warning("No documents provided for splitting")
        return []
    
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
    )
    
    chunks = text_splitter.split_documents(documents)
    logger.info(f"Split documents into {len(chunks)} chunks")
    return chunks


def answer_question(query: str) -> str:
    """Run semantic search on the vector store."""
    try:
        vector_store = load_existing_vector_store(CHROMA_DB_DIR)
        if vector_store is None:
            return "Error: Vector store not found. Please ensure the system is properly initialized."
        
        query_engine = create_vector_query_engine(vector_store)  # This returns a callable function
        response = query_engine(query)  # Call the function directly, not through .query()
        return str(response)
    except Exception as e:
        logger.error(f"Error in answer_question: {e}")
        return f"Error answering question: {e}"