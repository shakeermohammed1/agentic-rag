from typing import List, Optional, Callable
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_chroma import Chroma
from langchain.chains import RetrievalQA
from langchain.chains.summarize import load_summarize_chain
from config.settings import EMBEDDING_MODEL, CHROMA_DB_DIR, LLM_MODEL, VECTOR_SEARCH_K
from src.utils import setup_logging
import shutil
import os

logger = setup_logging()

def create_embeddings():
    """Create OpenAI embeddings instance"""
    return OpenAIEmbeddings(model=EMBEDDING_MODEL)

def rebuild_vector_store_fresh(chunks: List[Document], persist_directory: str = CHROMA_DB_DIR) -> Optional[Chroma]:
    """Rebuild vector store completely from scratch - FIXED VERSION"""
    logger.info("Rebuilding vector store from scratch...")
    
    # Remove existing vector store completely
    if os.path.exists(persist_directory):
        shutil.rmtree(persist_directory)
        logger.info(f"Removed existing vector store at {persist_directory}")
    
    if not chunks:
        logger.error("No chunks provided for vector store creation")
        return None
    
    try:
        embeddings = create_embeddings()
        vector_store = Chroma.from_documents(
            documents=chunks,
            embedding=embeddings,
            persist_directory=persist_directory
        )
        logger.info(f"Created fresh vector store with {len(chunks)} chunks")
        return vector_store
    except Exception as e:
        logger.error(f"Error creating vector store: {e}")
        return None

def create_vector_index(chunks: List[Document], persist_directory: str = CHROMA_DB_DIR) -> Optional[Chroma]:
    """Create vector index - Always rebuild for consistent results"""
    return rebuild_vector_store_fresh(chunks, persist_directory)

def load_existing_vector_store(persist_directory: str = CHROMA_DB_DIR) -> Optional[Chroma]:
    """Load existing vector store from disk"""
    try:
        if not os.path.exists(persist_directory):
            logger.info("No existing vector store found")
            return None
            
        embeddings = create_embeddings()
        vector_store = Chroma(
            persist_directory=persist_directory,
            embedding_function=embeddings
        )
        
        # Test if it works
        test_results = vector_store.similarity_search("test", k=1)
        logger.info("Loaded existing vector store successfully")
        return vector_store
    except Exception as e:
        logger.error(f"Error loading vector store: {e}")
        return None

def create_vector_query_engine(vector_store: Chroma) -> Callable[[str], str]:
    """Create vector query engine - FIXED VERSION"""
    llm = ChatOpenAI(model=LLM_MODEL, temperature=0)
    
    retrieval_qa = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vector_store.as_retriever(search_kwargs={"k": VECTOR_SEARCH_K}),
        return_source_documents=True,
        verbose=False  # Reduce noise
    )
    
    def vector_query(query: str) -> str:
        """Fixed vector query function"""
        try:
            logger.info(f"Vector search for: {query}")
            result = retrieval_qa.invoke({"query": query})
            answer = result["result"]
            
            # Add source information
            sources = result.get("source_documents", [])
            if sources:
                answer += f"\n\n📚 Found {len(sources)} sources:\n"
                for i, doc in enumerate(sources[:2]):
                    content = doc.page_content[:200] + "..." if len(doc.page_content) > 200 else doc.page_content
                    answer += f"Source {i+1}: {content}\n"
            
            return answer
        except Exception as e:
            logger.error(f"Error in vector search: {e}")
            return f"Error in vector search: {e}"
    
    return vector_query

def create_summary_query_engine(chunks: List[Document]) -> Callable[[str], str]:
    """Create summary query engine - FIXED VERSION"""
    llm = ChatOpenAI(model=LLM_MODEL, temperature=0)
    
    def summary_query(query: str) -> str:
        """Fixed summary query function"""
        try:
            if not chunks:
                return "No document chunks available for summarization."
            
            logger.info(f"Summary search for: {query}")
            
            # Better keyword matching
            query_keywords = [word.lower() for word in query.split() if len(word) > 2]
            relevant_chunks = []
            
            for chunk in chunks:
                content_lower = chunk.page_content.lower()
                if any(keyword in content_lower for keyword in query_keywords):
                    relevant_chunks.append(chunk)
            
            if not relevant_chunks:
                relevant_chunks = chunks[:15]  # Use first 15 chunks for broad queries
            
            if len(relevant_chunks) > 20:
                relevant_chunks = relevant_chunks[:20]
            
            summarize_chain = load_summarize_chain(
                llm=llm,
                chain_type="map_reduce",
                verbose=False
            )
            
            summary = summarize_chain.invoke(relevant_chunks)
            return summary["output_text"]
            
        except Exception as e:
            logger.error(f"Error in summarization: {e}")
            return f"Error in summarization: {e}"
    
    return summary_query

def create_router_query_engine(vector_query_engine: Callable, summary_query_engine: Callable) -> Callable[[str], str]:
    """Create smart router - FIXED VERSION"""
    
    def router_query(query: str) -> str:
        """Smart router that decides which engine to use"""
        query_lower = query.lower()
        
        # Use summary for broad, high-level questions
        # --- Keyword lists -----------------------------------------------------------

        vector_keywords = [
            "who is",
            "who are",   # ← NEW: ensures identity questions route to vector search
            "percentage",
            "specific",
            "according to",
            "what does",
            "statistics",
            "data",
            "numbers",
            "exhibit",
            "figure",
            "says",
        ]

        summary_keywords = [
            "overview",
            "summary",
            "key",
            "main",
            "insight",
        ]
        
        use_summary = any(keyword in query_lower for keyword in summary_keywords)
        use_vector = any(keyword in query_lower for keyword in vector_keywords)
        
        if use_summary and not use_vector:
            logger.info("🔍 Using Summary Tool for broad question")
            return summary_query_engine(query)
        else:
            logger.info("🔍 Using Vector Tool for specific question")
            return vector_query_engine(query)
    
    return router_query