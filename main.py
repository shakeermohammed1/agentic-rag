import os
import uuid
from src.utils import setup_logging, validate_api_keys, ensure_directories
from src.document_processor import load_and_process_documents
from src.vector_store import (
    create_vector_index, load_existing_vector_store,
    create_vector_query_engine, create_summary_query_engine,
    create_router_query_engine
)
from src.tools import create_all_tools
from src.agent import create_enhanced_agent, ask_question
from src.observability import initialize_observability, shutdown_observability
from config.settings import PDF_FILE_PATH, CHROMA_DB_DIR

logger = setup_logging()

def setup_system():
    """Setup the complete system with observability - ENHANCED VERSION"""
    logger.info("🚀 Setting up Agentic RAG system with observability...")
    
    # Initialize observability first
    logger.info("📊 Initializing observability...")
    langfuse_handler, tracer_provider = initialize_observability()
    
    # Validate setup
    validate_api_keys()
    ensure_directories()
    
    # Check if PDF exists
    if not os.path.exists(PDF_FILE_PATH):
        raise FileNotFoundError(f"PDF file not found: {PDF_FILE_PATH}")
    
    # Load and process documents
    documents, chunks = load_and_process_documents(PDF_FILE_PATH)
    
    # Always create fresh vector store for consistency
    logger.info("Creating fresh vector store...")
    vector_store = create_vector_index(chunks)
    if vector_store is None:
        raise ValueError("Failed to create vector store")
    
    # Create query engines
    vector_query_engine = create_vector_query_engine(vector_store)
    summary_query_engine = create_summary_query_engine(chunks)
    
    # Create router query engine
    router_query_engine = create_router_query_engine(vector_query_engine, summary_query_engine)
    
    # Create tools
    tools = create_all_tools()

    # Create enhanced agent with observability
    enhanced_agent = create_enhanced_agent(tools)
    
    logger.info(" System setup complete with observability!")
    return enhanced_agent

def test_system():
    """Test the system with various questions and observability"""
    logger.info(" Testing enhanced system with observability...")
    
    enhanced_agent = setup_system()
    
    # Generate a session ID for this test session
    session_id = str(uuid.uuid4())
    user_id = "test_user"
    
    test_questions = [
        "Who is Lareina Yee according to the document?",
        "What does Alexander Sukharevsky say about AI implementation?",
        "What percentage of organizations use AI according to the report?",
        "What are the main organizational changes companies are making for AI adoption?"
    ]
    
    print("\n Testing Enhanced System with Observability:")
    print("=" * 60)
    print(f" Session ID: {session_id}")
    print(f" User ID: {user_id}")
    print("=" * 60)
    
    for i, question in enumerate(test_questions, 1):
        print(f"\n Question {i}: {question}")
        print("-" * 50)
        
        try:
            response = ask_question(
                enhanced_agent, 
                question, 
                chat_history=[],
                session_id=session_id,
                user_id=user_id
            )
            print(f"🤖 Answer: {response}")
        except Exception as e:
            print(f" Error: {e}")
        
        print("=" * 60)
    
    return enhanced_agent

def interactive_mode():
    """Interactive mode for asking questions with observability"""
    enhanced_agent = setup_system()
    
    print("\n Interactive Mode with Observability - Type 'quit' to exit")
    print("=" * 50)
    
    # Generate session for this interactive session
    session_id = str(uuid.uuid4())
    user_id = input("👤 Enter your user ID (or press Enter for 'anonymous'): ").strip() or "anonymous"
    
    print(f" Session ID: {session_id}")
    print(f" User ID: {user_id}")
    print("=" * 50)
    
    chat_history = []
    
    while True:
        question = input("\n Your question: ").strip()
        
        if question.lower() in ['quit', 'exit', 'q']:
            print("👋 Goodbye!")
            break
        
        if not question:
            continue
        
        try:
            answer = ask_question(
                enhanced_agent, 
                question, 
                chat_history,
                session_id=session_id,
                user_id=user_id
            )
            print(f"\n🤖 Answer: {answer}")
            
            # Update chat history
            chat_history.append({"question": question, "answer": answer})
            if len(chat_history) > 5:
                chat_history = chat_history[-5:]
                
        except Exception as e:
            print(f" Error: {e}")

def main():
    """Main function with observability - ENHANCED VERSION"""
    print(" LangChain Agentic RAG System with Langfuse Observability")
    print("=" * 70)
    
    try:
        # Choose mode
        print("\nChoose mode:")
        print("1. Test system with sample questions")
        print("2. Interactive mode")
        
        choice = input("Enter choice (1 or 2): ").strip()
        
        if choice == "1":
            test_system()
        elif choice == "2":
            interactive_mode()
        else:
            print("Invalid choice. Running interactive mode by default.")
            interactive_mode()
            
    except Exception as e:
        logger.error(f"Error in main: {e}")
        print(f" Error: {e}")
    finally:
        # Shutdown observability
        print("\n Shutting down observability...")
        shutdown_observability()

if __name__ == "__main__":
    main()