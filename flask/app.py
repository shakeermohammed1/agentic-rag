#!/usr/bin/env python3
"""
Flask web application with Langfuse observability for the Agentic RAG System
"""

import sys
import os
import uuid

# Add the parent directory to Python path to find src modules
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)

from flask import Flask, request, jsonify, render_template
import logging
import traceback

# Import your existing system
from src.utils import setup_logging, validate_api_keys, ensure_directories
from src.document_processor import load_and_process_documents
from src.vector_store import create_vector_index
from src.tools import create_all_tools
from src.agent import create_enhanced_agent, ask_question
from src.observability import initialize_observability, shutdown_observability
from config.settings import PDF_FILE_PATH

# Initialize Flask app
app = Flask(__name__)
app.secret_key = os.urandom(24)  # For session management

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global variables for the system
agent = None
system_initialized = False
langfuse_handler = None

def initialize_system():
    """Initialize the agentic RAG system with observability"""
    global agent, system_initialized, langfuse_handler
    
    try:
        logger.info("Initializing Agentic RAG System with Langfuse observability...")
        
        # Initialize observability first
        logger.info("Setting up observability...")
        langfuse_handler, tracer_provider = initialize_observability()
        
        # Validate setup
        validate_api_keys()
        ensure_directories()
        
        # Check if PDF exists
        if not os.path.exists(PDF_FILE_PATH):
            raise FileNotFoundError(f"PDF file not found: {PDF_FILE_PATH}")
        
        # Load and process documents
        logger.info("Loading documents...")
        documents, chunks = load_and_process_documents(PDF_FILE_PATH)
        
        # Create vector store
        logger.info("Creating vector store...")
        vector_store = create_vector_index(chunks)
        if vector_store is None:
            raise ValueError("Failed to create vector store")
        
        # Create tools and agent
        logger.info("Creating tools and agent with observability...")
        tools = create_all_tools()
        agent = create_enhanced_agent(tools)
        
        system_initialized = True
        logger.info("System initialized successfully with observability!")
        return True
        
    except Exception as e:
        logger.error(f"Error initializing system: {e}")
        traceback.print_exc()
        return False

@app.route('/')
def home():
    """Serve the main UI"""
    return render_template('index.html')

@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'system_initialized': system_initialized,
        'observability_enabled': langfuse_handler is not None,
        'message': 'Agentic RAG System with Langfuse observability is running'
    })

@app.route('/api/initialize', methods=['POST'])
def initialize():
    """Initialize the system with observability"""
    try:
        success = initialize_system()
        if success:
            return jsonify({
                'status': 'success',
                'message': 'System initialized successfully with Langfuse observability',
                'observability_enabled': langfuse_handler is not None
            })
        else:
            return jsonify({
                'status': 'error',
                'message': 'Failed to initialize system'
            }), 500
    except Exception as e:
        logger.error(f"Error in initialize endpoint: {e}")
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500

@app.route('/api/ask', methods=['POST'])
def ask_question_endpoint():
    """Ask a question to the agentic RAG system with observability"""
    try:
        # Check if system is initialized
        if not system_initialized or agent is None:
            return jsonify({
                'status': 'error',
                'message': 'System not initialized. Please initialize first.'
            }), 400
        
        # Get question from request
        data = request.get_json()
        if not data or 'question' not in data:
            return jsonify({
                'status': 'error',
                'message': 'No question provided'
            }), 400
        
        question = data['question'].strip()
        if not question:
            return jsonify({
                'status': 'error',
                'message': 'Empty question provided'
            }), 400
        
        # Get session info for observability
        session_id = data.get('session_id')
        user_id = data.get('user_id', 'anonymous')
        
        # Generate session ID if not provided
        if not session_id:
            session_id = str(uuid.uuid4())
        
        logger.info(f"Question received: {question} (Session: {session_id}, User: {user_id})")
        
        # Ask the question with observability
        response = ask_question(
            agent,
            question,
            chat_history=[],
            session_id=session_id,
            user_id=user_id
        )
        
        logger.info("Answer generated successfully with observability")
        
        return jsonify({
            'status': 'success',
            'question': question,
            'answer': response,
            'session_id': session_id,
            'user_id': user_id,
            'observability_enabled': langfuse_handler is not None,
            'message': 'Question answered successfully with observability tracking'
        })
        
    except Exception as e:
        logger.error(f"Error processing question: {e}")
        traceback.print_exc()
        return jsonify({
            'status': 'error',
            'message': f'Error processing question: {str(e)}'
        }), 500

@app.teardown_appcontext
def close_observability(error):
    """Clean up observability on app teardown"""
    if error:
        logger.error(f"App context teardown with error: {error}")

if __name__ == '__main__':
    # Initialize system on startup
    logger.info("Starting Flask application with Langfuse observability...")
    
    try:
        # Run the app
        port = int(os.getenv('PORT', 8004))
        debug = os.getenv('FLASK_ENV') == 'development'
        
        app.run(
            host='0.0.0.0',
            port=port,
            debug=debug
        )
    finally:
        # Shutdown observability on exit
        logger.info("Shutting down observability...")
        shutdown_observability()