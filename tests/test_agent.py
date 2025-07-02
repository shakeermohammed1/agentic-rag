import unittest
import os
import sys
from unittest.mock import patch, MagicMock

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from src.document_processor import split_documents
from src.vector_store import create_embeddings
from src.utils import validate_api_keys

class TestAgenticRAG(unittest.TestCase):
    
    def test_split_documents(self):
        """Test document splitting"""
        # Mock document
        mock_doc = MagicMock()
        mock_doc.page_content = "This is a test document " * 100
        mock_doc.metadata = {}
        
        chunks = split_documents([mock_doc], chunk_size=100, chunk_overlap=20)
        self.assertGreater(len(chunks), 1)
    
    @patch.dict(os.environ, {'OPENAI_API_KEY': 'test-key', 'SERPAPI_API_KEY': 'test-key'})
    def test_validate_api_keys(self):
        """Test API key validation"""
        try:
            validate_api_keys()
        except ValueError:
            self.fail("validate_api_keys() raised ValueError unexpectedly!")
    
    def test_create_embeddings(self):
        """Test embeddings creation"""
        with patch.dict(os.environ, {'OPENAI_API_KEY': 'test-key'}):
            embeddings = create_embeddings()
            self.assertIsNotNone(embeddings)

if __name__ == '__main__':
    unittest.main()