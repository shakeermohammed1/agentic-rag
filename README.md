# Agentic RAG System with Langfuse Observability

An advanced Retrieval-Augmented Generation (RAG) system built with LangChain, featuring comprehensive observability through Langfuse for conversation tracking, performance monitoring, and system analytics.

## Features

- **Document Processing**: Processes McKinsey AI State Report PDF
- **Vector Search**: Semantic search using ChromaDB and OpenAI embeddings
- **Web Search**: Real-time web search capabilities via SerpAPI
- **Academic Search**: ArXiv paper search integration
- **Agentic Workflow**: Intelligent tool selection and execution
- **Langfuse Observability**: Complete conversation tracking and analytics
- **Modern Web UI**: Beautiful Flask-based interface
- **API Endpoints**: RESTful API for integration

## Enhanced Observability Features

### Langfuse Integration
- **Conversation Tracking**: Every user interaction is logged with session IDs
- **Tool Usage Analytics**: Monitor which tools are being used and how often
- **Performance Metrics**: Track response times and system performance
- **Error Monitoring**: Comprehensive error tracking and debugging
- **User Journey Mapping**: Understand user behavior patterns
- **Cost Tracking**: Monitor API usage and costs

### What Gets Tracked
- **User Questions**: Complete conversation history
- **Agent Responses**: AI-generated answers with metadata
- **Tool Executions**: Which tools were called and their outputs
- **Response Times**: Performance metrics for optimization
- **Errors**: Detailed error logs for debugging
- **User Sessions**: Anonymous session tracking for analytics

## Installation & Setup

### 1. Environment Setup

```bash
# Clone the repository
git clone <your-repo-url>
cd agentic-rag-langfuse

# Create conda environment
conda env create -f environment.yaml
conda activate agentic-rag

# Install additional Langfuse dependencies if needed
pip install langfuse opentelemetry-sdk opentelemetry-exporter-otlp
```

### 2. Environment Variables

Create a `.env` file in the root directory:

```env
# API Keys for the RAG System
OPENAI_API_KEY=your_openai_api_key_here
SERPAPI_API_KEY=your_serpapi_key_here

# Langfuse Observability Keys
LANGFUSE_PUBLIC_KEY=''
LANGFUSE_SECRET_KEY=''
LANGFUSE_HOST=https://cloud.langfuse.com

# Enable/Disable Observability
ENABLE_LANGFUSE=true

# Flask Configuration
FLASK_ENV=development
PORT=8004
```

### 3. Document Setup

Place the McKinsey AI State Report PDF in the `data/` directory:
```bash
mkdir -p data
# Copy state.pdf to data/state.pdf
```

## Usage

### Web Interface (Recommended)

```bash
# Start the Flask application with observability
python app.py
```

Visit `http://localhost:8004` to access the beautiful web interface with real-time observability tracking.

### Command Line Interface

```bash
# Interactive mode with observability
python main.py
```

### Programmatic API

```python
from src.agent import create_enhanced_agent, ask_question
from src.tools import create_all_tools
from src.observability import initialize_observability

# Initialize observability
initialize_observability()

# Create agent
tools = create_all_tools()
agent = create_enhanced_agent(tools)

# Ask questions with session tracking
answer = ask_question(
    agent, 
    "Who is Lareina Yee according to the document?",
    session_id="my_session_123",
    user_id="user_456"
)
```

## Monitoring with Langfuse

### Accessing Your Dashboard

1. **Login to Langfuse**: Visit [cloud.langfuse.com](https://cloud.langfuse.com)
2. **View Conversations**: See all tracked conversations in real-time
3. **Analyze Performance**: Monitor response times and success rates
4. **Track Costs**: Monitor OpenAI API usage and costs
5. **Debug Issues**: Detailed error logs and traces

### Key Metrics to Monitor

- **Response Time**: How fast is your system responding?
- **Success Rate**: Percentage of successful vs failed interactions
- **Tool Usage**: Which tools are most frequently used?
- **User Patterns**: How are users interacting with your system?
- **Cost Analysis**: Track spending on OpenAI and other APIs

### Sample Langfuse Views

1. **Traces View**: See the complete flow of each conversation
2. **Sessions View**: Group conversations by user sessions
3. **Analytics Dashboard**: High-level metrics and trends
4. **Playground**: Test and debug your prompts

## API Endpoints

### Health Check
```bash
GET /api/health
```

### Initialize System
```bash
POST /api/initialize
```

### Ask Question (with Observability)
```bash
POST /api/ask
Content-Type: application/json

{
  "question": "Who is Lareina Yee according to the document?",
  "session_id": "optional_session_id",
  "user_id": "optional_user_id"
}
```

## Testing with Observability

Run the test suite to see observability in action:

```bash
# Run tests with sample questions
python main.py
# Choose option 1 for test mode

# Check Langfuse dashboard to see traced conversations
```

## Privacy & Security

- **Anonymous by Default**: Users are tracked anonymously unless they provide a user ID
- **No PII Storage**: No personally identifiable information is stored
- **Session-Based**: All tracking is session-based and temporary
- **Configurable**: Observability can be completely disabled via environment variables

## Configuration Options

### Disable Observability
```env
ENABLE_LANGFUSE=false
```

### Custom Langfuse Instance
```env
LANGFUSE_HOST=https://your-custom-langfuse-instance.com
```

### Logging Levels
```python
# In settings.py
LOG_LEVEL = "DEBUG"  # For detailed observability logs
```

## Performance Optimization

The observability layer is designed to be lightweight:

- **Async Logging**: Non-blocking observability calls
- **Batched Exports**: Efficient data transmission to Langfuse
- **Minimal Overhead**: < 50ms additional latency per request
- **Error Resilience**: System continues working even if observability fails

## Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Troubleshooting

### Observability Issues

**Problem**: Langfuse not tracking conversations
```bash
# Check your environment variables
echo $LANGFUSE_PUBLIC_KEY
echo $LANGFUSE_SECRET_KEY

# Check logs for observability errors
tail -f app.log | grep -i langfuse
```

**Problem**: High latency with observability
```bash
# Disable observability temporarily
export ENABLE_LANGFUSE=false
python app.py
```

**Problem**: Langfuse connection errors
```bash
# Test connectivity
curl -H "Authorization: Basic $(echo -n $LANGFUSE_PUBLIC_KEY:$LANGFUSE_SECRET_KEY | base64)" \
     https://cloud.langfuse.com/api/public/health
```

### General Issues

**Problem**: Vector store not loading
```bash
# Delete and recreate vector store
rm -rf ./chroma_db_langchain
python main.py
```

**Problem**: API key issues
```bash
# Verify API keys are set
python -c "from src.utils import validate_api_keys; validate_api_keys()"
```

## Support

- **Issues**: Open a GitHub issue
- **Documentation**: Check the `/docs` folder
- **Langfuse Help**: Visit [Langfuse Documentation](https://langfuse.com/docs)

---

**Built with love using LangChain, Langfuse, and modern AI technologies**