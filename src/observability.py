"""
Observability module for Langfuse integration with LangChain
Includes fallback handling when Langfuse is not available
"""
import os
from typing import Optional
from src.utils import setup_logging
from config.settings import ENABLE_LANGFUSE

logger = setup_logging()
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Set Langfuse credentials from environment
os.environ["LANGFUSE_PUBLIC_KEY"] = os.getenv("LANGFUSE_PUBLIC_KEY")
os.environ["LANGFUSE_SECRET_KEY"] = os.getenv("LANGFUSE_SECRET_KEY")
os.environ["LANGFUSE_HOST"] = os.getenv("LANGFUSE_HOST", "https://cloud.langfuse.com")  # Default fallback

# Optional: base64 auth if needed
import base64
auth = base64.b64encode(
    f'{os.environ["LANGFUSE_PUBLIC_KEY"]}:{os.environ["LANGFUSE_SECRET_KEY"]}'.encode()
).decode()


# Try to import Langfuse components
try:
    from langfuse.openai import LangfuseCallbackHandler
    from langfuse import Langfuse
    LANGFUSE_AVAILABLE = True
    logger.info("✅ Langfuse imports successful")
except ImportError as e:
    logger.warning(f"⚠️ Langfuse not available: {e}")
    logger.warning("📊 Running without observability - install langfuse to enable tracking")
    LANGFUSE_AVAILABLE = False

    # Create mock classes for fallback
    class CallbackHandler:
        def __init__(self, *args, **kwargs):
            pass

    class Langfuse:
        def __init__(self, *args, **kwargs):
            pass

        def trace(self, *args, **kwargs):
            return None

        def generation(self, *args, **kwargs):
            pass

        def flush(self):
            pass

# Try to import OpenTelemetry components
try:
    from opentelemetry.sdk.resources import Resource
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import BatchSpanProcessor
    from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
    from opentelemetry import trace
    import base64
    OPENTELEMETRY_AVAILABLE = True
    logger.info("✅ OpenTelemetry imports successful")
except ImportError as e:
    logger.warning(f"⚠️ OpenTelemetry not available: {e}")
    OPENTELEMETRY_AVAILABLE = False

# Import settings with fallback
try:
    from config.settings import (
        LANGFUSE_PUBLIC_KEY,
        LANGFUSE_SECRET_KEY,
        LANGFUSE_HOST
    )
except ImportError:
    logger.warning("⚠️ Settings not found, using defaults")
    LANGFUSE_PUBLIC_KEY = None
    LANGFUSE_SECRET_KEY = None
    LANGFUSE_HOST = "https://cloud.langfuse.com"

class ObservabilityManager:
    """Manages observability setup for the RAG system with fallback handling"""

    def __init__(self):
        self.langfuse_handler: Optional[object] = None
        self.langfuse_client: Optional[object] = None
        self.tracer_provider: Optional = None
        self.enabled = ENABLE_LANGFUSE and LANGFUSE_AVAILABLE

    def initialize_langfuse(self) -> Optional[object]:
        """Initialize Langfuse for LangChain observability with fallback"""
        if not self.enabled:
            logger.info("📊 Langfuse observability is disabled or unavailable")
            return None

        if not all([LANGFUSE_PUBLIC_KEY, LANGFUSE_SECRET_KEY]):
            logger.warning("⚠️ Langfuse credentials not found. Observability disabled.")
            return None

        try:
            # Initialize Langfuse client
            self.langfuse_client = Langfuse(
                public_key=LANGFUSE_PUBLIC_KEY,
                secret_key=LANGFUSE_SECRET_KEY,
                host=LANGFUSE_HOST
            )

            # Create callback handler for LangChain
            self.langfuse_handler = LangfuseCallbackHandler(
                public_key=LANGFUSE_PUBLIC_KEY,
                secret_key=LANGFUSE_SECRET_KEY,
                host=LANGFUSE_HOST
            )

            logger.info("✅ Langfuse observability initialized successfully")
            return self.langfuse_handler

        except Exception as e:
            logger.error(f"❌ Failed to initialize Langfuse: {e}")
            return None

    def initialize_opentelemetry(self) -> Optional:
        """Initialize OpenTelemetry for detailed tracing with fallback"""
        if not self.enabled or not OPENTELEMETRY_AVAILABLE:
            return None

        if not all([LANGFUSE_PUBLIC_KEY, LANGFUSE_SECRET_KEY]):
            return None

        try:
            # Create authentication header
            auth = base64.b64encode(
                f'{LANGFUSE_PUBLIC_KEY}:{LANGFUSE_SECRET_KEY}'.encode()
            ).decode()

            # Setup tracer provider
            self.tracer_provider = TracerProvider(
                resource=Resource.create({"service.name": "agentic-rag-system"})
            )

            # Setup OTLP exporter to Langfuse
            otlp_exporter = OTLPSpanExporter(
                endpoint=f'{LANGFUSE_HOST}/api/public/otel',
                headers={"Authorization": f"Basic {auth}"},
            )

            self.tracer_provider.add_span_processor(
                BatchSpanProcessor(otlp_exporter)
            )
            trace.set_tracer_provider(self.tracer_provider)

            logger.info("✅ OpenTelemetry tracing initialized successfully")
            return self.tracer_provider

        except Exception as e:
            logger.error(f"❌ Failed to initialize OpenTelemetry: {e}")
            return None

    def get_langfuse_handler(self) -> Optional[object]:
        """Get the Langfuse callback handler"""
        return self.langfuse_handler if self.enabled else None

    def create_trace_session(self, session_id: str, user_id: Optional[str] = None):
        """Create a new trace session"""
        if self.langfuse_client and self.enabled:
            try:
                return self.langfuse_client.trace(
                    id=session_id,
                    user_id=user_id,
                    session_id=session_id
                )
            except Exception as e:
                logger.error(f"Failed to create trace: {e}")
        return None

    def log_generation(self,
                      name: str,
                      input_text: str,
                      output_text: str,
                      model: str,
                      metadata: dict = None):
        """Log a generation event"""
        if self.langfuse_client and self.enabled:
            try:
                self.langfuse_client.generation(
                    name=name,
                    input=input_text,
                    output=output_text,
                    model=model,
                    metadata=metadata or {}
                )
            except Exception as e:
                logger.error(f"Failed to log generation: {e}")

    def shutdown(self):
        """Shutdown observability components"""
        try:
            if self.langfuse_client and self.enabled:
                self.langfuse_client.flush()
            if self.tracer_provider and OPENTELEMETRY_AVAILABLE:
                self.tracer_provider.shutdown()
            logger.info("📊 Observability components shutdown successfully")
        except Exception as e:
            logger.error(f"Error during observability shutdown: {e}")


# Global observability manager instance
observability_manager = ObservabilityManager()

def initialize_observability():
    """Initialize all observability components with fallback"""
    logger.info("🔧 Initializing observability...")

    if not LANGFUSE_AVAILABLE:
        logger.info("📊 Running in fallback mode - no observability tracking")
        return None, None

    # Initialize Langfuse
    langfuse_handler = observability_manager.initialize_langfuse()

    # Initialize OpenTelemetry
    tracer_provider = observability_manager.initialize_opentelemetry()

    return langfuse_handler, tracer_provider

def get_langfuse_handler() -> Optional[object]:
    """Get the global Langfuse handler"""
    return observability_manager.get_langfuse_handler()

def create_trace(session_id: str, user_id: Optional[str] = None):
    """Create a new trace session"""
    return observability_manager.create_trace_session(session_id, user_id)

def log_generation(name: str, input_text: str, output_text: str, model: str, metadata: dict = None):
    """Log a generation event"""
    observability_manager.log_generation(name, input_text, output_text, model, metadata)

def shutdown_observability():
    """Shutdown observability components"""
    observability_manager.shutdown()
