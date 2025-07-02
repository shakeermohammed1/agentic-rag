from typing import List, Callable, TypedDict
from langchain_core.tools import Tool, tool
from langchain_community.utilities import SerpAPIWrapper, ArxivAPIWrapper

from src.utils import setup_logging
logger = setup_logging()

# ---------------------------------------------------------------------------
# McKinsey report tool with strict argument schema
# ---------------------------------------------------------------------------

from pydantic import BaseModel

class McKinseyToolInput(BaseModel):
    query: str

@tool(args_schema=McKinseyToolInput)
def mckinsey_report_tool(query: str) -> str:
    """Search the McKinsey State of AI March 2025 report for a specific question."""
    from src.document_processor import answer_question
    logger.info(f"→ mckinsey_report_tool called with query: {query}")
    return answer_question(query)

# ---------------------------------------------------------------------------
# Web and ArXiv search tools (unchanged)
# ---------------------------------------------------------------------------

def create_web_search_tool() -> Tool:
    """Create web search tool using SERP API"""
    search = SerpAPIWrapper()

    def web_search(query: str) -> str:
        try:
            return search.run(query)
        except Exception as e:
            logger.error(f"Error in web search: {e}")
            return f"Error in web search: {e}"

    return Tool(
        name="web_search",
        description="Search the web for current information, recent news, or info not in the McKinsey report.",
        func=web_search,
    )

def create_arxiv_tool() -> Tool:
    """Create ArXiv search tool"""
    arxiv = ArxivAPIWrapper()

    def arxiv_search(query: str) -> str:
        try:
            return arxiv.run(query)
        except Exception as e:
            logger.error(f"Error in ArXiv search: {e}")
            return f"Error in ArXiv search: {e}"

    return Tool(
        name="arxiv_search",
        description="Search ArXiv for academic research papers. Only use when asked for academic material.",
        func=arxiv_search,
    )

# ---------------------------------------------------------------------------
# Combine all tools
# ---------------------------------------------------------------------------

def create_all_tools() -> List[Tool]:
    """Create all tools for the agent."""
    return [
        mckinsey_report_tool,
        create_web_search_tool(),
        create_arxiv_tool(),
    ]