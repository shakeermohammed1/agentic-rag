from __future__ import annotations

from typing import List, Optional
import uuid

from langchain_core.tools import Tool
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langchain.agents import create_openai_tools_agent, AgentExecutor
from langchain_openai import ChatOpenAI

from config.settings import LLM_MODEL
from src.utils import setup_logging
from src.observability import get_langfuse_handler, create_trace, log_generation

logger = setup_logging()

# ---------------------------------------------------------------------------
# System prompt - VERY DIRECT AND SIMPLE
# ---------------------------------------------------------------------------

def create_system_prompt() -> ChatPromptTemplate:
    """Return a simple, direct system prompt that forces tool usage."""

    return ChatPromptTemplate.from_messages([
        ("system", """You are a McKinsey AI report assistant. You MUST use tools to answer questions.

RULES:
1. For questions about the McKinsey report, use mckinsey_report_tool with the user's exact question
2. For web search questions, use web_search  
3. For academic papers, use arxiv_search
4. You MUST call a tool for every question - never give direct answers without using tools
5. Pass the user's question exactly as they wrote it to mckinsey_report_tool

Example:
User: "Who is Lareina Yee according to the document?"
You MUST call: mckinsey_report_tool with query "Who is Lareina Yee according to the document?"

ALWAYS USE TOOLS. DO NOT ANSWER WITHOUT USING TOOLS."""),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{input}"),
        MessagesPlaceholder(variable_name="agent_scratchpad"),
    ])

# ---------------------------------------------------------------------------
# Enhanced Agent factory with observability
# ---------------------------------------------------------------------------

def create_enhanced_agent(tools: List[Tool]) -> AgentExecutor:
    """Create a simple agent that always uses tools with Langfuse observability."""

    # Get Langfuse callback handler
    langfuse_handler = get_langfuse_handler()
    callbacks = [langfuse_handler] if langfuse_handler else []

    llm = ChatOpenAI(
        model=LLM_MODEL,
        temperature=0,
        callbacks=callbacks  # Add Langfuse callback to LLM
    )

    agent = create_openai_tools_agent(
        llm=llm,
        tools=tools,
        prompt=create_system_prompt(),
    )

    executor = AgentExecutor(
        agent=agent,
        tools=tools,
        verbose=True,
        handle_parsing_errors=True,
        max_iterations=2,
        early_stopping_method="force",
        callbacks=callbacks  # Add Langfuse callback to agent executor
    )

    logger.info("Created enhanced agent with Langfuse observability")
    return executor

# ---------------------------------------------------------------------------
# Helper function with observability
# ---------------------------------------------------------------------------

def ask_question(
    agent_executor: AgentExecutor,
    question: str,
    chat_history: List | None = None,
    session_id: Optional[str] = None,
    user_id: Optional[str] = None,
) -> str:
    """Ask a question using the agent with full observability."""

    # Generate session ID if not provided
    if not session_id:
        session_id = str(uuid.uuid4())

    # Create Langfuse trace for this conversation
    trace = create_trace(session_id, user_id)

    raw_history = chat_history or []
    formatted_history: List = []

    for turn in raw_history:
        if isinstance(turn, dict):
            q = turn.get("question")
            a = turn.get("answer")
            if q:
                formatted_history.append(HumanMessage(content=q))
            if a:
                formatted_history.append(AIMessage(content=a))
        else:
            formatted_history.append(turn)

    try:
        # Get Langfuse callback handler
        langfuse_handler = get_langfuse_handler()
        callbacks = [langfuse_handler] if langfuse_handler else []

        logger.info(f"🤖 Processing question with session_id: {session_id}")
        
        response = agent_executor.invoke(
            {
                "input": question,
                "chat_history": formatted_history,
            },
            config={
                "callbacks": callbacks,
                "metadata": {
                    "session_id": session_id,
                    "user_id": user_id,
                    "question_type": "agentic_rag"
                }
            }
        )
        
        answer = response["output"]
        
        # Log the final generation to Langfuse
        log_generation(
            name="agentic_rag_response",
            input_text=question,
            output_text=answer,
            model=LLM_MODEL,
            metadata={
                "session_id": session_id,
                "user_id": user_id,
                "tools_used": [tool.name for tool in agent_executor.tools]
            }
        )
        
        logger.info(f"✅ Question answered successfully for session: {session_id}")
        return answer
        
    except Exception as exc:
        error_msg = f"Error: {exc}"
        logger.error(f"❌ Error while asking question: {exc}")
        
        # Log the error generation
        log_generation(
            name="agentic_rag_error",
            input_text=question,
            output_text=error_msg,
            model=LLM_MODEL,
            metadata={
                "session_id": session_id,
                "user_id": user_id,
                "error": str(exc)
            }
        )
        
        return error_msg