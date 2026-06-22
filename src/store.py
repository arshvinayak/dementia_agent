"""
Memory store initialization for the agent.
Uses PostgreSQL via LangGraph's PostgresStore for persistent storage with semantic embeddings.

Following LangChain best practices for long-term memory:
https://docs.langchain.com/oss/python/langchain/long-term-memory
"""

import os
from typing import Optional
from dotenv import load_dotenv
from langgraph.store.base import IndexConfig

# Load environment variables
load_dotenv()

# Ensure GEMINI_API_KEY is available
if "GEMINI_API_KEY" not in os.environ:
    os.environ["GEMINI_API_KEY"] = os.getenv("GEMINI_API_KEY", "")

# Database configuration
DB_URI = os.getenv(
    "DATABASE_URL",
    "postgresql://postgres:postgres@localhost:5432/mini_proj?sslmode=disable"
)

# Global store instance
_store: Optional[object] = None
_embeddings = None


def _get_embeddings():
    """Lazily initialize embeddings (deferred until needed)."""
    global _embeddings
    if _embeddings is None:
        from langchain_google_genai import GoogleGenerativeAIEmbeddings
        _embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")
    return _embeddings


def get_store():
    """
    Get the PostgreSQL memory store (singleton for app lifetime).
    
    Returns:
        LangGraph PostgresStore object for persistent storage
        
    Raises:
        ImportError: If langgraph.store.postgres is not installed
        Exception: If PostgreSQL connection fails
    """
    global _store
    
    if _store is not None:
        return _store
    
    # Get embeddings
    embeddings = _get_embeddings()
    
    # Initialize PostgreSQL store
    from langgraph.store.postgres import PostgresStore  # type: ignore[import-not-found]
    
    _store = PostgresStore.from_conn_string(
        DB_URI,
        index=IndexConfig(
            embed=embeddings.embed_documents,
            dims=1536
        ),
    ).__enter__()
    print("✓ Using PostgreSQL store for persistent memory")
    return _store


def close_store():
    """Close the store connection."""
    global _store
    if _store is not None:
        try:
            _store.close()
        except (AttributeError, Exception):
            pass
        _store = None
