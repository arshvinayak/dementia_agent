#!/usr/bin/env python3
"""
Database setup and initialization script.
Creates both:
1. PostgreSQL schema for tasks, reminders, medications (SQLAlchemy models)
2. LangGraph PostgreSQL store for agent memory

Follows LangChain best practices from:
https://docs.langchain.com/oss/python/langgraph/add-memory#database-management
"""

import os
import sys
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Ensure GEMINI_API_KEY is available
if "GEMINI_API_KEY" not in os.environ:
    os.environ["GEMINI_API_KEY"] = os.getenv("GEMINI_API_KEY", "")


def setup_database():
    """Initialize the database schema."""
    
    print("=" * 60)
    print("Database Initialization")
    print("=" * 60)
    
    db_uri = os.getenv(
        "DATABASE_URL",
        "postgresql://postgres:postgres@localhost:5432/mini_proj?sslmode=disable"
    )
    
    try:
        # Initialize SQLAlchemy models
        print("\n[1/2] Initializing SQLAlchemy models (tasks, reminders, medications)...")
        from database import init_db_sync
        init_db_sync()
        print("✓ SQLAlchemy models initialized")
        
        # Initialize LangGraph PostgreSQL store
        print("\n[2/2] Initializing LangGraph PostgreSQL store...")
        from langgraph.store.postgres import PostgresStore  # type: ignore[import-not-found]
        from langgraph.store.base import IndexConfig
        from langchain_google_genai import GoogleGenerativeAIEmbeddings
        
        # Initialize embeddings
        embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")
        
        # Create store with embeddings for semantic search using context manager
        with PostgresStore.from_conn_string(
            db_uri,
            index=IndexConfig(
                embed=embeddings.embed_documents,
                dims=1536
            ),
        ) as store:
            store.setup()
            print("✓ LangGraph PostgreSQL store initialized")
        
        print("\n" + "=" * 60)
        print("✓ Database initialization completed successfully!")
        print("=" * 60)
        print(f"Connected to: {db_uri.split('?')[0]}")
        print("\nYou can now start the application:")
        print("  python src/agent.py")
        
        return True
        
    except ImportError as e:
        print(f"\n✗ Missing dependency: {e}")
        print("Install with: pip install -r requirements.txt")
        return False
        
    except Exception as e:
        print(f"\n✗ Error initializing database: {e}")
        print("\nTroubleshooting:")
        print("1. Verify PostgreSQL is running: docker-compose ps")
        print("2. Check connection string in .env")
        print("3. Verify database exists and user has permissions")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = setup_database()
    sys.exit(0 if success else 1)

