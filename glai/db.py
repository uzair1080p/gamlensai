"""
Database configuration and session management for GameLens AI
"""

import os
from pathlib import Path
from typing import Generator
from sqlalchemy import create_engine, text, event
from sqlalchemy.engine import Engine
from sqlalchemy.orm import sessionmaker, Session
from dotenv import load_dotenv

from .models import Base

# Load environment variables
load_dotenv()

# Database configuration
# Use an absolute path for the local SQLite database so Streamlit page CWD changes
# never cause accidental relative-path resolution or file access errors.
_default_sqlite_path = (
    Path(__file__).resolve().parents[1] / "gamelens.db"
)
DATABASE_URL = os.getenv("DATABASE_URL", f"sqlite:///{_default_sqlite_path}")

# Handle SQLite vs PostgreSQL
if DATABASE_URL.startswith("sqlite"):
    # For file-based SQLite, avoid StaticPool in multi-threaded Streamlit apps.
    # Enable pre_ping to recycle stale connections and set safe pragmas.
    engine = create_engine(
        DATABASE_URL,
        connect_args={"check_same_thread": False},
        pool_pre_ping=True,
    )
else:
    engine = create_engine(DATABASE_URL, pool_pre_ping=True)


if DATABASE_URL.startswith("sqlite"):
    @event.listens_for(engine, "connect")
    def _set_sqlite_pragma(dbapi_connection, connection_record):
        try:
            cursor = dbapi_connection.cursor()
            # WAL improves concurrency and reduces locking; NORMAL is a good trade-off.
            cursor.execute("PRAGMA journal_mode=WAL;")
            cursor.execute("PRAGMA synchronous=NORMAL;")
            cursor.close()
        except Exception:
            # Best effort; ignore if pragmas are not supported in the environment
            pass

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


def create_tables():
    """Create all database tables"""
    Base.metadata.create_all(bind=engine)


def get_db() -> Generator[Session, None, None]:
    """Dependency to get database session"""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def get_db_session() -> Session:
    """Get a database session (for direct use)"""
    return SessionLocal()


def test_connection():
    """Test database connection"""
    try:
        with engine.connect() as conn:
            result = conn.execute(text("SELECT 1"))
            return result.fetchone()[0] == 1
    except Exception as e:
        print(f"Database connection failed: {e}")
        return False


def init_database():
    """Initialize database with tables"""
    try:
        create_tables()
        print("✅ Database tables created successfully")
        return True
    except Exception as e:
        print(f"❌ Failed to create database tables: {e}")
        return False
