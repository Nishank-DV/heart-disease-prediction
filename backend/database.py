"""
Database Configuration and Setup
SQLite database with SQLAlchemy ORM
"""

from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
import os

try:
    # Prefer central config / environment-based URL
    from config import DATABASE_URL as SQLALCHEMY_DATABASE_URL
except ImportError:
    # Fallback to a sane local default
    SQLALCHEMY_DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./heart_disease_predictions.db")

# Create engine
# connect_args={"check_same_thread": False} is needed for SQLite
engine = create_engine(
    SQLALCHEMY_DATABASE_URL,
    connect_args={"check_same_thread": False},
    echo=False  # Set to True for SQL query logging
)

# Create SessionLocal class
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# Base class for models
Base = declarative_base()


def get_db():
    """
    Database dependency for FastAPI routes
    
    Yields:
        Database session
    """
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def init_db():
    """
    Initialize database - create all tables
    Call this once at application startup
    """
    Base.metadata.create_all(bind=engine)
    print("[OK] Database initialized successfully!")


def reset_db():
    """
    Reset database - drop all tables and recreate
    WARNING: This will delete all data!
    """
    Base.metadata.drop_all(bind=engine)
    Base.metadata.create_all(bind=engine)
    print("[OK] Database reset successfully!")

