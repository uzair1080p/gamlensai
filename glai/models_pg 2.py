"""
PostgreSQL models for csv_uploads table integration with n8n workflow
"""

from sqlalchemy import Column, Integer, String, Date, Double, Text, create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
import os
from dotenv import load_dotenv

load_dotenv()

Base = declarative_base()

class CsvUpload(Base):
    """Model for csv_uploads table matching the DDL structure"""
    __tablename__ = 'csv_uploads'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    source_file = Column(Text, nullable=False)
    
    # Dimension columns
    game = Column(Text)
    channel = Column(Text)
    platform = Column(Text)
    country = Column(Text)
    date = Column(Date)
    
    # Core metrics
    installs = Column(Integer)
    cost = Column(Integer)
    ad_revenue = Column(Double)
    revenue = Column(Double)
    
    # ROAS columns
    roas_d0 = Column(Double)
    roas_d1 = Column(Double)
    roas_d3 = Column(Double)
    roas_d7 = Column(Double)
    roas_d14 = Column(Text)  # TEXT as per DDL
    roas_d30 = Column(Text)
    roas_d60 = Column(Text)
    roas_d90 = Column(Text)
    
    # Retention columns
    retention_rate_d1 = Column(Double)
    retention_rate_d2 = Column(Double)
    retention_rate_d3 = Column(Double)
    retention_rate_d7 = Column(Text)  # TEXT as per DDL
    retention_rate_d14 = Column(Text)
    retention_rate_d30 = Column(Text)
    
    # Level events columns
    level_1_events = Column(Text)
    level_5_events = Column(Text)
    level_10_events = Column(Text)
    level_15_events = Column(Text)
    level_20_events = Column(Text)
    level_25_events = Column(Text)
    level_30_events = Column(Text)
    level_40_events = Column(Text)
    level_50_events = Column(Text)

# Database connection setup
def get_pg_engine():
    """Get PostgreSQL engine using DATABASE_URL from .env"""
    database_url = os.getenv("DATABASE_URL")
    if not database_url:
        raise ValueError("DATABASE_URL not found in environment variables")
    
    engine = create_engine(database_url, echo=False)
    return engine

def get_pg_session():
    """Get PostgreSQL session"""
    engine = get_pg_engine()
    Session = sessionmaker(bind=engine)
    return Session()

def get_distinct_source_files():
    """Get all distinct source file names from csv_uploads table"""
    session = get_pg_session()
    try:
        result = session.query(CsvUpload.source_file).distinct().all()
        return [row[0] for row in result if row[0]]
    finally:
        session.close()

def get_data_by_source_file(source_file):
    """Get all data for a specific source file"""
    session = get_pg_session()
    try:
        return session.query(CsvUpload).filter(CsvUpload.source_file == source_file).all()
    finally:
        session.close()
