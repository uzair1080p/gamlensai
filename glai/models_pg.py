import os
from sqlalchemy import create_engine, Column, Integer, String, Date, Float, Text
from sqlalchemy.orm import sessionmaker
from sqlalchemy.ext.declarative import declarative_base
from dotenv import load_dotenv

load_dotenv()

DATABASE_URL = os.getenv("DATABASE_URL")
if not DATABASE_URL:
    raise ValueError("DATABASE_URL environment variable not set")

Base = declarative_base()

class CsvUpload(Base):
    __tablename__ = 'csv_uploads'

    id = Column(Integer, primary_key=True)
    source_file = Column(Text, nullable=False)
    game = Column(Text)
    channel = Column(Text)
    platform = Column(Text)
    country = Column(Text)
    date = Column(Text)  # Changed from Date to Text to match actual schema
    installs = Column(Text)  # Changed from Integer to Text
    cost = Column(Text)  # Changed from Integer to Text
    ad_revenue = Column(Text)  # Changed from Float to Text
    revenue = Column(Text)  # Changed from Float to Text
    roas_d0 = Column(Text)  # Changed from Float to Text
    roas_d1 = Column(Text)  # Changed from Float to Text
    roas_d3 = Column(Text)  # Changed from Float to Text
    roas_d7 = Column(Text)  # Changed from Float to Text
    roas_d14 = Column(Text)  # Changed from Float to Text
    roas_d30 = Column(Text)  # Changed from Float to Text
    roas_d60 = Column(Text)  # Changed from Float to Text
    roas_d90 = Column(Text)  # Changed from Float to Text
    retention_rate_d1 = Column(Text)  # Changed from Float to Text
    retention_rate_d2 = Column(Text)  # Changed from Float to Text
    retention_rate_d3 = Column(Text)  # Changed from Float to Text
    retention_rate_d7 = Column(Text)  # Changed from Float to Text
    retention_rate_d14 = Column(Text)  # Changed from Float to Text
    retention_rate_d30 = Column(Text)  # Changed from Float to Text
    level_1_events = Column(Text)  # Changed from Float to Text
    level_5_events = Column(Text)  # Changed from Float to Text
    level_10_events = Column(Text)  # Changed from Float to Text
    level_15_events = Column(Text)  # Changed from Float to Text
    level_20_events = Column(Text)  # Changed from Float to Text
    level_25_events = Column(Text)  # Changed from Float to Text
    level_30_events = Column(Text)  # Changed from Float to Text
    level_40_events = Column(Text)  # Changed from Float to Text
    level_50_events = Column(Text)  # Changed from Float to Text

    def __repr__(self):
        return f"<CsvUpload(source_file='{self.source_file}', game='{self.game}', date='{self.date}')>"

# Create engine and session
engine = create_engine(DATABASE_URL)
Session = sessionmaker(bind=engine)

def get_db_session_pg():
    return Session()

def get_distinct_source_files():
    session = get_db_session_pg()
    try:
        source_files = [row[0] for row in session.query(CsvUpload.source_file).distinct().all()]
        return source_files
    finally:
        session.close()

def get_data_by_source_file(source_file_name: str):
    session = get_db_session_pg()
    try:
        data = session.query(CsvUpload).filter(CsvUpload.source_file == source_file_name).all()
        return data
    finally:
        session.close()
