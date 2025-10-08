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
    date = Column(Date)
    installs = Column(Integer)
    cost = Column(Integer)
    ad_revenue = Column(Float)
    revenue = Column(Float)
    roas_d0 = Column(Float)
    roas_d1 = Column(Float)
    roas_d3 = Column(Float)
    roas_d7 = Column(Float)
    roas_d14 = Column(Float)
    roas_d30 = Column(Float)
    roas_d60 = Column(Float)
    roas_d90 = Column(Float)
    retention_rate_d1 = Column(Float)
    retention_rate_d2 = Column(Float)
    retention_rate_d3 = Column(Float)
    retention_rate_d7 = Column(Float)
    retention_rate_d14 = Column(Float)
    retention_rate_d30 = Column(Float)
    level_1_events = Column(Float)
    level_5_events = Column(Float)
    level_10_events = Column(Float)
    level_15_events = Column(Float)
    level_20_events = Column(Float)
    level_25_events = Column(Float)
    level_30_events = Column(Float)
    level_40_events = Column(Float)
    level_50_events = Column(Float)

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
