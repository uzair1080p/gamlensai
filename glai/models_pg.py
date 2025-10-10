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
    import pandas as pd
    session = get_db_session_pg()
    try:
        data = session.query(CsvUpload).filter(CsvUpload.source_file == source_file_name).all()
        
        # Convert SQLAlchemy objects to pandas DataFrame
        if data:
            # Convert each object to a dictionary
            data_dicts = []
            for row in data:
                row_dict = {}
                for column in CsvUpload.__table__.columns:
                    row_dict[column.name] = getattr(row, column.name)
                data_dicts.append(row_dict)
            
            # Create DataFrame
            df = pd.DataFrame(data_dicts)
            return df
        else:
            # Return empty DataFrame with expected columns
            return pd.DataFrame(columns=[
                'source_file', 'game', 'channel', 'platform', 'country', 'date',
                'installs', 'cost', 'ad_revenue', 'revenue',
                'roas_d0', 'roas_d1', 'roas_d3', 'roas_d7', 'roas_d14', 'roas_d30', 'roas_d60', 'roas_d90',
                'retention_rate_d1', 'retention_rate_d2', 'retention_rate_d3', 'retention_rate_d7', 'retention_rate_d14', 'retention_rate_d30',
                'level_1_events', 'level_5_events', 'level_10_events', 'level_15_events', 'level_20_events', 'level_25_events', 'level_30_events', 'level_40_events', 'level_50_events'
            ])
    finally:
        session.close()
