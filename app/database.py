from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from urllib.parse import quote_plus
import time
from sqlalchemy.exc import OperationalError
from sqlalchemy.ext.declarative import declarative_base

# Encode special characters in the password
password = quote_plus("S@ny_Device_1997")
database_url = (
    f"mssql+pyodbc://sa:{password}@sql_server_db:1433/mydatabase?"
    f"driver=ODBC+Driver+17+for+SQL+Server"
)

# SQLAlchemy engine and session
engine = create_engine(database_url)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

# Dependency for database session
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# Function to retry database connection
from sqlalchemy import text

def connect_with_retry():
    retries = 10
    for attempt in range(retries):
        try:
            engine = create_engine(database_url)
            with engine.connect() as connection:
                # Use the text() function to safely execute a SQL query
                result = connection.execute(text("SELECT 1")).scalar()
                if result == 1:
                    print("Connected successfully!")
            return engine
        except OperationalError:
            print(f"Connection attempt {attempt + 1} failed, retrying...")
            time.sleep(10)  # Increased sleep time between retries
    raise Exception("Unable to connect to SQL Server after multiple attempts.")



    # Try to connect to the database with retries
engine = connect_with_retry()

from models import Employee, FaceVector, Transaction, Camera
# Base.metadata.drop_all(engine)
# Base.metadata.drop_all(engine)
Base.metadata.create_all(bind=engine)  # This will create the tables
