from fastapi import FastAPI, UploadFile, File, Form, Depends
from sqlalchemy.orm import Session

# import module
from rabbitmq import start_consumer
from redis_client import redis_client
from face_recognition import extract_face_vector
from database import *
from models import *

# import lib
import numpy as np
import pickle
import cv2
import asyncio
import threading
import logging


app = FastAPI()

@app.get("/")
def read_root():
    return {"Hello": "World"}

# Function to run Consumer in background thread.
def start_consumer_thread():
    def run_consumer():
        start_consumer()

    # Run start_consumer() in the background thread.
    consumer_thread = threading.Thread(target=run_consumer)
    consumer_thread.daemon = True  # Let it run in the background
    consumer_thread.start()

@app.on_event("startup")
async def startup():
    logging.info("🚀 FastAPI starts...")
    start_consumer_thread()

@app.on_event("startup")
async def startup():
    # Connect to the database via session
    db: Session = next(get_db())

    try:
        # Delete all data in Redis related to face_vector
        keys = redis_client.keys("face_vector:*")
        if keys:
            redis_client.delete(*keys)
            logging.info("✅ Deleted old face_vector data from Redis")

        # Get all data from FaceVector table
        face_vectors = db.query(FaceVector).all()

        for face_vector in face_vectors:
            # Convert data to dictionary
            face_vector_dict = face_vector.to_dict()

            # Create Redis Key for emp_id
            redis_key = f"face_vector:{face_vector.emp_id}"

            # Store data in Redis (use Pickle to store byte data)
            redis_client.set(redis_key, pickle.dumps(face_vector_dict))

        logging.info("✅ Synced data from FaceVector to Redis")

    finally:
        db.close()

@app.get("/api/vector-redis")
def get_all_face_vectors():
    # Get all keys associated with face_vector
    keys = redis_client.keys("face_vector:*")

    if not keys:
        raise HTTPException(status_code=404, detail="No face vectors found in Redis")

    face_vectors = []

    for key in keys:
        # Fetch data from Redis 
        data = redis_client.get(key)

        if data:
            # Convert data from binary back to dictionary
            face_vector_dict = pickle.loads(data)
            face_vectors.append(face_vector_dict)

    return {"face_vectors": face_vectors}
    
# API get emp_id and images → convert to vector → store in SQL Server
@app.post("/api/upload_face")
async def upload_face(emp_id: int = Form(...), file: UploadFile = File(...), db: Session = Depends(get_db)):
    # images read
    image_bytes = await file.read()
    np_arr = np.frombuffer(image_bytes, np.uint8)
    
    # Convert image to OpenCV
    image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    if image is None:
        return {"error": "Invalid image file"}

    # Convert to face vector
    face_vector = extract_face_vector(image)  # extract_face_vector function from face_recognition.py
    if face_vector is None:
        return {"error": "No face detected"}

    # Convert vector to binary
    binary_vector = pickle.dumps(face_vector)

    # Find out if there are employees
    employee = db.query(Employee).filter(Employee.id == emp_id).first()
    if not employee:
        return {"error": "Employee not found"}

    # Check if FaceVector of this emp_id exists.
    face_record = db.query(FaceVector).filter(FaceVector.emp_id == emp_id).first()
    
    if face_record:
        # Update face vector 
        face_record.vector = binary_vector
    else:
        # Add new if not already available
        face_record = FaceVector(emp_id=emp_id, vector=binary_vector)
        db.add(face_record)

    # Commit data to database
    db.commit()

    # ✅ **Update Redis immediately after updating the database**
    face_vector_dict = face_record.to_dict()
    redis_key = f"face_vector:{emp_id}"
    redis_client.set(redis_key, pickle.dumps(face_vector_dict))
    print(f"Updated Redis: {redis_key}")

    return {"message": "Face vector saved successfully"}


