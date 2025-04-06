from fastapi import FastAPI, UploadFile, File, Form, Depends, WebSocket, Response
from fastapi.responses import StreamingResponse
# from prometheus_client import Gauge, generate_latest, CONTENT_TYPE_LATEST
from fastapi.middleware.cors import CORSMiddleware
from prometheus_client import generate_latest, CONTENT_TYPE_LATEST
from prometheus_client import start_http_server

# import module
from app.rabbitmq import start_consumer, update_queue_metrics
from app.redis_client import redis_client
from app.face_recognition import extract_face_vector
import app.rabbitmq

# import lib
import numpy as np
import pickle
import cv2
import asyncio
import threading
import logging
import json
import math



app = FastAPI()

# อนุญาตให้ทุก Origin เชื่อมต่อ (ควรกำหนดเฉพาะ Origin ที่ใช้งานจริง)
origins = ["*"]  # หรือใช้ ["http://localhost:5173"] ถ้าเป็น Vue.js

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

start_http_server(9100)

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
    # Connect to the database via session

# nomalization
def adjust_brightness_clahe(image):
    # แปลงเป็น Grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # ใช้ CLAHE เพื่อปรับ contrast
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    enhanced = clahe.apply(gray)

    # แปลงกลับเป็นภาพสี
    return cv2.cvtColor(enhanced, cv2.COLOR_GRAY2BGR)

# api check data in redis
def sanitize_vector(vector):
    return [v if isinstance(v, float) and math.isfinite(v) else 0.0 for v in vector]

@app.get("/metrics")
def metrics():
    return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)
    
@app.get("/api/redis/all-vectors")
def get_all_face_vectors():
    keys = redis_client.keys("face_vector:*")
    result = {}

    for key in keys:
        print(f"Reading Redis key: {key}")
        data = redis_client.get(key)
        if data:
            try:
                # ถ้าใช้ decode_responses=True → ไม่ต้อง decode
                vector = json.loads(data)
                result[key] = sanitize_vector(vector)
            except Exception as e:
                result[key] = f"decode error: {str(e)}"

    return {
        "total": len(result),
        "vectors": result
    }

# embeded for .net
@app.post("/api/extract_vector")
async def extract_vector(file: UploadFile = File(...)):
    try:
        image_bytes = await file.read()
        np_arr = np.frombuffer(image_bytes, np.uint8)
        image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

        if image is None:
            return JSONResponse(status_code=400, content={"error": "Invalid image file"})

        image = adjust_brightness_clahe(image)
        face_vector = extract_face_vector(image)
        if face_vector is None:
            return {"error": "No face detected"}

        if face_vector is None:
            return JSONResponse(status_code=404, content={"error": "No face detected"})

        return {
            "vector": face_vector.tolist(),
            "vector_size": len(face_vector),
            "message": "Face vector extracted successfully"
        }

    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})

