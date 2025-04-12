from fastapi import FastAPI, UploadFile, File, Form, Depends, WebSocket, Response
from fastapi.responses import StreamingResponse
# from prometheus_client import Gauge, generate_latest, CONTENT_TYPE_LATEST
from fastapi.middleware.cors import CORSMiddleware
from prometheus_client import generate_latest, CONTENT_TYPE_LATEST
from prometheus_client import start_http_server

# import module
from rabbitmq import start_consumer, update_queue_metrics
from face_recognition import extract_face_vector
import rabbitmq

# import lib
from typing import List
import numpy as np
import pickle
import cv2
import asyncio
import threading
import logging
import json
import math



app = FastAPI()

logging.basicConfig(level=logging.DEBUG)

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
    logging.info("🚀 FastAPI starts..-------------------------------------------------------------------.")
 
    start_consumer_thread()
    # Connect to the database via session


# nomalization
def adjust_brightness_clahe(image):
    # แปลงเป็น Grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    enhanced = clahe.apply(gray)

    return cv2.cvtColor(enhanced, cv2.COLOR_GRAY2BGR)


@app.get("/metrics")
def metrics():
    return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)

# embeded for .net
@app.post("/api/extract_vector")
async def extract_vector(files: List[UploadFile] = File(...)):
    results = []

    for file in files:
        try:
            image_bytes = await file.read()
            np_arr = np.frombuffer(image_bytes, np.uint8)
            image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

            if image is None:
                results.append({"filename": file.filename, "error": "Invalid image file"})
                continue

            image = adjust_brightness_clahe(image)
            face_vector = extract_face_vector(image)

            if face_vector is None:
                results.append({"filename": file.filename, "error": "No face detected"})
                continue

            results.append({
                "filename": file.filename,
                "vector": face_vector.tolist(),
                "vector_size": len(face_vector),
                "message": "Success"
            })

        except Exception as e:
            results.append({"filename": file.filename, "error": str(e)})

    return results


@app.post("/api/check_face_detected")
async def check_face_detected(file: UploadFile = File(...)):
    try:
        image_bytes = await file.read()
        np_arr = np.frombuffer(image_bytes, np.uint8)
        image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

        if image is None:
            return {"success": False, "message": "Invalid image format"}

        image = adjust_brightness_clahe(image)
        face_vector = extract_face_vector(image)

        if face_vector is None:
            return {
                "success": True,
                "detected": False,
                "message": "No face detected"
            }

        return {
            "success": True,
            "detected": True,
            "vector_size": len(face_vector),
            "message": "Face detected"
        }

    except Exception as e:
        logging.error(f"❌ Error in check_face_detected: {e}")
        return {"success": False, "message": str(e)}
