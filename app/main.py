from fastapi import FastAPI
from database import *
from fastapi import FastAPI, UploadFile, File, Form, Depends
from sqlalchemy.orm import Session
from models import *
from face_recognition import extract_face_vector
import numpy as np
import pickle
import cv2
from redis_client import redis_client
import asyncio
import threading
import logging
from rabbitmq import start_consumer



app = FastAPI()

@app.get("/")
def read_root():
    return {"Hello": "World"}

# ฟังก์ชันสำหรับรัน Consumer ใน background thread
def start_consumer_thread():
    def run_consumer():
        start_consumer()

    # รัน start_consumer() ใน background thread
    consumer_thread = threading.Thread(target=run_consumer)
    consumer_thread.daemon = True  # ให้มันทำงานใน background
    consumer_thread.start()

@app.on_event("startup")
async def startup():
    logging.info("🚀 FastAPI กำลังเริ่มต้น...")
    start_consumer_thread()

@app.on_event("startup")
async def startup():
    # เชื่อมต่อฐานข้อมูลผ่าน session
    db: Session = next(get_db())

    try:
        # ดึงข้อมูลทั้งหมดจากตาราง FaceVector
        face_vectors = db.query(FaceVector).all()

        for face_vector in face_vectors:
            # แปลงข้อมูลเป็น dictionary
            face_vector_dict = face_vector.to_dict()

            # สร้าง Redis Key สำหรับ emp_id
            redis_key = f"face_vector:{face_vector.emp_id}"

            # เก็บข้อมูลใน Redis (ใช้ Pickle เพื่อเก็บข้อมูลแบบไบต์)
            redis_client.set(redis_key, pickle.dumps(face_vector_dict))

    finally:
        db.close()

@app.get("/api/vector-redis")
def get_all_face_vectors():
    # ดึงคีย์ทั้งหมดที่เกี่ยวข้องกับ face_vector
    keys = redis_client.keys("face_vector:*")

    if not keys:
        raise HTTPException(status_code=404, detail="No face vectors found in Redis")

    face_vectors = []

    for key in keys:
        # ดึงข้อมูลจาก Redis (ไม่ต้อง decode เป็น UTF-8)
        data = redis_client.get(key)

        if data:
            # แปลงข้อมูลจาก binary กลับเป็น dictionary
            face_vector_dict = pickle.loads(data)
            face_vectors.append(face_vector_dict)

    return {"face_vectors": face_vectors}
    
# API รับ emp_id และรูปภาพ → แปลงเป็นเวกเตอร์ → เก็บลง SQL Server
@app.post("/api/upload_face")
async def upload_face(emp_id: int = Form(...), file: UploadFile = File(...), db: Session = Depends(get_db)):
    # อ่านไฟล์ภาพจากอัปโหลด
    image_bytes = await file.read()
    np_arr = np.frombuffer(image_bytes, np.uint8)
    
    # แปลงเป็นภาพ OpenCV
    image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    if image is None:
        return {"error": "Invalid image file"}

    # แปลงเป็นเวกเตอร์ใบหน้า
    face_vector = extract_face_vector(image)
    if face_vector is None:
        return {"error": "No face detected"}

    # แปลงเวกเตอร์เป็น binary
    binary_vector = pickle.dumps(face_vector)

    # ค้นหาว่ามีพนักงานอยู่หรือไม่
    employee = db.query(Employee).filter(Employee.id == emp_id).first()
    if not employee:
        return {"error": "Employee not found"}

    # ตรวจสอบว่ามี FaceVector ของ emp_id นี้อยู่หรือไม่
    face_record = db.query(FaceVector).filter(FaceVector.emp_id == emp_id).first()
    
    if face_record:
        # อัปเดตเวกเตอร์ใบหน้า
        face_record.vector = binary_vector
    else:
        # เพิ่มใหม่ถ้ายังไม่มี
        face_record = FaceVector(emp_id=emp_id, vector=binary_vector)
        db.add(face_record)

    # Commit ข้อมูลลงฐานข้อมูล
    db.commit()

    # ✅ **อัปเดต Redis ทันทีหลังจากอัปเดตฐานข้อมูล**
    face_vector_dict = face_record.to_dict()
    redis_key = f"face_vector:{emp_id}"
    redis_client.set(redis_key, pickle.dumps(face_vector_dict))
    print(f"Updated Redis: {redis_key}")

    return {"message": "Face vector saved successfully"}


