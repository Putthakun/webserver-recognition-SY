from database import *
from sqlalchemy.orm import Session
from models import *
import numpy as np
import pickle
import cv2
from redis_client import redis_client
import logging
from database import *
from redis_client import redis_client
import numpy as np
import logging
import pickleutcnow
from sqlalchemy.orm import Session
from datetime import datetime
import pytz

def get_best_match(new_vector, redis_client, camera_id, db: Session, threshold=0.60, use_cosine=True):
    if new_vector is None or not isinstance(new_vector, np.ndarray):
        logging.error(f"❌ ไม่สามารถดึง embedding จากกล้อง {camera_id}")
        return

    logging.info(f"🎯 new_vector shape: {new_vector.shape}, dtype: {new_vector.dtype}")

    new_vector = np.array(new_vector, dtype=np.float32)
    logging.info(f"🎯 Converted new_vector shape: {new_vector.shape}, dtype={new_vector.dtype}")

    keys = redis_client.keys("face_vector:*")
    if not keys:
        logging.warning("⚠️ No face vectors found in Redis")
        return

    best_match = None
    best_score = float("-inf") if use_cosine else float("inf")

    for key in keys:
        data = redis_client.get(key)
        if not data:
            continue

        face_data = pickle.loads(data)
        stored_vector = np.array(face_data.get("vector"), dtype=np.float32)
        emp_id = face_data.get("emp_id")

        if stored_vector.shape != new_vector.shape:
            logging.warning(f"⚠️ ขนาดเวกเตอร์ไม่ตรงกัน: {stored_vector.shape} != {new_vector.shape}")
            continue

        if use_cosine:
            score = np.dot(new_vector, stored_vector) / (np.linalg.norm(new_vector) * np.linalg.norm(stored_vector))
            if score >= threshold and score > best_score:
                best_score = score
                best_match = {"emp_id": emp_id, "similarity": best_score}
        else:
            score = np.linalg.norm(new_vector - stored_vector)
            if score < best_score:
                best_score = score
                best_match = {"emp_id": emp_id, "distance": best_score}

    if best_match:
        emp_id = best_match["emp_id"]
        cache_key = f"recent_transaction:{emp_id}"

        # ✅ เช็กว่ามี Transaction ล่าสุดอยู่ใน Redis หรือไม่
        if redis_client.exists(cache_key):
            logging.info(f"⚠️ Transaction for emp_id={emp_id} already exists. Skipping...")
            return

        # ✅ บันทึก Transaction ลงใน Database
        save_transaction(db, emp_id, camera_id)

        # ✅ บันทึกว่า emp_id นี้มี Transaction ล่าสุด พร้อมตั้งค่า TTL 60 วินาที
        redis_client.set(cache_key, "1", ex=60)
        logging.info(f"✅ Transaction saved and cached: emp_id={emp_id}, camera_id={camera_id}")

    else:
        logging.info(f"❌ No matching face found for Camera {camera_id}")


def save_transaction(db: Session, emp_id: int, camera_id: int):
    """
    บันทึก transaction ลงในฐานข้อมูลเมื่อมีการจับคู่สำเร็จ โดยใช้เวลาประเทศไทย (UTC+7)
    """

    bangkok_tz = pytz.timezone('Asia/Bangkok')
    timestamp = datetime.now(bangkok_tz)  # ✅ เวลาปัจจุบันของไทย

    new_transaction = Transaction(
        emp_id=emp_id,
        camera_id=camera_id,
        timestamp=timestamp
    )

    db.add(new_transaction)
    db.commit()
    db.refresh(new_transaction)
    logging.info(f"📝 Transaction saved: emp_id={emp_id}, camera_id={camera_id}, timestamp={timestamp}")