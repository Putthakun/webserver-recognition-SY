from sqlalchemy.orm import Session

# import module
from redis_client import redis_client
from datetime import datetime
from models import *
from database import *

# import lib
import numpy as np
import logging
import pytz
import numpy as np
import pickle
import cv2

def get_best_match(new_vector, redis_client, camera_id, db: Session, threshold=0.50, use_cosine=True):
    if new_vector is None or not isinstance(new_vector, np.ndarray):
        logging.error(f"❌ Unable to extract embedding from camera {camera_id}")
        return

    logging.error(f"🎯 threshold : {threshold}")
    new_vector = np.array(new_vector, dtype=np.float32)

    keys = redis_client.keys("face_vector:*")
    if not keys:
        logging.warning(" No face vectors found in Redis")
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
            logging.warning(f"❌ Vector sizes do not match: {stored_vector.shape} != {new_vector.shape}")
            continue

        if use_cosine:
            score = np.dot(new_vector, stored_vector) / (np.linalg.norm(new_vector) * np.linalg.norm(stored_vector))
            logging.info(f"Cosine Score for {emp_id}: {score}")
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

        # Check if there is a recent transaction in Redis.
        if redis_client.exists(cache_key):
            logging.info(f"⚠️ Transaction for emp_id={emp_id} already exists. Skipping...")
            return

        # Record transactions in the database
        save_transaction(db, emp_id, camera_id)

        # Note that this emp_id has the latest transaction with TTL set to 60 seconds.
        redis_client.set(cache_key, "1", ex=60)
        logging.info(f"✅ Transaction saved and cached: emp_id={emp_id}, camera_id={camera_id}")

    else:
        logging.info(f"❌ No matching face found for Camera {camera_id}")


def save_transaction(db: Session, emp_id: int, camera_id: int):

    # Time thai
    bangkok_tz = pytz.timezone('Asia/Bangkok')
    timestamp = datetime.now(bangkok_tz)  # Curren time in thai

    # Data for transaction
    new_transaction = Transaction(
        emp_id=emp_id,
        camera_id=camera_id,
        timestamp=timestamp
    )

    db.add(new_transaction)
    db.commit()
    db.refresh(new_transaction)
    logging.info(f"📝 Transaction saved: emp_id={emp_id}, camera_id={camera_id}, timestamp={timestamp}")