import numpy as np
import time
import logging
import pickle
import requests
import pytz
import json

from datetime import datetime
from app.redis_client import redis_client


# Guess ID
guessID = 0

# -------------------------------
# 🔁 Call .NET API to record a transaction
# -------------------------------
def post_transaction_api(emp_id: str, camera_id: str):
    try:
        url = "http://employeeapi:5000/api/Transactions"
        data = {
            "EmpID": emp_id,
            "CameraID": str(camera_id)
        }
        response = requests.post(url, data=data)
        if response.status_code == 200:
            logging.info(f"✅ POSTED to .NET API: emp_id={emp_id}, camera_id={camera_id}")
        else:
            logging.error(f"❌ Failed to post transaction. Status: {response.status_code}, Response: {response.text}")
    except Exception as e:
        logging.error(f"❌ Exception while posting to .NET API: {e}")

# -------------------------------
# 🔍 Match face vector with Redis and handle guess
# -------------------------------
def get_best_match(new_vector, redis_client, camera_id, threshold=0.50, use_cosine=True):
    start_time = time.time()

    if new_vector is None or not isinstance(new_vector, np.ndarray):
        logging.error(f"❌ No vector found or invalid type from camera {camera_id}")
        return

    logging.info(f"🎯 Matching for camera {camera_id} | threshold={threshold} | cosine={use_cosine}")
    new_vector = np.array(new_vector, dtype=np.float32)

    keys = redis_client.keys("face_vector:*")
    logging.debug(f"🔍 Found {len(keys)} vectors in Redis")

    best_match = None
    best_score = float("-inf") if use_cosine else float("inf")

    for key in keys:
        data = redis_client.get(key)
        if not data:
            continue

        try:
            vector = json.loads(data.decode())
            emp_id = key.decode().split(":")[1]
            stored_vector = np.array(vector, dtype=np.float32)

            if stored_vector.shape != new_vector.shape:
                continue

            score = (np.dot(new_vector, stored_vector) / 
                     (np.linalg.norm(new_vector) * np.linalg.norm(stored_vector))) if use_cosine \
                    else np.linalg.norm(new_vector - stored_vector)

            if use_cosine:
                if score >= threshold and score > best_score:
                    best_score = score
                    best_match = {"emp_id": emp_id, "similarity": score}
            else:
                if score < best_score:
                    best_score = score
                    best_match = {"emp_id": emp_id, "distance": score}

        except Exception as e:
            logging.error(f"❌ Error with key {key}: {e}")

    if best_match:
        emp_id = best_match["emp_id"]
        cache_key = f"recent_transaction:{emp_id}"
        if redis_client.exists(cache_key):
            logging.info(f"⚠️ Transaction exists for emp_id={emp_id}. Skipping...")
            return

        post_transaction_api(emp_id, camera_id)
        redis_client.set(cache_key, "1", ex=60)
        logging.info(f"✅ Transaction saved for emp_id={emp_id}, camera_id={camera_id}")

        # Clear guess cache for this camera if exists
        guess_key = f"recent_guess:{camera_id}"
        redis_client.delete(guess_key)
    else:
        logging.info(f"❌ No match found for camera {camera_id}")
        # Create a guess only if not already done recently
        guess_key = f"recent_guess:{camera_id}"
        if not redis_client.exists(guess_key):
            post_transaction_api(guessID, camera_id)
            redis_client.set(guess_key, "1", ex=60)  # Avoid duplicate guess
            logging.info(f"✅ Guess transaction created for camera {camera_id}")

    logging.info(f"⏱️ Matching process took {time.time() - start_time:.4f}s")
