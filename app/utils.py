import os
import numpy as np
import time
import hashlib
import logging
import requests
from prometheus_client import Counter, Histogram
from app.redis_client import redis_client
from redis.commands.search.query import Query

# ---------- Logging ----------
logging.basicConfig(level=logging.DEBUG, format="%(asctime)s - %(levelname)s - %(message)s")

# ---------- Metrics ----------
face_match_duration = Histogram("face_match_duration_seconds", "Time taken to match face vectors")
face_match_success_total = Counter("face_match_success_total", "Successful face matches")
face_guess_total = Counter("face_guess_total", "Total guess transactions")

# ---------- .NET API ----------
def post_transaction_api(emp_id: str, camera_id: str):
    try:
        url = "http://employeeapi:5000/api/Transactions"
        data = {"EmpID": emp_id, "CameraID": str(camera_id)}
        response = requests.post(url, data=data)
        if response.status_code == 200:
            logging.info(f"✅ POSTED to .NET API: emp_id={emp_id}, camera_id={camera_id}")
        else:
            logging.error(f"❌ Failed to post transaction: {response.status_code} - {response.text}")
    except Exception as e:
        logging.error(f"❌ Exception posting transaction: {e}")


# ---------- Face Matching (Hybrid Delay + Confidence Check) ----------
@face_match_duration.time()
def get_best_match(new_vector, redis_client, camera_id, threshold=0.40):
    if new_vector is None or not isinstance(new_vector, np.ndarray):
        logging.error(f"❌ Invalid vector from {camera_id}")
        return

    new_vector = np.array(new_vector, dtype=np.float32)
    norm = np.linalg.norm(new_vector)
    if norm == 0:
        return
    new_vector /= norm

    try:
        query = Query("*=>[KNN 5 @vector $vec]").sort_by("__vector_score").paging(0, 5).dialect(2)
        result = redis_client.ft("face_vectors_idx").search(query, {"vec": new_vector.tobytes()})
    except Exception as e:
        logging.error(f"❌ Redis vector search error: {e}")
        return

    if not result.docs:
        logging.warning(f"No match found via Redis HNSW ({camera_id})")
        return handle_guess(new_vector, redis_client, camera_id)

    scores = []
    for doc in result.docs:
        emp_id = doc.id.split(":")[1]
        score = getattr(doc, "__vector_score", None)
        if score is not None:
            sim = 1.0 - float(score)
            scores.append((emp_id, sim))

    if not scores:
        return handle_guess(new_vector, redis_client, camera_id)

    emp_id, sim = scores[0]
    if sim >= threshold:
        redis_client.setex(f"recent_match:{camera_id}", 3, emp_id)
        return handle_match(emp_id, redis_client, camera_id)

    elif sim >= (threshold - 0.05):
        logging.info(f"🟡 Potential match ({sim:.3f}) near threshold, holding guess for {camera_id}")
        wait_time = 1.5
        interval = 0.3
        elapsed = 0
        matched = False

        while elapsed < wait_time:
            if redis_client.exists(f"recent_match:{camera_id}"):
                matched = True
                logging.info(f"🟢 Match appeared during hold ({camera_id}), cancel guess")
                break
            time.sleep(interval)
            elapsed += interval

        if not matched:
            return handle_guess(new_vector, redis_client, camera_id)
        return

    else:
        return handle_guess(new_vector, redis_client, camera_id)


# ---------- Match ----------
def handle_match(emp_id, redis_client, camera_id):
    face_match_success_total.inc()
    key = f"recent_transaction:{emp_id}"
    if redis_client.exists(key):
        logging.info(f"⚠️ Duplicate match ignored for emp_id={emp_id}")
        return
    post_transaction_api(emp_id, camera_id)
    redis_client.set(key, "1", ex=60)
    logging.info(f"✅ Match transaction created for emp_id={emp_id}")


# ---------- Guess (Anti-Duplicate + Similarity Lock) ----------
def handle_guess(new_vector, redis_client, camera_id, delay=0.3):
    face_guess_total.inc()
    logging.debug(f"[GUESS] ⏳ start for {camera_id}")

    # --- Normalize vector ---
    new_vector = np.array(new_vector, dtype=np.float32)
    norm = np.linalg.norm(new_vector)
    if norm == 0:
        return
    new_vector /= norm

    recent_key = f"recent_guess:{camera_id}"
    if redis_client.exists(recent_key):
        logging.info(f"[GUESS] 🕓 Cooldown active for {camera_id}, skip")
        return
    redis_client.setex(recent_key, 60, "1")  # กันซ้ำแค่ 10 วิ

    try:
        logging.debug(f"[GUESS] 🚀 Sending guess for {camera_id}")
        time.sleep(delay)  # หน่วงนิดหน่อยป้องกัน burst
        post_transaction_api("0", camera_id)
        logging.info(f"[GUESS] ✅ Guess created for {camera_id}")
    except Exception as e:
        logging.error(f"[GUESS] ❌ Error {e}")
