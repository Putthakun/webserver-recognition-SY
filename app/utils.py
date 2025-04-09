import numpy as np
import time
import hashlib
import logging
import requests
from prometheus_client import Counter, Histogram
from redis_client import redis_client
from redis.commands.search.query import Query

logging.basicConfig(level=logging.DEBUG)

# ใช้ ID สำหรับ guess (ในที่นี้คือ "0" หรือจะเปลี่ยนเป็น UUID ก็ได้)
guessID = 0

# Prometheus Metrics
face_match_duration = Histogram("face_match_duration_seconds", "Time taken to match face vectors")
face_match_success_total = Counter("face_match_success_total", "Successful face matches")
face_guess_total = Counter("face_guess_total", "Total guess transactions")


# 🔗 .NET API Call
def post_transaction_api(emp_id: str, camera_id: str):
    try:
        url = "http://employeeapi:5000/api/Transactions"
        data = {"EmpID": emp_id, "CameraID": str(camera_id)}
        response = requests.post(url, data=data)
        if response.status_code == 200:
            logging.info(f"✅ POSTED to .NET API: emp_id={emp_id}, camera_id={camera_id}")
        else:
            logging.error(f"❌ Failed to post transaction. Status: {response.status_code}, Response: {response.text}")
    except Exception as e:
        logging.error(f"❌ Exception while posting to .NET API: {e}")


# 🎯 ตรวจสอบว่ามี match หรือไม่
@face_match_duration.time()
def get_best_match(new_vector, redis_client, camera_id, threshold=0.50):
    if new_vector is None or not isinstance(new_vector, np.ndarray):
        logging.error(f"❌ Invalid vector from camera {camera_id}")
        return

    new_vector = np.array(new_vector, dtype=np.float32)
    norm = np.linalg.norm(new_vector)
    if norm == 0:
        logging.warning("🚫 Cannot normalize — vector norm is 0")
        return
    new_vector /= norm

    binary_vector = new_vector.tobytes()

    try:
        query = Query("*=>[KNN 5 @vector $vec_param]") \
            .sort_by("__vector_score") \
            .paging(0, 5) \
            .dialect(2)

        result = redis_client.ft("face_vectors_idx").search(
            query, {"vec_param": binary_vector}
        )
    except Exception as e:
        logging.error(f"❌ Redis vector search error: {e}")
        return

    if result.total == 0 or not result.docs:
        logging.warning("😢 No match found via Redis HNSW")
        return handle_guess(new_vector, redis_client, camera_id)

    all_scores = []
    for i, doc in enumerate(result.docs):
        try:
            emp_id = doc.id.split(":")[1]
            score_raw = (
                getattr(doc, "score", None)
                or doc.__dict__.get("__vector_score")
                or doc.__dict__.get("_score")
            )

            if score_raw is None:
                raise ValueError("❌ No valid similarity score found.")

            similarity = 1.0 - float(score_raw)  # Redis returns cosine distance
            all_scores.append((emp_id, similarity))
            logging.info(f"   {i+1}. emp_id={emp_id}, similarity={similarity:.4f}")
        except Exception as e:
            logging.warning(f"⚠️ Failed to parse result {i+1}: {e}")

    if not all_scores:
        logging.warning("🚫 No valid matches found — fallback to guess")
        return handle_guess(new_vector, redis_client, camera_id)

    emp_id, similarity = all_scores[0]
    logging.info(f"🎯 Top match: emp_id={emp_id}, similarity={similarity:.4f} (threshold={threshold})")

    if similarity >= threshold:
        return handle_match(emp_id, redis_client, camera_id)
    else:
        return handle_guess(new_vector, redis_client, camera_id)


# ✅ จัดการเมื่อ match สำเร็จ
def handle_match(emp_id, redis_client, camera_id):
    face_match_success_total.inc()
    cache_key = f"recent_transaction:{emp_id}"
    if redis_client.exists(cache_key):
        logging.info(f"⚠️ Duplicate match ignored for emp_id={emp_id}")
        return
    post_transaction_api(emp_id, camera_id)
    redis_client.set(cache_key, "1", ex=60)

    # ลบ wait_match_key และตั้ง flag ว่า match เกิดขึ้น
    for key in redis_client.scan_iter(f"wait_match:{camera_id}:*"):
        vector_hash = key.decode().split(":")[-1]
        match_flag_key = f"match_confirmed:{camera_id}:{vector_hash}"
        redis_client.set(match_flag_key, "1", ex=5)  # valid ชั่วคราว
        redis_client.delete(key)
        logging.debug(f"🧹 Match confirmed and wait_match deleted: {key}")


def handle_guess(new_vector, redis_client, camera_id, delay=3):
    face_guess_total.inc()
    vector_hash = hashlib.md5(new_vector.tobytes()).hexdigest()
    guess_key = f"recent_guess:{camera_id}:{vector_hash}"
    wait_key = f"wait_match:{camera_id}:{vector_hash}"
    match_flag_key = f"match_confirmed:{camera_id}:{vector_hash}"

    if redis_client.exists(wait_key):
        logging.info(f"⏳ Cooldown already in place for camera {camera_id}, skipping guess")
        return

    if redis_client.setnx(wait_key, "1"):
        redis_client.expire(wait_key, delay)
        logging.info(f"⏱️ Delaying guess creation for {delay}s to wait for possible match...")
        time.sleep(delay)

        # 🧪 เช็คว่า match เกิดขึ้นไหม (แทนการพึ่ง TTL)
        if redis_client.exists(match_flag_key):
            logging.info(f"✅ Match confirmed during delay — skipping guess")
        else:
            if not redis_client.exists(guess_key):
                post_transaction_api(str(guessID), camera_id)
                redis_client.set(guess_key, "1", ex=60)
                logging.info(f"✅ Guess transaction created for camera {camera_id}")
            else:
                logging.info(f"⏳ Duplicate guess ignored for camera {camera_id}")
    else:
        logging.info(f"🧪 Match already pending — skipping guess creation")

