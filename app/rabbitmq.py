import os
import time
import json
import base64
import zlib
import hashlib
import logging
import numpy as np
import cv2
import pika
import requests

from prometheus_client import Gauge, Counter
from app.face_recognition import extract_face_embedding_rabbitmq
from app.redis_client import redis_client
from app.utils import get_best_match


# ----------------------- Logging Setup -----------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)


# ----------------------- RabbitMQ Config -----------------------
RABBITMQ_HOST = "SY_rabbitmq"
RABBITMQ_USER = "S@ony_devide0102"
RABBITMQ_PASS = "S@ony_devide0102"
QUEUE_NAME = "face_images"

# Prometheus Metrics
rabbitmq_connection_status = Gauge("rabbitmq_connection_status", "RabbitMQ Connection Status (1=Connected, 0=Disconnected)")
rabbitmq_queue_image_count = Gauge("rabbitmq_queue_image_count", "Number of Images in RabbitMQ Queue")
images_consumed_total = Counter("images_consumed_total", "Total images processed from RabbitMQ")

# ----------------------- Utility Functions -----------------------
def is_image_bytes(b: bytes) -> bool:
    return (b[:3] == b"\xFF\xD8\xFF") or (b[:8] == b"\x89PNG\r\n\x1a\n")


def adjust_brightness_clahe(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)
    return cv2.cvtColor(enhanced, cv2.COLOR_GRAY2BGR)


# ----------------------- Main Callback -----------------------
def callback(ch, method, properties, body):
    try:
        msg = json.loads(body.decode("utf-8"))
        camera_id = msg.get("camera_id")
        image_base64 = msg.get("image")

        if not camera_id or not image_base64:
            ch.basic_ack(delivery_tag=method.delivery_tag)
            return

        raw = base64.b64decode(image_base64)
        jpg = zlib.decompress(raw)
        if not is_image_bytes(jpg):
            ch.basic_ack(delivery_tag=method.delivery_tag)
            return

        arr = np.frombuffer(jpg, dtype=np.uint8)
        image = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if image is None:
            ch.basic_ack(delivery_tag=method.delivery_tag)
            return

        image = adjust_brightness_clahe(image)
        result = extract_face_embedding_rabbitmq(camera_id, image)
        if not result:
            logging.info(f"⚠️ No face found for {camera_id}")
            ch.basic_ack(delivery_tag=method.delivery_tag)
            return

        embedding = np.array(result["embedding"], dtype=np.float32)
        norm = np.linalg.norm(embedding)
        if norm == 0:
            ch.basic_ack(delivery_tag=method.delivery_tag)
            return
        embedding /= norm

        face_hash = hashlib.sha1(embedding.tobytes()).hexdigest()[:16]
        global_hash_key = f"global_facehash:{face_hash}"

        if redis_client.exists(global_hash_key):
            logging.info(f"🧱 Duplicate facehash {face_hash} — skip")
            ch.basic_ack(delivery_tag=method.delivery_tag)
            return
        redis_client.setex(global_hash_key, 20, "1")

        active_key = f"guess_active:{camera_id}"
        if redis_client.exists(active_key):
            logging.info(f"🚫 Guess already active for {camera_id}, skip")
            ch.basic_ack(delivery_tag=method.delivery_tag)
            return

        if not redis_client.set(active_key, "1", nx=True, ex=25):
            logging.info(f"🔒 Another server is handling {camera_id}")
            ch.basic_ack(delivery_tag=method.delivery_tag)
            return

        get_best_match(embedding, redis_client, camera_id, threshold=0.40)

        logging.info(f"✅ Processed camera {camera_id} [hash={face_hash}]")
        time.sleep(0.5)
        redis_client.delete(active_key)
        ch.basic_ack(delivery_tag=method.delivery_tag)

    except Exception as e:
        logging.exception(f"❌ Error in callback: {e}")
        ch.basic_nack(delivery_tag=method.delivery_tag, requeue=False)


# ----------------------- Consumer Setup -----------------------
def start_consumer():
    credentials = pika.PlainCredentials(RABBITMQ_USER, RABBITMQ_PASS)
    params = pika.ConnectionParameters(
        host=RABBITMQ_HOST,
        credentials=credentials,
        heartbeat=60,
        blocked_connection_timeout=60
    )

    connection = pika.BlockingConnection(params)
    channel = connection.channel()
    channel.queue_declare(queue=QUEUE_NAME, durable=True)
    channel.basic_qos(prefetch_count=1)

    channel.basic_consume(
        queue=QUEUE_NAME,
        on_message_callback=callback,
        auto_ack=False
    )

    logging.info("🎧 Listening for messages (Work Queue Mode, durable queue, prefetch=1)...")
    rabbitmq_connection_status.set(1)

    try:
        channel.start_consuming()
    except KeyboardInterrupt:
        logging.warning("🛑 Consumer stopped manually.")
        rabbitmq_connection_status.set(0)
        connection.close()


# ----------------------- Main Entry -----------------------
if __name__ == "__main__":
    start_consumer()
