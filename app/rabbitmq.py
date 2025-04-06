from prometheus_client import Gauge, Counter, start_http_server

# import module
from app.face_recognition import extract_face_embedding_rabbitmq
from app.redis_client import redis_client
from app.utils import get_best_match

# import lib
import pika
import base64
import json
import zlib
import logging
import cv2
import numpy as np

# Logging setup
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# RabbitMQ config
RABBITMQ_HOST = "SY_rabbitmq"
RABBITMQ_USER = "S@ony_devide0102"
RABBITMQ_PASS = "S@ony_devide0102"
QUEUE_NAME = "face_images"

# Prometheus Metrics
rabbitmq_connection_status = Gauge("rabbitmq_connection_status", "RabbitMQ Connection Status (1=Connected, 0=Disconnected)")
rabbitmq_queue_image_count = Gauge("rabbitmq_queue_image_count", "Number of Images in RabbitMQ Queue")
images_produced_total = Counter("images_produced_total", "Total images sent to RabbitMQ")
images_consumed_total = Counter("images_consumed_total", "Total images processed from RabbitMQ")
images_in_queue_estimate = Gauge("images_in_queue_estimate", "Estimated images in queue (produced - consumed)")

# Helper: CLAHE enhance
def adjust_brightness_clahe(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)
    return cv2.cvtColor(enhanced, cv2.COLOR_GRAY2BGR)

# Helper: update RabbitMQ message count directly (optional)
def update_queue_metrics(channel):
    try:
        queue = channel.queue_declare(queue=QUEUE_NAME, passive=True)
        count = queue.method.message_count
        rabbitmq_queue_image_count.set(count)
        logging.info(f"📦 rabbitmq_queue_image_count = {count}")
    except Exception as e:
        logging.error(f"❌ Failed to update queue metrics: {e}")
        rabbitmq_queue_image_count.set(0)

# RabbitMQ connection
def get_rabbitmq_connection():
    try:
        credentials = pika.PlainCredentials(RABBITMQ_USER, RABBITMQ_PASS)
        parameters = pika.ConnectionParameters(
            host=RABBITMQ_HOST,
            credentials=credentials,
            heartbeat=3000
        )
        connection = pika.BlockingConnection(parameters)
        channel = connection.channel()
        channel.queue_declare(queue=QUEUE_NAME, durable=True)
        rabbitmq_connection_status.set(1)
        logging.info("✅ RabbitMQ connected")
        return connection, channel
    except pika.exceptions.AMQPConnectionError as e:
        logging.error(f"❌ RabbitMQ connection failed: {e}")
        rabbitmq_connection_status.set(0)
        raise

# Message handler
def callback(ch, method, properties, body):
    try:
        message = json.loads(body)
        camera_id = message.get("camera_id")
        image_base64 = message.get("image")

        if camera_id and image_base64:
            image_bytes = base64.b64decode(image_base64)
            decompressed_data = zlib.decompress(image_bytes)
            image_array = np.frombuffer(decompressed_data, dtype=np.uint8)
            image = cv2.imdecode(image_array, cv2.IMREAD_UNCHANGED)

            if image is not None:
                # ✅ นับว่ารับเข้ามาแล้ว
                images_consumed_total.inc()

                # ✅ อัปเดตคิวและ estimate
                update_queue_metrics(ch)
                images_in_queue_estimate.set(
                    images_produced_total._value.get() - images_consumed_total._value.get()
                )

                image = adjust_brightness_clahe(image)
                result = extract_face_embedding_rabbitmq(camera_id, image)

                if result:
                    embedding_array = np.array(result["embedding"])
                    get_best_match(embedding_array, redis_client, camera_id, threshold=0.50, use_cosine=True)
                else:
                    logging.warning(f"❌ No embedding found for {camera_id}")

                ch.basic_ack(delivery_tag=method.delivery_tag)
            else:
                logging.warning(f"❌ cv2 decode failed for camera {camera_id}")
        else:
            logging.warning("❌ Invalid message: no camera_id or image")

    except Exception as e:
        logging.error(f"❌ Error in callback: {e}")

# Consumer runner
def start_consumer():
    connection, channel = get_rabbitmq_connection()
    channel.basic_qos(prefetch_count=1)
    channel.basic_consume(queue=QUEUE_NAME, on_message_callback=callback)
    logging.info("🎧 Listening for messages...")
    channel.start_consuming()

# Entry point
if __name__ == "__main__":
    start_http_server(9100)
    logging.info("📊 Prometheus metrics exposed at :9100/metrics")
    start_consumer()
