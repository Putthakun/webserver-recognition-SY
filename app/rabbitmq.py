from sqlalchemy.orm import Session
from prometheus_client import Gauge, start_http_server

# import module
from face_recognition import extract_face_embedding_rabbitmq
from redis_client import redis_client
from utils import get_best_match

# import lib
import pika
import base64
import json
import zlib
import logging
import cv2
import numpy as np
import os

# Set up log recording
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# Set up RabbitMQ connection
RABBITMQ_HOST = "SY_rabbitmq"  # Container name of RabbitMQ
RABBITMQ_USER = "S@ony_devide0102"
RABBITMQ_PASS = "S@ony_devide0102"
QUEUE_NAME = "face_images"

# Prometheus metrics
rabbitmq_connection_status = Gauge("rabbitmq_connection_status", "RabbitMQ Connection Status (1=Connected, 0=Disconnected)")
rabbitmq_queue_image_count = Gauge("rabbitmq_queue_image_count", "Number of Images in RabbitMQ Queue")

def adjust_brightness_clahe(image):
    # แปลงเป็น Grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # ใช้ CLAHE เพื่อปรับ contrast
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    enhanced = clahe.apply(gray)

    # แปลงกลับเป็นภาพสี
    return cv2.cvtColor(enhanced, cv2.COLOR_GRAY2BGR)

def update_queue_metrics(channel):
    try:
        queue = channel.queue_declare(queue=QUEUE_NAME, passive=True)
        count = queue.method.message_count
        rabbitmq_queue_image_count.set(count)
        logging.info(f"rabbitmq_queue_image_count={count}%")
    except Exception as e:
        logging.error(f"❌ Failed to update queue metrics: {e}")
        rabbitmq_queue_image_count.set(0)

# RabbitMQ connection function
def get_rabbitmq_connection():
    try:
        credentials = pika.PlainCredentials(RABBITMQ_USER, RABBITMQ_PASS)
        parameters = pika.ConnectionParameters(
            host=RABBITMQ_HOST,
            credentials=credentials,
            heartbeat=3000  # timeout protaction
        )
        connection = pika.BlockingConnection(parameters)
        channel = connection.channel()
        
        rabbitmq_connection_status.set(1)
        logging.info("✅ RabbitMQ connection established successfully")
        
        channel.queue_declare(queue=QUEUE_NAME, durable=True)
        
        return connection, channel
    except pika.exceptions.AMQPConnectionError as e:
        logging.error(f"❌ Connection failed: {e}")
        raise


def callback(ch, method, properties, body):
    try:
        message = json.loads(body)
        camera_id = message.get("camera_id")
        image_base64 = message.get("image")

        if camera_id and image_base64:
            # Decode Base64
            image_bytes = base64.b64decode(image_base64)

            # Decompress Data
            decompressed_data = zlib.decompress(image_bytes)

            # Convert to NumPy array
            image_array = np.frombuffer(decompressed_data, dtype=np.uint8)

            # Decode image
            image = cv2.imdecode(image_array, cv2.IMREAD_UNCHANGED)
            if image is not None:
                update_queue_metrics(ch)
                image = adjust_brightness_clahe(image)
                 # Get embedding result
                result = extract_face_embedding_rabbitmq(camera_id, image)
                
                if result:  # Check result not be None
                    embedding_array = np.array(result["embedding"])  # Convert to numpy array
                    camera_id = result["camera_id"]  # Return camera_id 

                    # Call get_best_match function from utils.py
                    get_best_match(embedding_array, redis_client, camera_id, threshold=0.50, use_cosine=True)

                else:
                    logging.error(f"❌ Unable to extract embedding from camera {camera_id}")


                ch.basic_ack(delivery_tag=method.delivery_tag)
            else:
                logging.error("❌ cv2.imdecode() returns None, indicating that the image cannot be converted back.")
                logging.warning(f"❌ Can't convert image from {camera_id}")

        else:
            logging.warning(f"❌ Can't convert image from {camera_id} ได้")

    except Exception as e:
        logging.error(f"❌ Error processing message: {e}")


# Consumer start function
def start_consumer():
    connection, channel = get_rabbitmq_connection()
    channel.basic_qos(prefetch_count=1)
    channel.basic_consume(queue=QUEUE_NAME, on_message_callback=callback)
    
    logging.info("🎧 Consumer is waiting for images from RabbitMQ....")
    channel.start_consuming()