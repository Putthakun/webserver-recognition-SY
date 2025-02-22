
from sqlalchemy.orm import Session

# import module
from face_recognition import extract_face_embedding_rabbitmq
from redis_client import redis_client
from utils import get_best_match
from database import *

# import lib
import pika
import base64
import json
import zlib
import logging
import cv2
import numpy as np
import os

db_session = next(get_db())

# Set up log recording
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# Set up RabbitMQ connection
RABBITMQ_HOST = "SY_rabbitmq"  # Container name of RabbitMQ
RABBITMQ_USER = "S@ony_devide0102"
RABBITMQ_PASS = "S@ony_devide0102"
QUEUE_NAME = "face_images"

output_folder = "images" 
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

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
        
        logging.info("✅ RabbitMQ connection established successfully")
        
        return connection, channel
    except pika.exceptions.AMQPConnectionError as e:
        logging.error(f"❌ Connection failed: {e}")
        raise

# Images processing function
def process_image(camera_id, image):

    # images count
    image_count = len(os.listdir(output_folder)) + 1
    filename = os.path.join(output_folder, f"faces_image_{image_count}_{camera_id}.jpg")

    # Use cv2.imwrite() for recording
    success = cv2.imwrite(filename, image)
    if success:
        logging.info(f"✅ Record images from the camera {camera_id} successfully: {filename}")
    else:
        logging.error(f"❌ Can't record images from {camera_id}")

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
                process_image(camera_id, image)

                 # Get embedding result
                result = extract_face_embedding_rabbitmq(camera_id, image)
                
                if result:  # Check result not be None
                    embedding_array = np.array(result["embedding"])  # Convert to numpy array
                    camera_id = result["camera_id"]  # Return camera_id 
                    logging.info(f"🔢 First 10 values of embedding: {embedding_array[:10]}")

                    # Call get_best_match function from utils.py
                    get_best_match(embedding_array, redis_client, camera_id ,db=db_session, threshold=0.50, use_cosine=True)

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