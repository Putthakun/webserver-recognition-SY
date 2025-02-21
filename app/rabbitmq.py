import pika
import base64
import json
import zlib
import logging
import cv2
import numpy as np
import os
from face_recognition import extract_face_embedding_rabbitmq
from redis_client import redis_client
from utils import get_best_match
from sqlalchemy.orm import Session
from database import *

db_session = next(get_db())

# ตั้งค่าการบันทึก log
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# ตั้งค่าการเชื่อมต่อ RabbitMQ
RABBITMQ_HOST = "SY_rabbitmq"  # ชื่อ Container ของ RabbitMQ
RABBITMQ_USER = "S@ony_devide0102"
RABBITMQ_PASS = "S@ony_devide0102"
QUEUE_NAME = "face_images"

output_folder = "images"  # ตำแหน่งที่ต้องการให้โฟลเดอร์ images อยู่ใน Container
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# ฟังก์ชันเชื่อมต่อกับ RabbitMQ
def get_rabbitmq_connection():
    try:
        credentials = pika.PlainCredentials(RABBITMQ_USER, RABBITMQ_PASS)
        parameters = pika.ConnectionParameters(
            host=RABBITMQ_HOST,
            credentials=credentials,
            heartbeat=3000  # ป้องกันการ timeout
        )
        connection = pika.BlockingConnection(parameters)
        channel = connection.channel()
        
        # เพิ่มการ log เมื่อเชื่อมต่อสำเร็จ
        logging.info("✅ เชื่อมต่อ RabbitMQ สำเร็จ")
        
        return connection, channel
    except pika.exceptions.AMQPConnectionError as e:
        logging.error(f"❌ Connection failed: {e}")
        raise

# ฟังก์ชันประมวลผลภาพที่ได้รับ
def process_image(camera_id, image):
    # สร้างโฟลเดอร์ถ้ายังไม่มี
    output_folder = "images"
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    image_count = len(os.listdir(output_folder)) + 1
    filename = os.path.join(output_folder, f"faces_image_{image_count}_{camera_id}.jpg")

    # ✅ ใช้ cv2.imwrite() เพื่อบันทึกรูปภาพ
    success = cv2.imwrite(filename, image)
    if success:
        logging.info(f"✅ บันทึกภาพจากกล้อง {camera_id} สำเร็จ: {filename}")
    else:
        logging.error(f"❌ ไม่สามารถบันทึกภาพจากกล้อง {camera_id}")

def callback(ch, method, properties, body):
    try:
        message = json.loads(body)
        camera_id = message.get("camera_id")
        image_base64 = message.get("image")

        if camera_id and image_base64:
            logging.info(f"📩 Received Base64 size: {len(image_base64)} characters")

            # Decode Base64
            image_bytes = base64.b64decode(image_base64)
            logging.info(f"📩 Decoded image_bytes size: {len(image_bytes)} bytes")

            # Decompress Data
            decompressed_data = zlib.decompress(image_bytes)
            logging.info(f"📩 Decompressed data size: {len(decompressed_data)} bytes")

            # Convert to NumPy array
            image_array = np.frombuffer(decompressed_data, dtype=np.uint8)
            logging.info(f"📩 Image array size: {image_array.shape if image_array.size > 0 else 'None'}")

            # 🔍 Debug: แสดงข้อมูลบรรทัดแรกของ image_array
            logging.info(f"📩 First 10 bytes of image array: {image_array[:10]}")

            # Decode image
            image = cv2.imdecode(image_array, cv2.IMREAD_UNCHANGED)
            if image is not None:
                logging.info(f"✅ Image successfully decoded! Shape: {image.shape}")
                process_image(camera_id, image)

                 # Get embedding result
                result = extract_face_embedding_rabbitmq(camera_id, image)
                
                if result:  # ตรวจสอบว่า result ไม่เป็น None
                    embedding_array = np.array(result["embedding"])  # แปลงเป็น numpy array
                    camera_id = result["camera_id"]  # ดึงค่า camera_id (ถ้ามี)
                    logging.info(f"✅ Extracted embedding from Camera {camera_id}: {embedding_array.shape}")
                    logging.info(f"🔍 Type of embedding: {type(result['embedding'])}")
                    logging.info(f"🔢 First 10 values of embedding: {embedding_array[:10]}")
                    get_best_match(embedding_array, redis_client, camera_id ,db=db_session, threshold=0.60, use_cosine=True)

                else:
                    logging.error(f"❌ ไม่สามารถดึง embedding จากกล้อง {camera_id}")


                ch.basic_ack(delivery_tag=method.delivery_tag)
            else:
                logging.error("❌ cv2.imdecode() คืนค่า None แสดงว่าแปลงภาพกลับไม่ได้")
                logging.warning(f"⚠️ ไม่สามารถแปลงภาพจากกล้อง {camera_id} ได้")

        else:
            logging.warning(f"⚠️ ไม่สามารถแปลงภาพจากกล้อง {camera_id} ได้")

    except Exception as e:
        logging.error(f"❌ Error processing message: {e}")


# ฟังก์ชันเริ่ม Consumer
def start_consumer():
    connection, channel = get_rabbitmq_connection()
    channel.basic_qos(prefetch_count=1)
    channel.basic_consume(queue=QUEUE_NAME, on_message_callback=callback)
    
    logging.info("🎧 Consumer กำลังรอรับภาพจาก RabbitMQ ...")
    channel.start_consuming()