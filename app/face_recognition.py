import insightface
import numpy as np
import pickle
import cv2
import logging

# โหลด ArcFace Model
model = insightface.app.FaceAnalysis(name="buffalo_l", providers=["CPUExecutionProvider"])
model.prepare(ctx_id=0, det_size=(640, 640))

# แปลงภาพเป็นเวกเตอร์
def extract_face_vector(image: np.ndarray) -> np.ndarray:
    faces = model.get(image)
    if len(faces) == 0:
        return None
    return faces[0].normed_embedding

def extract_face_embedding_rabbitmq(camera_id, image):
    """
    รับภาพจากกล้อง, ตรวจจับใบหน้า, และแปลงเป็นเวกเตอร์ (Face Embedding)
    
    :param camera_id: ID ของกล้อง
    :param image: ภาพจากกล้อง (NumPy array)
    :return: Dict { "camera_id": camera_id, "embedding": embedding.tolist() } หรือ None ถ้าไม่เจอใบหน้า
    """
    # ตรวจจับใบหน้าในภาพ
    faces = model.get(image)

    if len(faces) == 0:
        logging.error(f"❌ ไม่พบใบหน้าในภาพจากกล้อง {camera_id}")
        return None

    # ใช้ใบหน้าที่ใหญ่ที่สุด
    largest_face = max(faces, key=lambda face: face.bbox[2] * face.bbox[3])
    embedding = largest_face.normed_embedding

    if embedding is None:
        logging.error(f"❌ ไม่สามารถคำนวณ embedding จากกล้อง {camera_id}")
        return None

    # logging.info(f"✅ Extracted embedding from Camera {camera_id}: {embedding.shape}")

    return {
        "camera_id": camera_id,
        "embedding": embedding.tolist()  # แปลงเป็น List เพื่อให้ JSON-friendly
    }
