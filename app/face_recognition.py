# import lib
import insightface
import numpy as np
import pickle
import cv2
import logging

# insightface Model
model = insightface.app.FaceAnalysis(name="buffalo_l", providers=["CPUExecutionProvider"])
model.prepare(ctx_id=0, det_size=(640, 640))

# Convert images to vector
def extract_face_vector(image: np.ndarray) -> np.ndarray:
    faces = model.get(image)
    if len(faces) == 0:
        return None
    return faces[0].normed_embedding

def extract_face_embedding_rabbitmq(camera_id, image):

    # Face detection
    faces = model.get(image)

    if len(faces) == 0:
        logging.error(f"❌ No face found in camera image {camera_id}")
        return None

    # Use the largest face
    largest_face = max(faces, key=lambda face: face.bbox[2] * face.bbox[3])
    embedding = largest_face.normed_embedding

    if embedding is None:
        logging.error(f"❌ Unable to calculate embedding from camera {camera_id}")
        return None

    return {
        "camera_id": camera_id,
        "embedding": embedding.tolist()  # Convert to List for JSON-friendly
    }
