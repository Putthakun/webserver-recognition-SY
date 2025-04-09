import os
import redis
import numpy as np
import logging
from redis.commands.search.field import VectorField
from redis.commands.search.indexDefinition import IndexDefinition, IndexType

logging.basicConfig(level=logging.DEBUG)

# 📌 Load Redis config from env (or use default)
REDIS_HOST = "redis" 
REDIS_PORT = 6379
REDIS_DB = 0

# ✅ Binary-safe Redis client
redis_client = redis.Redis(
    host=REDIS_HOST,
    port=REDIS_PORT,
    db=REDIS_DB,
    decode_responses=False
)

# 🧹 ลบ index เดิมก่อน (ใช้ใน dev เท่านั้น!)
try:
    redis_client.ft("face_vectors_idx").dropindex(delete_documents=False)
    print("🗑️ Dropped existing index")
except Exception as e:
    print(f"ℹ️ No index to drop: {e}")

# 🔍 ตรวจสอบว่า index มีอยู่แล้วหรือยัง
try:
    redis_client.ft("face_vectors_idx").info()
    print("ℹ️ Redis index 'face_vectors_idx' already exists. Skipping creation.")
except redis.exceptions.ResponseError:
    print("🚀 Creating Redis HNSW vector index...")

    schema = (
        VectorField("vector", "HNSW", {
            "TYPE": "FLOAT32",
            "DIM": 512,
            "DISTANCE_METRIC": "COSINE",  # ✅ ต้องมีบรรทัดนี้
            "INITIAL_CAP": 2000,
            "EF_CONSTRUCTION": 200,
            "M": 16
        }),
    )

    redis_client.ft("face_vectors_idx").create_index(
        fields=schema,
        definition=IndexDefinition(prefix=["face_vector:"], index_type=IndexType.HASH)
    )
    print("✅ Redis vector index created.")

