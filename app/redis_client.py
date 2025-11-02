import os
import redis
import logging
from redis.commands.search.field import VectorField
from redis.commands.search.index_definition import IndexDefinition, IndexType

logging.basicConfig(level=logging.INFO)

# --- Redis config ---
REDIS_HOST = "redis" 
REDIS_PORT = 6379
REDIS_DB = 0

redis_client = redis.Redis(
    host=REDIS_HOST,
    port=REDIS_PORT,
    db=REDIS_DB,
    decode_responses=False  
)

# --- check connected ---
try:
    info = redis_client.info()
    logging.info(f"✅ Connected to Redis {info.get('redis_version')} on {REDIS_HOST}:{REDIS_PORT}")
except Exception as e:
    logging.error(f"❌ Cannot connect to Redis: {e}")
    exit(1)

# --- check index ---
try:
    redis_client.ft("face_vectors_idx").info()
    print("ℹ️ Redis index 'face_vectors_idx' already exists. Skipping creation.")
except redis.exceptions.ResponseError:
    print("🚀 Creating Redis HNSW vector index...")

    schema = (
        VectorField("vector", "HNSW", {
            "TYPE": "FLOAT32",
            "DIM": 512,
            "DISTANCE_METRIC": "COSINE",
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
