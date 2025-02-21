import redis
import os

# กำหนดค่า default เป็น 'redis' เพื่อให้สื่อสารกับ Redis container ได้
REDIS_HOST = os.getenv("REDIS_HOST", "redis")
REDIS_PORT = int(os.getenv("REDIS_PORT", 6379))

redis_client = redis.Redis(host=REDIS_HOST, port=REDIS_PORT, db=0)
