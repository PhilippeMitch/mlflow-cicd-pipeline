import os
import redis
from dotenv import load_dotenv
load_dotenv(".env")
override = False

def initialize_redis():
    try:
        redis_password = os.environ.get("REDIS_PASSWORD")
        redis_port = int(os.environ.get("REDIS_PORT", 6379))
        redis_host = os.environ.get("REDIS_HOST", "localhost")
        print(f"redis_port: {redis_port}")
        r = redis.Redis(
            host=redis_host,
            port=redis_port,
            db=0,
            password=redis_password,
            decode_responses=True
        )
        r.ping()
    except redis.ConnectionError as e:
        print(f"Warning: Failed to connect to Redis: {e}")
        r = None
    return r

r = initialize_redis()