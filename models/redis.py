from redis import from_url
from redis_cache import RedisCache

from .config import config

redis_client = from_url(config('redis_url'))
cache = RedisCache(redis_client=redis_client)
