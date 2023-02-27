from .config import config
from redis_rate_limit import RateLimit
from .redis import redis_client


def max_60_per_minute(function):
    def wrapped_function(*args, **kwargs):
        client = config('client')
        pool = redis_client.connection_pool
        rate_limit = RateLimit("openai", client, 60, 60, pool)
        return rate_limit(function)(*args, **kwargs)
    return wrapped_function
