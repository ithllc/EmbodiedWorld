"""HMMAF_Live_Test: Redis client wrapper.

Defaults to REAL Redis (no fakeredis), so the framework exercises the same
pub/sub semantics as a production deployment.
"""
import logging
import asyncio
from typing import Optional, Union

import redis
import fakeredis

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("HMMAF.Comm")


class RedisClient:
    """Thin async-facing wrapper around redis-py's pub/sub.

    `use_fakeredis=False` is the live-test default. Pass `True` only in unit
    tests that intentionally want process-local in-memory pub/sub.
    """

    def __init__(self, host: str = 'localhost', port: int = 6379, use_fakeredis: bool = False):
        self.host = host
        self.port = port
        self.use_fakeredis = use_fakeredis
        self.client: Optional[Union[redis.Redis, fakeredis.FakeRedis]] = None
        self.pubsub: Optional[redis.client.PubSub] = None

    async def connect(self):
        try:
            if self.use_fakeredis:
                logger.info("Connecting to fakeredis (simulation mode)...")
                self.client = fakeredis.FakeRedis(decode_responses=True)
            else:
                logger.info(f"Connecting to Redis at {self.host}:{self.port}...")
                self.client = redis.Redis(host=self.host, port=self.port, decode_responses=True)

            if self.client.ping():
                logger.info("Successfully connected to Redis.")
            else:
                raise ConnectionError("Redis ping failed.")
        except Exception as e:
            logger.error(f"Failed to connect to Redis: {e}")
            raise

    async def publish(self, channel: str, message: str):
        if not self.client:
            raise RuntimeError("Redis client not connected. Call connect() first.")
        try:
            self.client.publish(channel, message)
            logger.debug(f"Published to {channel}: {message[:120]}")
        except Exception as e:
            logger.error(f"Error publishing to {channel}: {e}")
            raise

    async def subscribe(self, channel: str):
        if not self.client:
            raise RuntimeError("Redis client not connected. Call connect() first.")
        try:
            ps = self.client.pubsub()
            ps.subscribe(channel)
            logger.info(f"Subscribed to channel: {channel}")
            return ps
        except Exception as e:
            logger.error(f"Error subscribing to {channel}: {e}")
            raise

    async def close(self):
        if self.client:
            self.client.close()
            logger.info("Redis connection closed.")


if __name__ == "__main__":
    async def test():
        client = RedisClient()
        await client.connect()
        await client.publish("test_channel", "Hello HMMAF Live!")
        ps = await client.subscribe("test_channel")
        await asyncio.sleep(0.1)
        msg = ps.get_message(ignore_subscribe_messages=True)
        print(f"Received: {msg}")
        await client.close()

    asyncio.run(test())
