"""World Model (Memory Agent).

In-process dict store with optional Redis persistence. Keeps the same
public API and history schema as the original HMMAF workspace.
"""
import json
import logging
import asyncio
from typing import Any, Dict, List, Optional

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger('HMMAF.Memory.WorldModel')


class WorldModel:
    def __init__(self, redis_client=None):
        self.redis = redis_client
        self.entities: Dict[str, Dict[str, Any]] = {}
        self._lock = asyncio.Lock()

    async def update(self, entity_id: str, attributes: Dict[str, Any]) -> bool:
        async with self._lock:
            logger.info(f"WorldModel: Updating entity '{entity_id}' with {attributes}")
            if entity_id not in self.entities:
                self.entities[entity_id] = {"last_seen": None, "history": []}
            self.entities[entity_id].update(attributes)
            self.entities[entity_id]["history"].append({
                "timestamp": asyncio.get_event_loop().time(),
                "attributes": attributes,
            })
            if self.redis and getattr(self.redis, 'client', None) is not None:
                try:
                    self.redis.client.set(
                        f"world:entity:{entity_id}",
                        json.dumps(self.entities[entity_id], default=str),
                    )
                except Exception as e:
                    logger.warning(f"Redis persistence failed for {entity_id}: {e}")
            return True

    async def query(self, query_str: str) -> List[Dict[str, Any]]:
        logger.info(f"WorldModel: Querying for '{query_str}'")
        results = []
        for eid, data in self.entities.items():
            if query_str.lower() in str(data).lower() or query_str.lower() in eid.lower():
                results.append({"entity_id": eid, **data})
        return results

    async def get_entity(self, entity_id: str) -> Optional[Dict[str, Any]]:
        return self.entities.get(entity_id)


async def main():
    wm = WorldModel()
    await wm.update("person_1", {"label": "person", "color": "red"})
    await wm.update("person_1", {"activity": "walking"})
    print(await wm.get_entity("person_1"))
    print(await wm.query("person"))


if __name__ == "__main__":
    asyncio.run(main())
