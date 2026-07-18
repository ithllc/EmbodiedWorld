"""YOLO Perception Proxy.

Wraps Ultralytics YOLO26 (CPU) and publishes structured VisionEvent JSON
to Redis on the observation_plane channel.

Two modes of use:
  - Streaming mode (`start()` / `stop()`): polls a video source continuously.
  - On-demand mode (`detect_once(image)`): used by the Gradio UI to score a single frame.
"""
import asyncio
import datetime
import json
import logging
from typing import Any, Dict, List, Optional, Union

import numpy as np
from ultralytics import YOLO

from src.comm.redis_client import RedisClient
from src.models.vision_schema import Detection, VisionEvent

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger('HMMAF.Perception.YOLOProxy')


class YOLOProxy:
    def __init__(
        self,
        redis_client: Optional[RedisClient],
        model_path: str = "yolo26n.pt",
        observation_channel: str = "observation_plane",
        source: Union[str, int, None] = 0,
        device: str = "cpu",
        conf: float = 0.25,
    ):
        self.redis = redis_client
        self.model_path = model_path
        self.observation_channel = observation_channel
        self.source = source
        self.device = device
        self.conf = conf
        self.is_running = False
        self._stop_event = asyncio.Event()
        self.model: Optional[YOLO] = None

    def _ensure_model_loaded(self):
        if self.model is None:
            logger.info(f"Loading YOLO model: {self.model_path} on {self.device}")
            self.model = YOLO(self.model_path)

    async def load(self):
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(None, self._ensure_model_loaded)
        logger.info("YOLO model ready.")

    async def start(self):
        await self.load()
        logger.info(f"Starting YOLOProxy detection loop on source: {self.source}")
        self.is_running = True
        self._stop_event.clear()
        asyncio.create_task(self._run_detection())

    async def stop(self):
        logger.info("Stopping YOLOProxy ...")
        self.is_running = False
        self._stop_event.set()

    async def _run_detection(self):
        loop = asyncio.get_running_loop()
        try:
            while self.is_running:
                if self._stop_event.is_set():
                    break
                results_gen = await loop.run_in_executor(
                    None,
                    lambda: self.model.predict(
                        source=self.source,
                        conf=self.conf,
                        verbose=False,
                        stream=True,
                        device=self.device,
                    ),
                )
                for result in results_gen:
                    if not self.is_running:
                        break
                    event = self._process_result(result)
                    if event and self.redis:
                        await self._publish_event(event)
                    await asyncio.sleep(0)
                await asyncio.sleep(0.01)
        except Exception as e:
            logger.error(f"Error in YOLOProxy detection loop: {e}")
        finally:
            logger.info("YOLOProxy detection loop exited.")

    def detect_once_sync(self, image) -> VisionEvent:
        """Synchronous single-frame detect. `image` is a numpy array (HxWxC, BGR or RGB)
        or a filepath/URL string."""
        self._ensure_model_loaded()
        results = self.model.predict(
            source=image, conf=self.conf, verbose=False, device=self.device
        )
        # results is a list when stream=False
        if not results:
            return VisionEvent(
                timestamp=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                sensor_id='yolo_camera_01',
                event_type='object_detection',
                detections=[],
            )
        event = self._process_result(results[0])
        if event is None:
            event = VisionEvent(
                timestamp=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                sensor_id='yolo_camera_01',
                event_type='object_detection',
                detections=[],
            )
        return event

    async def detect_once(self, image) -> VisionEvent:
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, self.detect_once_sync, image)

    def _process_result(self, result) -> Optional[VisionEvent]:
        detections: List[Detection] = []
        if result.boxes is None:
            return None
        for box in result.boxes:
            cls_id = int(box.cls[0])
            label = result.names[cls_id]
            conf = float(box.conf[0])
            bbox_raw = box.xyxy[0].tolist()
            bbox = [int(coord) for coord in bbox_raw]
            attrs = self._get_extended_attributes(label, box)
            detections.append(Detection(label=label, confidence=round(conf, 3), bbox=bbox, attributes=attrs))

        if not detections:
            return VisionEvent(
                timestamp=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                sensor_id='yolo_camera_01',
                event_type='object_detection',
                detections=[],
            )
        return VisionEvent(
            timestamp=datetime.datetime.now(datetime.timezone.utc).isoformat(),
            sensor_id='yolo_camera_01',
            event_type='object_detection',
            detections=detections,
        )

    def _get_extended_attributes(self, label: str, box) -> Dict[str, Any]:
        return {"source": "ultralytics_yolo26"}

    async def _publish_event(self, event: VisionEvent):
        try:
            await self.redis.publish(self.observation_channel, json.dumps(event.to_dict()))
        except Exception as e:
            logger.error(f"Failed to publish event: {e}")


async def main():
    client = RedisClient()
    await client.connect()
    proxy = YOLOProxy(client, model_path="yolo26n.pt", source=None)
    await proxy.load()
    # quick single-frame demo with the bus.jpg from upstream repo
    event = await proxy.detect_once("https://ultralytics.com/images/bus.jpg")
    print(json.dumps(event.to_dict(), indent=2))
    await client.close()


if __name__ == '__main__':
    asyncio.run(main())
