"""Guardrail / Monitor Agent (Agent 5).

Watches both planes for agentic recursion and latency budget violations.
"""
import asyncio
import logging
import time
from typing import Any, Dict, List, Optional

from src.comm.redis_client import RedisClient
from src.orchestrator.error_handlers import ErrorHandler

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger('HMMAF.GuardrailAgent')


class GuardrailAgent:
    def __init__(self, redis_client: RedisClient, error_handler: ErrorHandler):
        self.redis = redis_client
        self.error_handler = error_handler
        self.is_running = False
        self.observation_channel = 'observation_plane'
        self.control_channel = 'control_plane'

        self.directive_history: List[Dict[str, Any]] = []
        self.max_history = 100
        self.recursion_threshold = 3
        self.recursion_window = 5.0

        self.event_timestamps: Dict[str, float] = {}
        self.max_latency = 1.5

        self.listener_task_obs: Optional[asyncio.Task] = None
        self.listener_task_ctrl: Optional[asyncio.Task] = None

    async def start(self):
        logger.info("Starting Guardrail/Monitor Agent...")
        self.is_running = True
        if not hasattr(self.redis, 'client') or self.redis.client is None:
            await self.redis.connect()
        self.listener_task_obs = asyncio.create_task(self._listen_observation_plane())
        self.listener_task_ctrl = asyncio.create_task(self._listen_control_plane())

    async def stop(self):
        logger.info("Stopping Guardrail Agent...")
        self.is_running = False
        for task in (self.listener_task_obs, self.listener_task_ctrl):
            if task is not None:
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass

    async def _listen_observation_plane(self):
        ps = await self.redis.subscribe(self.observation_channel)
        try:
            while self.is_running:
                message = ps.get_message(ignore_subscribe_messages=True)
                if message:
                    await self._handle_observation(message['data'])
                await asyncio.sleep(0.01)
        except asyncio.CancelledError:
            pass
        except Exception as e:
            logger.error(f"Guardrail error on observation plane: {e}")

    async def _listen_control_plane(self):
        ps = await self.redis.subscribe(self.control_channel)
        try:
            while self.is_running:
                message = ps.get_message(ignore_subscribe_messages=True)
                if message:
                    await self._handle_control(message['data'])
                await asyncio.sleep(0.01)
        except asyncio.CancelledError:
            pass
        except Exception as e:
            logger.error(f"Guardrail error on control plane: {e}")

    async def _handle_observation(self, event_data: str):
        event_hash = str(hash(event_data))
        self.event_timestamps[event_hash] = time.time()
        if len(self.event_timestamps) > self.max_history:
            oldest = list(self.event_timestamps.keys())[0]
            del self.event_timestamps[oldest]

    async def _handle_control(self, directive_data: str):
        now = time.time()

        if self.event_timestamps:
            latest_event_time = list(self.event_timestamps.values())[-1]
            latency = now - latest_event_time
            if latency > self.max_latency:
                msg = f"Latency violation: {latency:.2f}s > {self.max_latency}s"
                logger.warning(msg)
                await self.error_handler.handle_error(TimeoutError(msg), "guardrail_latency_check")

        self.directive_history.append({'time': now, 'directive': directive_data})
        self.directive_history = [d for d in self.directive_history if now - d['time'] <= self.recursion_window]

        identical_count = sum(1 for d in self.directive_history if d['directive'] == directive_data)
        if identical_count >= self.recursion_threshold:
            msg = (f"Agentic Recursion Detected! Directive '{directive_data}' repeated "
                   f"{identical_count} times in {self.recursion_window}s.")
            logger.error(msg)
            await self.error_handler.handle_error(RecursionError(msg), "guardrail_recursion_check")
            self.directive_history = [d for d in self.directive_history if d['directive'] != directive_data]
