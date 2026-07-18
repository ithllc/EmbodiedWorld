"""Orchestrator (Agent 1 - The Brain).

Subscribes to the Observation Plane, drives the Reasoning Engine through tools,
and publishes Control Plane directives.
"""
import asyncio
import json
import logging
from typing import Any, Dict, Optional

from src.comm.redis_client import RedisClient
from src.reasoning.gemma_engine import GemmaEngine
from src.reasoning.mcp_tools import MCPTools
from src.memory.world_model import WorldModel
from src.orchestrator.error_handlers import ErrorHandler
from src.orchestrator.guardrail import GuardrailAgent

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger('HMMAF.Orchestrator')


class Orchestrator:
    def __init__(
        self,
        redis_client: RedisClient,
        gemma_engine: Optional[GemmaEngine],
        mcp_tools: Optional[MCPTools],
        world_model: Optional[WorldModel],
    ):
        self.redis = redis_client
        self.engine = gemma_engine
        self.tools = mcp_tools
        self.world_model = world_model

        self.error_handler = ErrorHandler(orchestrator=self)
        self.guardrail = GuardrailAgent(redis_client, self.error_handler)

        self.is_running = False
        self.current_state: Dict[str, Any] = {
            'status': 'idle',
            'active_agents': [],
            'last_event': None,
        }

        self.observation_channel = 'observation_plane'
        self.control_channel = 'control_plane'
        self.last_response_text: Optional[str] = None

    async def start(self):
        logger.info('Starting Agentic HMMAF Orchestrator...')
        self.is_running = True
        if not hasattr(self.redis, 'client') or self.redis.client is None:
            await self.redis.connect()
        await self.guardrail.start()
        self.listener_task = asyncio.create_task(self._listen_loop())
        logger.info('Orchestrator is running.')

    async def stop(self):
        logger.info('Stopping HMMAF Orchestrator...')
        self.is_running = False
        if hasattr(self, 'listener_task'):
            self.listener_task.cancel()
            try:
                await self.listener_task
            except asyncio.CancelledError:
                pass
        await self.guardrail.stop()
        await self.redis.close()
        logger.info('Orchestrator stopped.')

    async def _listen_loop(self):
        ps = await self.redis.subscribe(self.observation_channel)
        try:
            while self.is_running:
                message = ps.get_message(ignore_subscribe_messages=True)
                if message:
                    await self._handle_event(message['data'])
                await asyncio.sleep(0.01)
        except asyncio.CancelledError:
            logger.info('Listener task cancelled.')
        except Exception as e:
            logger.error(f'Error in listener loop: {e}')
        finally:
            logger.info('Listener loop exited.')

    async def _handle_event(self, event_data: str):
        logger.info(f'New Event Received: {event_data[:200]}')
        self.current_state['last_event'] = event_data
        try:
            event_dict = json.loads(event_data)
            await self._process_agentic_reasoning(event_dict)
        except json.JSONDecodeError:
            logger.warning('Received non-JSON event. Falling back to raw text.')
            await self._process_agentic_reasoning({'event_type': 'raw_text', 'content': event_data})

    async def _process_agentic_reasoning(self, event_dict: Dict[str, Any]):
        self.current_state['status'] = 'reasoning'
        context_str = json.dumps(event_dict)

        if event_dict.get('event_type') == 'user_speech':
            prompt = (
                "The user said: \"" + str(event_dict.get('content', '')) + "\". "
                "Relevant current perception context (most recent YOLO detections) is in the Context block. "
                "If you have something to say to the user, call the 'speak' tool with a short reply. "
                "If you need to remember a fact, call 'update_world_model'. If you need to recall, call 'search_memory'."
            )
        elif event_dict.get('event_type') == 'object_detection':
            prompt = (
                "A new object-detection event arrived from the YOLO perception module. "
                "Decide whether this is worth noting: if it adds new information about the user's scene, "
                "call 'update_world_model' to remember it. Otherwise reply with RESPOND and a short reasoning line. "
                "Do NOT call 'speak' unless a user has explicitly asked something."
            )
        else:
            prompt = ("An event occurred: " + context_str +
                      ". Analyze it and use a tool if action is required.")

        try:
            reasoning_result = await self.engine.reason(
                prompt=prompt,
                context=context_str,
                tools=self.tools.get_tool_definitions() if self.tools else None,
            )
            logger.info(f'Reasoning Output: {reasoning_result}')
            self.current_state['status'] = 'acting'

            if reasoning_result.get('action') and reasoning_result['action'] != 'RESPOND':
                action_name = reasoning_result['action']
                args = reasoning_result.get('arguments') or {}
                logger.info(f'Executing Agentic Action: {action_name} with {args}')
                tool_result = await self.tools.call_tool(action_name, args)
                if tool_result.get('status') == 'success':
                    await self._publish_directive(
                        f'ACTION_COMPLETED | {action_name} | {json.dumps(args, default=str)}'
                    )
                else:
                    await self._publish_directive(
                        f'ACTION_FAILED | {action_name} | {tool_result.get("error") or tool_result.get("message")}'
                    )
            else:
                text = reasoning_result.get('reasoning') or ''
                if text:
                    self.last_response_text = text
                    await self._publish_directive(f'TEXT_RESPONSE | {text}')

            self.current_state['status'] = 'idle'
        except Exception as e:
            logger.error(f'Agentic reasoning failure: {e}')
            self.current_state['status'] = 'idle'
            await self._publish_directive(f'ERROR | Orchestrator reasoning failed: {e}')

    async def _publish_directive(self, directive: str):
        logger.info(f'Publishing Directive: {directive[:200]}')
        await self.redis.publish(self.control_channel, directive)

    async def get_status(self) -> Dict[str, Any]:
        return {'is_running': self.is_running, 'state': self.current_state}


async def main():
    client = RedisClient()
    engine = GemmaEngine()
    world_model = WorldModel()
    tools = MCPTools(world_model=world_model)
    orch = Orchestrator(client, engine, tools, world_model)
    await orch.start()
    logger.info("Orchestrator main loop running. Ctrl-C to stop.")
    try:
        while True:
            await asyncio.sleep(1)
    except KeyboardInterrupt:
        pass
    finally:
        await orch.stop()


if __name__ == '__main__':
    asyncio.run(main())
