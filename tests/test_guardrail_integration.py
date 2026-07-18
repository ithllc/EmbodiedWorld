"""End-to-end guardrail interception test against REAL Redis.

Mirrors HMMAF/tests/test_guardrail_integration.py but the Orchestrator's
RedisClient now talks to the local redis-server.
"""
import asyncio
import json

import pytest

from src.comm.redis_client import RedisClient
from src.memory.world_model import WorldModel
from src.orchestrator.core import Orchestrator
from src.reasoning.gemma_engine import GemmaEngine
from src.reasoning.mcp_tools import MCPTools


class RogueGemmaEngine(GemmaEngine):
    async def reason(self, prompt, context=None, tools=None, images=None):
        return {
            "action": "mock_action",
            "arguments": {"target": "spam"},
            "reasoning": "I must do this forever",
        }


class MockMCPTools(MCPTools):
    async def call_tool(self, name, args):
        return {"status": "success", "result": "mocked"}


@pytest.mark.asyncio
async def test_e2e_runaway_reasoning_interception():
    redis_client = RedisClient(host='localhost', port=6379)
    rogue_engine = RogueGemmaEngine()
    tools = MockMCPTools()
    world_model = WorldModel()

    orch = Orchestrator(redis_client, rogue_engine, tools, world_model)
    await orch.start()

    error_raised = asyncio.Event()

    async def mock_handle_error(error, context):
        print(f"ERROR HANDLER CALLED: {context} - {error}")
        if "recursion" in context.lower():
            error_raised.set()

    orch.error_handler.handle_error = mock_handle_error
    orch.guardrail.error_handler.handle_error = mock_handle_error

    await redis_client.publish('observation_plane', json.dumps({
        'event_type': 'test', 'data': 'start',
    }))

    for i in range(5):
        await orch._process_agentic_reasoning({"event": "mock"})
        await asyncio.sleep(0.1)

    try:
        await asyncio.wait_for(error_raised.wait(), timeout=3.0)
    except asyncio.TimeoutError:
        pytest.fail("Guardrail did not catch recursive tool calls.")

    await orch.stop()
