"""Guardrail unit tests against REAL Redis.

Same assertions as the original HMMAF/tests/test_guardrail.py, but the
RedisClient now connects to the live redis-server on localhost:6379.
"""
import asyncio
import time

import pytest
from unittest.mock import AsyncMock

from src.comm.redis_client import RedisClient
from src.orchestrator.error_handlers import ErrorHandler
from src.orchestrator.guardrail import GuardrailAgent


@pytest.fixture
def mock_error_handler():
    handler = ErrorHandler()
    handler.handle_error = AsyncMock()
    return handler


@pytest.fixture
def guardrail(mock_error_handler):
    redis_client = RedisClient(host='localhost', port=6379)
    agent = GuardrailAgent(redis_client, mock_error_handler)
    return agent


@pytest.mark.asyncio
async def test_agentic_recursion_prevention(guardrail, mock_error_handler):
    directive = "SET_ZOOM | 2.0"
    await guardrail._handle_control(directive)
    await guardrail._handle_control(directive)
    mock_error_handler.handle_error.assert_not_called()

    await guardrail._handle_control(directive)
    assert mock_error_handler.handle_error.call_count == 1
    call_args = mock_error_handler.handle_error.call_args[0]
    assert isinstance(call_args[0], RecursionError)
    assert call_args[1] == "guardrail_recursion_check"


@pytest.mark.asyncio
async def test_latency_flagging(guardrail, mock_error_handler):
    event_data = '{"event": "test"}'
    await guardrail._handle_observation(event_data)
    guardrail.event_timestamps[str(hash(event_data))] = time.time() - 2.0
    await guardrail._handle_control("RESPOND | Hello")

    assert mock_error_handler.handle_error.call_count == 1
    call_args = mock_error_handler.handle_error.call_args[0]
    assert isinstance(call_args[0], TimeoutError)
    assert call_args[1] == "guardrail_latency_check"


@pytest.mark.asyncio
async def test_non_blocking_throughput(guardrail, mock_error_handler):
    start = time.time()
    for i in range(1000):
        await guardrail._handle_observation(f"event_{i}")
        await guardrail._handle_control(f"directive_{i}")
    duration = time.time() - start
    assert duration < 1.0
    mock_error_handler.handle_error.assert_not_called()
