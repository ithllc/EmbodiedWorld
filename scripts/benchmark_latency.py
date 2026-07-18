"""Latency benchmarks for HMMAF Live Test.

Two modes — both run against REAL Redis (no fakeredis):

  --mode=overhead   (default)
      Measures Orchestrator + Redis + asyncio overhead only, using a
      synthetic 100 ms reasoning engine. This matches the methodology
      of the original HMMAF/scripts/benchmark_latency.py (which used a
      mock engine returning RESPOND), so the numbers are directly
      comparable to /tests/phase4_optimization_report.md.

  --mode=live
      Full end-to-end against the real LiteLLM gateway serving
      gemma-4-e4b-multimodal. Time-to-directive depends on the LLM's
      tokens/sec; the 1.5s PRA-loop budget from the PRD assumes a
      vLLM-class accelerated deployment. Failure here means the LLM is
      the bottleneck, not the framework.

Usage:
    python scripts/benchmark_latency.py [N_SAMPLES] [--mode=overhead|live]
"""
import asyncio
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import List

from src.comm.redis_client import RedisClient
from src.memory.world_model import WorldModel
from src.orchestrator.core import Orchestrator
from src.reasoning.gemma_engine import GemmaEngine
from src.reasoning.mcp_tools import MCPTools

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger('HMMAF.Benchmark')

REPORT_PATH = Path(__file__).resolve().parent.parent / 'tests' / 'latency_benchmark_report.txt'


class LatencyTracker:
    def __init__(self):
        self.samples: List[float] = []

    def record(self, latency: float):
        self.samples.append(latency)
        logger.info(f"Captured latency sample: {latency:.3f}s")

    def report(self) -> str:
        if not self.samples:
            return "No latency samples collected.\n"
        avg = sum(self.samples) / len(self.samples)
        mn = min(self.samples)
        mx = max(self.samples)
        ok = avg < 1.5
        text = (
            "\n" + "=" * 30 + "\n"
            "LATENCY BENCHMARK RESULTS\n"
            + "=" * 30 + "\n"
            f"Samples Collected: {len(self.samples)}\n"
            f"Average Latency:   {avg:.3f}s\n"
            f"Minimum Latency:   {mn:.3f}s\n"
            f"Maximum Latency:   {mx:.3f}s\n"
            f"Target Latency:    < 1.500s\n"
            f"Status:            {'PASS' if ok else 'FAIL'}\n"
            + "=" * 30 + "\n"
        )
        return text


class _StubEngine:
    """Synthetic engine: 100 ms 'reason' returning RESPOND. Matches the
    original /scripts/benchmark_latency.py mock to isolate framework overhead."""

    async def reason(self, prompt, context=None, tools=None, images=None):
        await asyncio.sleep(0.1)
        return {'action': 'RESPOND', 'reasoning': 'Acknowledged.'}


async def run_benchmark(n_samples: int = 5, mode: str = 'overhead'):
    print(f'--- Starting Latency Benchmark (real Redis, mode={mode}) ---')
    tracker = LatencyTracker()

    client = RedisClient()
    world_model = WorldModel()
    tools = MCPTools(world_model=world_model)
    if mode == 'overhead':
        engine = _StubEngine()
    elif mode == 'live':
        engine = GemmaEngine()
    else:
        raise SystemExit(f"Unknown mode: {mode}")
    orch = Orchestrator(client, engine, tools, world_model)
    await orch.start()

    # Wrap _handle_event so we record wall-clock latency from event arrival
    # to directive publication.
    original_handle = orch._handle_event
    original_publish = orch._publish_directive
    pending_start_times: List[float] = []

    async def instrumented_handle(event_data: str):
        pending_start_times.append(time.time())
        await original_handle(event_data)

    async def instrumented_publish(directive: str):
        if pending_start_times:
            t0 = pending_start_times.pop(0)
            tracker.record(time.time() - t0)
        await original_publish(directive)

    orch._handle_event = instrumented_handle
    orch._publish_directive = instrumented_publish

    # Subscribe a separate listener on control_plane so we can confirm directives
    # are flowing through Redis (not just in-process).
    ctrl_client = RedisClient()
    await ctrl_client.connect()
    ctrl_ps = await ctrl_client.subscribe('control_plane')
    seen = 0

    try:
        for i in range(n_samples):
            event = {
                'event_type': 'object_detection',
                'detections': [{'label': 'person', 'confidence': 0.9, 'bbox': [0, 0, 100, 100], 'attributes': {}}],
                'sensor_id': 'bench',
                'timestamp': time.time(),
            }
            await client.publish('observation_plane', json.dumps(event))
            # Wait for the corresponding control_plane message
            deadline = time.time() + 30
            while time.time() < deadline:
                msg = ctrl_ps.get_message(ignore_subscribe_messages=True)
                if msg and isinstance(msg.get('data'), str):
                    seen += 1
                    break
                await asyncio.sleep(0.02)
            else:
                logger.warning("Timed out waiting for directive on control_plane")
    finally:
        await orch.stop()
        await ctrl_client.close()

    report = tracker.report()
    print(report)
    REPORT_PATH.parent.mkdir(exist_ok=True)
    out_path = REPORT_PATH.with_name(f"latency_benchmark_report.{mode}.txt")
    out_path.write_text(f"Mode: {mode}\n" + report)
    REPORT_PATH.write_text(f"Mode: {mode}\n" + report)
    print(f"Wrote {out_path}")
    print(f"Directives seen on control_plane: {seen}/{n_samples}")
    return tracker


if __name__ == '__main__':
    n = 5
    mode = 'overhead'
    for arg in sys.argv[1:]:
        if arg.startswith('--mode='):
            mode = arg.split('=', 1)[1]
        elif arg.isdigit():
            n = int(arg)
    asyncio.run(run_benchmark(n, mode))
