# HMMAF Live Test — Test Results

Run date: 2026-05-27
Environment: Python 3.11.1 / Linux WSL2 / CPU only
Live targets: native redis-server @ localhost:6379, LiteLLM gateway @ http://192.168.1.168:4000/v1 (`gemma-4-e4b-multimodal`)

## Component smoke test (`scripts/smoke_test.py`)

| # | Component | Status | Detail |
|---|---|---|---|
| 1 | Redis ping / publish | **PASS** | Native redis-server, no fakeredis |
| 2 | LLM gateway reachable | **PASS** | `gemma-4-e4b-multimodal` present in `/v1/models` |
| 3 | `GemmaEngine.reason` | **PASS** | 2.32s for trivial prompt, returned `'hello.'` |
| 4 | YOLO26n single-frame detect | **PASS** | 0.31s on `bus.jpg` → bus + 4 persons |
| 5 | SuperTonic2 TTS synth | **PASS** | ONNX model `supertonic-2` (5-language) auto-downloaded from HF, voice `M1` |
| 6 | faster-whisper STT | **PASS** | `small` model on CPU, int8 |

## Test suite (`pytest tests/`)

| Test | Status | Notes |
|---|---|---|
| `test_guardrail.py::test_agentic_recursion_prevention` | **PASS** | Recursion threshold 3-in-5s enforced |
| `test_guardrail.py::test_latency_flagging` | **PASS** | 1.5s latency budget enforced |
| `test_guardrail.py::test_non_blocking_throughput` | **PASS** | 1000 events / directives processed in < 1s |
| `test_guardrail_integration.py::test_e2e_runaway_reasoning_interception` | **PASS** | RogueGemmaEngine → Guardrail intercepts in < 3s |

**4/4 tests pass against REAL Redis** (no fakeredis).

## Live reasoning loop (`tests/integration_reasoning_loop.py`)

Status: **PASS** — Gemma 4 correctly emitted real tool calls for both scenarios:

| Scenario | Engine action | Tool result |
|---|---|---|
| "Record a red car entering the scene" | `update_world_model({entity_id:'red_car_1', attributes:{color:'red', label:'car'}})` | `status=success` |
| "Is there a red car?" | `search_memory({query:'red car'})` | 1 entity returned, memory roundtrip verified |

## Latency benchmarks (`scripts/benchmark_latency.py`)

### Overhead mode (matches original `HMMAF/scripts/benchmark_latency.py` methodology)

```
Samples Collected: 9
Average Latency:   0.101s    ← 100ms stub-engine sleep + ~1ms framework overhead
Minimum Latency:   0.101s
Maximum Latency:   0.102s
Target Latency:    < 1.500s
Status:            PASS
```
The framework's own overhead (Orchestrator + Redis pub/sub + Guardrail + asyncio routing) is **sub-millisecond**, consistent with the original workspace's reported `0.001s` overhead figure.

### Live mode (real Gemma 4 multimodal endpoint)

```
Samples Collected: 3 (out of 5 requested; remaining 2 still pending when run window closed)
Average Latency:   29.6s
Minimum Latency:   27.5s
Maximum Latency:   32.4s
Target Latency:    < 1.500s
Status:            FAIL  (LLM bottleneck, not framework)
```

The 1.5s PRA-loop budget from the PRD assumes a **vLLM-class accelerated** Gemma 4 deployment (50+ tokens/sec, batching, GPU). The remote LiteLLM gateway in this lab returns full multimodal completions in ~14–30s on this hardware, so end-to-end latency is dominated by inference, not by the agentic framework. The Guardrail correctly detected and logged the latency violation each time, demonstrating that monitoring works as designed.

Reports written to:
- `tests/latency_benchmark_report.overhead.txt`
- `tests/latency_benchmark_report.live.txt`
- `tests/latency_benchmark_report.txt` (last-run mirror)

## UI smoke test

`PYTHONPATH=. ./venv/bin/python src/ui/app.py` — booted cleanly:
- Bootstrapping HMMAF runtime
- Connected to Redis
- Orchestrator + Guardrail subscribed to both planes
- YOLO26n loaded (CPU)
- SuperTonic2 worker started
- Gradio served HTTP 200 on `http://localhost:7860`

## Overall

The Heterogeneous Multi-Modal Agentic Framework is **operational and ready for human acceptance testing.** All correctness tests pass, all live components are wired through real infrastructure (no mocks except where explicitly required to verify guardrail interception), and the multimodal UI exercises the full PRA loop.

The only `FAIL` is the **end-to-end latency budget**, which is a property of the *current LLM endpoint* (slow remote multimodal inference on this hardware), not of the framework itself. To meet the 1.5s budget, point the framework at a faster Gemma 4 deployment (e.g. vLLM with batching on a Blackwell-class GPU, per the PRD's hardware target).
