# HMMAF Live Test — How to Use

This workspace is a **live, end-to-end** instantiation of the Heterogeneous
Multi-Modal Agentic Framework (HMMAF). Every component runs against real
infrastructure: a local Redis daemon, a remote LiteLLM gateway serving
`gemma-4-e4b-multimodal`, Ultralytics **YOLO26n** on CPU, **SuperTonic2** (with
edge-tts fallback) for TTS, and **faster-whisper** for STT.

---

## 1. What you get

| Component | Implementation | Where |
|---|---|---|
| Reasoning Engine | Gemma 4 (multimodal) via LiteLLM at `http://192.168.1.168:4000/v1` | `src/reasoning/gemma_engine.py` |
| Perception | Ultralytics **YOLO26n.pt** on CPU | `src/perception/yolo_proxy.py` |
| Expression (TTS) | **SuperTonic2** (ONNX) → edge-tts fallback | `src/expression/supertone_worker.py` |
| Speech Input (STT) | faster-whisper `small` on CPU | `src/expression/supertone_worker.py` (`WhisperSTT`) |
| Communication | Redis (native, **not Docker**) on localhost:6379 | `src/comm/redis_client.py` |
| Memory | World Model (in-process dict + optional Redis-backed persistence) | `src/memory/world_model.py` |
| Orchestrator | Dual-plane async PRA loop | `src/orchestrator/core.py` |
| Guardrail | Recursion + latency monitor on both planes | `src/orchestrator/guardrail.py` |
| User Interface | Gradio web app | `src/ui/app.py` |

---

## 2. Using a different model endpoint (Nebius, RunPod, etc.)

The reasoning engine (`src/reasoning/gemma_engine.py`) is **highly
interchangeable** at the endpoint level by design. It talks to the model through
**LiteLLM** using the generic **OpenAI-compatible** provider prefix
(`model="openai/<name>"` + `api_base` + `api_key`), so any provider that exposes
an OpenAI-compatible `/v1` route — including **Nebius Token Factory** (AI Studio)
and **RunPod** (Serverless vLLM endpoints or RunPod's hosted API) — can be used
as a drop-in replacement.

### Swap the endpoint with three environment variables

No code changes are required — all three connection parameters are env-driven:

| Env var | Default | Set this to point elsewhere |
|---|---|---|
| `HMMAF_LLM_API_BASE` | `http://192.168.1.168:4000/v1` | Nebius / RunPod base URL (must end in `/v1`) |
| `HMMAF_LLM_MODEL` | `gemma-4-e4b-multimodal` | the model name the provider serves |
| `HMMAF_LLM_API_KEY` | `local-execution-key` | your provider API key |

Example:

```bash
export HMMAF_LLM_API_BASE="https://api.studio.nebius.com/v1"   # or your RunPod endpoint /v1
export HMMAF_LLM_MODEL="<model-id-served-there>"
export HMMAF_LLM_API_KEY="<your-key>"
```

> **RunPod note:** Serverless URLs look like
> `https://api.runpod.ai/v2/<id>/openai/v1` — double-check the exact path so it
> still ends in `/v1`.

### Caveats — pick a model that supports the features this app uses

The *transport* is fully interchangeable; what matters is whether the **model you
choose supports the features HMMAF exercises**:

1. **Multimodal / vision is required.** The app sends images as OpenAI
   `image_url` content blocks. Your target model **must** be vision-capable
   (e.g. a Gemma-3 / Qwen-VL / Llama-Vision-class model). A text-only model will
   error or ignore the image, breaking the "what do you see?" flow. This is the
   #1 thing to verify.
2. **Tool / function calling is required for the agentic loop.** Action
   detection depends on OpenAI-style `tool_calls`. If the served model or
   endpoint doesn't support tool calling, the action silently falls back to
   `RESPOND` — the app still talks, but the PRA loop's tool actions (memory
   updates, etc.) won't fire.
3. **The `reasoning` field is nonstandard.** It's read defensively with a
   fallback to `message.content`, so this won't break — you just won't get
   separated reasoning traces unless the provider emits them.

### Verify a new endpoint quickly

```bash
curl -s "$HMMAF_LLM_API_BASE/models" -H "Authorization: Bearer $HMMAF_LLM_API_KEY"
```

Expect a JSON list of models that includes the one you set in `HMMAF_LLM_MODEL`.

**Bottom line:** endpoint interchangeability is excellent (env-var swap, no code
edits); model interchangeability is conditional on choosing a **vision-capable,
tool-calling-capable** hosted model.

---

## 3. Prerequisites

- Python 3.11 (already present at `/usr/local/bin/python3.11`)
- A microphone and webcam reachable by the browser
- The LiteLLM gateway at `http://192.168.1.168:4000/v1` reachable from this host
  (or a different OpenAI-compatible endpoint configured per §2)
- `redis-server` available on PATH (already installed natively — see §8)

---

## 4. First-time setup (already done in this workspace)

```bash
cd /llm_models_python_code_src/HMMAF_Live_Test
python3.11 -m venv venv
./venv/bin/pip install -r requirements.txt
```

The venv contains every dependency. You should not have to re-run this.

---

## 5. Start the framework

```bash
cd /llm_models_python_code_src/HMMAF_Live_Test

# Start Redis (idempotent — does nothing if already up)
./start_redis.sh

# Launch the user-facing Gradio app (the app boots the whole framework
# in a background thread, then opens the UI on http://localhost:7860)
PYTHONPATH=. ./venv/bin/python src/ui/app.py
```

Open your browser at **http://localhost:7860**.

The first request triggers lazy model loads (YOLO26n weights, faster-whisper
`small`); subsequent requests are fast.

---

## 6. Using the UI

1. **Show it something.** Either click "Webcam" to use your camera, or upload
   an image. YOLO26n runs object detection and overlays bounding boxes.
2. **Talk to it.** Click the microphone, speak (e.g. "what do you see?"),
   stop recording. faster-whisper transcribes your speech.
3. **Or type.** Use the text field instead of the mic.
4. **Send.** The system runs YOLO on the current frame, sends the image plus
   your speech/text plus the structured YOLO detections to Gemma 4
   (multimodal), and synthesizes the response with SuperTonic2/edge-tts.
5. **Listen / read.** The annotated frame, the conversation transcript, and
   the spoken reply all appear in the panel.

### Example acceptance prompts
- _"What am I holding right now?"_ → expect an acknowledgement that mentions
  the dominant detected object class.
- _"Remember that this is my coffee mug."_ → expect the system to acknowledge
  and (via the background PRA loop) update the World Model.
- _"How many people do you see?"_ → expect a count consistent with the YOLO
  overlay.

---

## 7. Running the tests / benchmarks

All tests now hit **real** Redis (no fakeredis):

```bash
cd /llm_models_python_code_src/HMMAF_Live_Test

# 6 quick component checks (Redis, LLM, YOLO, TTS, STT)
PYTHONPATH=. ./venv/bin/python scripts/smoke_test.py

# Unit + integration tests (guardrail, runaway-reasoning interception)
PYTHONPATH=. ./venv/bin/pytest tests/ -v

# Live reasoning loop against the LiteLLM endpoint (real tool calls)
PYTHONPATH=. ./venv/bin/python tests/integration_reasoning_loop.py

# Framework-overhead benchmark (matches the original workspace's methodology)
PYTHONPATH=. ./venv/bin/python scripts/benchmark_latency.py 10 --mode=overhead

# End-to-end benchmark against the real LLM (LLM latency is the bottleneck)
PYTHONPATH=. ./venv/bin/python scripts/benchmark_latency.py 5 --mode=live
```

Latency reports are written to:
- `tests/latency_benchmark_report.overhead.txt` — framework overhead only
- `tests/latency_benchmark_report.live.txt` — full end-to-end with Gemma 4
- `tests/latency_benchmark_report.txt` — mirror of the last-run mode

The **overhead** benchmark PASSes the 1.5s PRD budget (sub-ms framework
overhead). The **live** benchmark depends on how fast your LLM gateway is.
See `tests/TEST_RESULTS.md` for the latest captured numbers and a full
discussion.

---

## 8. Native Redis — install / uninstall instructions

We are intentionally **not** using Docker because Docker Desktop's WSL
integration is not enabled for this distro. We use the native
`redis-server` package that is already installed on the system.

### Confirm it is up

```bash
redis-cli -p 6379 ping        # → PONG
```

### Start / stop the workspace-local instance

```bash
./start_redis.sh   # starts redis-server using ./redis.conf
./stop_redis.sh    # cleanly stops it; does not affect any system Redis
```

The workspace-local Redis writes its log to `./logs/redis.log` and its PID
file to `./redis_data/redis.pid`. It does **not** persist any data
(`appendonly no`, `save ""`), so dropping the workspace deletes everything.

### Cleanly uninstall Redis when you're done testing

```bash
# 1) Stop the workspace-local daemon first
./stop_redis.sh

# 2) If you also want to remove the OS-level packages:
sudo systemctl stop redis-server 2>/dev/null || true
sudo systemctl disable redis-server 2>/dev/null || true
sudo apt-get -y purge redis-server redis-tools redis-stack-server
sudo apt-get -y autoremove
sudo rm -rf /etc/redis /var/lib/redis /var/log/redis

# 3) Confirm it's gone
which redis-server redis-cli   # should print nothing
```

Skip step 2 if you only want to stop using Redis in this workspace but
keep it available system-wide.

---

## 9. Stopping everything

```bash
# Stop the Gradio UI: Ctrl-C in the terminal running src/ui/app.py
# Stop Redis:
./stop_redis.sh
```

---

## 10. Human Quality Acceptance Test plan

| # | Acceptance check | How to verify |
|---|---|---|
| 1 | UI loads | Open http://localhost:7860 and confirm the page renders. |
| 2 | YOLO sees the world | Upload `bus.jpg` (or use webcam) → expect green bounding boxes labelled `person`, `bus`. |
| 3 | STT works | Hold mic, say "Hello system, can you hear me?", release. The conversation pane should echo a near-verbatim transcription. |
| 4 | TTS works | After sending any message, the "Spoken response" player should auto-play audio. |
| 5 | Multimodal grounding | Show your face or any object, ask "what do you see?", and expect the model to mention the YOLO-detected class. |
| 6 | Memory roundtrip | Say "Please remember that this is my notebook." then on a new turn say "What did I just show you?" — expect retrieval. |
| 7 | PRA latency budget | Run `python scripts/benchmark_latency.py 5` and confirm `Status: PASS`. |
| 8 | Guardrail recursion | Run `pytest tests/test_guardrail.py -v` and confirm 3/3 pass. |
| 9 | Runaway interception | Run `pytest tests/test_guardrail_integration.py -v` and confirm pass. |
| 10 | Clean shutdown | Ctrl-C the UI, run `./stop_redis.sh`, confirm `redis-cli ping` errors. |

---

## 11. Troubleshooting

- **Webcam not appearing in Gradio**: the browser will ask for camera permission. If you're connecting from a remote host, use `http://<host>:7860` and allow permission for that origin.
- **LLM timeouts**: confirm `curl -s http://192.168.1.168:4000/v1/models` returns JSON. If not, the gateway is down or unreachable. (For a non-default endpoint, curl `$HMMAF_LLM_API_BASE/models` — see §2.)
- **YOLO weights download fails**: Ultralytics auto-downloads `yolo26n.pt` from its release page on first use. If you're offline, place the file manually at the workspace root.
- **TTS silent**: the worker tries SuperTonic2 first; if its ONNX bundle isn't downloadable, it falls back to edge-tts (which needs outbound HTTPS to Microsoft's edge speech endpoint).
- **Latency benchmark FAIL**: most often this is the LLM endpoint queueing under load. Re-run with `python scripts/benchmark_latency.py 10` and check the per-sample variance.
- **Swapped model ignores images or never calls tools**: the replacement model likely isn't vision- or tool-calling-capable. See the caveats in §2.
