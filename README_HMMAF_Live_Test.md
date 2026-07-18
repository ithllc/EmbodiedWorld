# HMMAF — Live Test Workspace

Production-grade, end-to-end live implementation of the Heterogeneous
Multi-Modal Agentic Framework (HMMAF) for human acceptance testing.

- **Reasoning**: Gemma 4 multimodal via LiteLLM (`http://192.168.1.168:4000/v1`, model `gemma-4-e4b-multimodal`)
- **Perception**: Ultralytics **YOLO26n.pt** on CPU
- **Expression**: **SuperTonic2** TTS (ONNX) with edge-tts fallback
- **Speech input**: faster-whisper (CPU, `small` model)
- **Communication**: native Redis on localhost:6379 (Docker not required)
- **Interface**: Gradio web UI (webcam, mic, chat, autoplay TTS)

## Environment setup

The virtual environment is **not** committed to the repo. Build your own
with the pinned dependencies (Python **3.11**, see `.python-version`):

```bash
cd HMMAF_Live_Test
python3.11 -m venv venv
source venv/bin/activate

# Option A: curated top-level dependencies
pip install -r requirements.txt

# Option B: exact, fully-pinned lockfile (reproduces the tested env)
pip install -r requirements-lock.txt
```

> Note: model weights (`yolo26n.pt`), the SuperTonic2 cache (`models_cache/`),
> and other large artifacts are downloaded on first run and are gitignored.

## Quickstart

```bash
cd HMMAF_Live_Test
./start_redis.sh
PYTHONPATH=. ./venv/bin/python src/ui/app.py
# open http://localhost:7860
```

See [`docs/HOW_TO_USE.md`](docs/HOW_TO_USE.md) for the full guide,
acceptance test plan, and clean-uninstall instructions.

## Tests / benchmarks

```bash
PYTHONPATH=. ./venv/bin/pytest tests/ -v
PYTHONPATH=. ./venv/bin/python tests/integration_reasoning_loop.py
PYTHONPATH=. ./venv/bin/python scripts/benchmark_latency.py 5
```
