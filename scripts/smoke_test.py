"""Quick smoke test: verify all live components import and the basics work.

Run AFTER pip install completes:
    PYTHONPATH=. ./venv/bin/python scripts/smoke_test.py
"""
import asyncio
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def check(name, ok, detail=""):
    flag = "OK " if ok else "FAIL"
    print(f"  [{flag}] {name}  {detail}")
    return ok


async def main():
    overall_ok = True

    # 1) Redis
    try:
        from src.comm.redis_client import RedisClient
        client = RedisClient()
        await client.connect()
        await client.publish('smoke', 'hello')
        overall_ok &= check("RedisClient ping/publish", True)
        await client.close()
    except Exception as e:
        overall_ok &= check("RedisClient", False, str(e))

    # 2) LiteLLM gateway reachability
    try:
        import requests
        r = requests.get('http://192.168.1.168:4000/v1/models', timeout=10)
        models = [m['id'] for m in r.json().get('data', [])]
        overall_ok &= check("LLM gateway reachable", 'gemma-4-e4b-multimodal' in models, f"models={models[:3]}...")
    except Exception as e:
        overall_ok &= check("LLM gateway", False, str(e))

    # 3) GemmaEngine simple text reason
    try:
        from src.reasoning.gemma_engine import GemmaEngine
        engine = GemmaEngine()
        t0 = time.time()
        res = await engine.reason("Say the single word 'hello'.")
        dt = time.time() - t0
        overall_ok &= check("GemmaEngine.reason", bool(res.get('reasoning')), f"took {dt:.2f}s, action={res.get('action')}, text={(res.get('reasoning') or '')[:60]!r}")
    except Exception as e:
        overall_ok &= check("GemmaEngine.reason", False, str(e))

    # 4) YOLO26n single-frame detect on bus.jpg
    try:
        from src.perception.yolo_proxy import YOLOProxy
        proxy = YOLOProxy(redis_client=None, model_path='yolo26n.pt', source=None, device='cpu')
        await proxy.load()
        bus_path = ROOT / 'assets' / 'bus.jpg'
        if not bus_path.exists():
            # Fall back to upstream URL
            target = "https://ultralytics.com/images/bus.jpg"
        else:
            target = str(bus_path)
        t0 = time.time()
        event = await proxy.detect_once(target)
        dt = time.time() - t0
        labels = [d.label for d in event.detections]
        overall_ok &= check("YOLO26n detect_once", len(event.detections) > 0, f"took {dt:.2f}s, labels={labels[:5]}")
    except Exception as e:
        overall_ok &= check("YOLO26n", False, str(e))

    # 5) TTS (Supertonic preferred, edge-tts fallback)
    try:
        from src.expression.supertone_worker import SupertoneWorker
        w = SupertoneWorker(output_dir=str(ROOT / 'output_audio'))
        await w.start()
        await w.enqueue_tts("Smoke test. The framework is operational.")
        await asyncio.sleep(4)
        await w.stop()
        ok = w.last_audio_path is not None and Path(w.last_audio_path).exists()
        overall_ok &= check("TTS synthesis", ok, f"path={w.last_audio_path}")
    except Exception as e:
        overall_ok &= check("TTS", False, str(e))

    # 6) Whisper STT load
    try:
        from src.expression.supertone_worker import WhisperSTT
        stt = WhisperSTT()
        stt._ensure()
        overall_ok &= check("faster-whisper load", stt._model is not None)
    except Exception as e:
        overall_ok &= check("faster-whisper", False, str(e))

    print()
    print("OVERALL:", "PASS" if overall_ok else "FAIL")
    sys.exit(0 if overall_ok else 1)


if __name__ == '__main__':
    asyncio.run(main())
