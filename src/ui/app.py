"""HMMAF Live Test — Gradio multimodal UI.

The user can:
  • Speak into the microphone (faster-whisper STT)
  • Show the system things via webcam or by uploading an image (YOLO26n detection)
  • Hear the system's response via Supertonic (or edge-tts fallback) TTS
  • Read the conversation transcript

Architecture:
  - The Redis-backed PRA loop (Orchestrator + Guardrail + YOLOProxy publishing
    to observation_plane) runs in the background so the system is exercising
    its real production path even while the user interacts.
  - The chat path additionally calls the same components (YOLOProxy.detect_once
    for visual grounding, GemmaEngine.reason for multimodal LLM reasoning,
    SupertoneWorker for TTS) so every part of the framework is used live.
"""
from __future__ import annotations

import asyncio
import base64
import io
import json
import logging
import os
import sys
import threading
import time
from pathlib import Path
from typing import List, Optional, Tuple

import gradio as gr
import numpy as np
from PIL import Image, ImageDraw, ImageFont

# Ensure project root is importable
ROOT = Path(__file__).resolve().parent.parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.comm.redis_client import RedisClient  # noqa: E402
from src.expression.supertone_worker import SupertoneWorker, WhisperSTT  # noqa: E402
from src.memory.world_model import WorldModel  # noqa: E402
from src.orchestrator.core import Orchestrator  # noqa: E402
from src.perception.yolo_proxy import YOLOProxy  # noqa: E402
from src.reasoning.gemma_engine import GemmaEngine  # noqa: E402
from src.reasoning.mcp_tools import MCPTools  # noqa: E402

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger('HMMAF.UI')


# ---------------------------------------------------------------------------
# Background runtime: spins up the live framework once at startup
# ---------------------------------------------------------------------------

class HMMAFRuntime:
    """Owns the long-lived components and the asyncio loop they run on."""

    def __init__(self):
        self.loop: Optional[asyncio.AbstractEventLoop] = None
        self.thread: Optional[threading.Thread] = None
        self.ready = threading.Event()

        # Components
        self.redis_client: Optional[RedisClient] = None
        self.world_model: Optional[WorldModel] = None
        self.tools: Optional[MCPTools] = None
        self.engine: Optional[GemmaEngine] = None
        self.orchestrator: Optional[Orchestrator] = None
        self.yolo: Optional[YOLOProxy] = None
        self.tts: Optional[SupertoneWorker] = None
        self.stt: Optional[WhisperSTT] = None

    def start(self):
        if self.thread is not None:
            return
        self.thread = threading.Thread(target=self._run, daemon=True, name="HMMAF-Runtime")
        self.thread.start()
        self.ready.wait()

    def _run(self):
        self.loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self.loop)
        self.loop.run_until_complete(self._init_components())
        self.ready.set()
        self.loop.run_forever()

    async def _init_components(self):
        logger.info("Bootstrapping HMMAF runtime ...")
        self.redis_client = RedisClient()
        await self.redis_client.connect()

        self.world_model = WorldModel(redis_client=self.redis_client)
        self.tools = MCPTools(world_model=self.world_model)
        self.engine = GemmaEngine()

        self.orchestrator = Orchestrator(self.redis_client, self.engine, self.tools, self.world_model)
        await self.orchestrator.start()

        # Hook the 'speak' MCP tool back to our TTS worker
        self.tts = SupertoneWorker(redis_client=None, output_dir=str(ROOT / "output_audio"))
        await self.tts.start()
        self.tools.set_tts_callback(self.tts.enqueue_tts)

        self.stt = WhisperSTT(model_size=os.environ.get('HMMAF_WHISPER_SIZE', 'small'), device='cpu', compute_type='int8')

        # Load YOLO eagerly so the first UI click is fast
        self.yolo = YOLOProxy(redis_client=self.redis_client, model_path='yolo26n.pt', source=None, device='cpu')
        try:
            await self.yolo.load()
        except Exception as e:
            logger.error(f"YOLO failed to load: {e}")
            raise
        logger.info("HMMAF runtime ready.")

    # --- helpers callable from the Gradio threads ----------------------------

    def submit(self, coro):
        """Schedule a coroutine on the runtime loop and wait for the result."""
        fut = asyncio.run_coroutine_threadsafe(coro, self.loop)
        return fut.result()


RUNTIME = HMMAFRuntime()


# ---------------------------------------------------------------------------
# Visualization helpers
# ---------------------------------------------------------------------------

def _draw_detections(img: np.ndarray, detections: List[dict]) -> np.ndarray:
    if img is None:
        return img
    pil = Image.fromarray(img.astype(np.uint8)) if img.dtype != np.uint8 else Image.fromarray(img)
    draw = ImageDraw.Draw(pil)
    try:
        font = ImageFont.load_default()
    except Exception:
        font = None
    for d in detections:
        x1, y1, x2, y2 = d['bbox']
        draw.rectangle([x1, y1, x2, y2], outline=(0, 255, 0), width=3)
        label = f"{d['label']} {d['confidence']:.2f}"
        draw.text((x1 + 4, max(0, y1 - 12)), label, fill=(0, 255, 0), font=font)
    return np.array(pil)


def _image_to_data_url(img: np.ndarray) -> str:
    pil = Image.fromarray(img.astype(np.uint8))
    buf = io.BytesIO()
    pil.save(buf, format='JPEG', quality=80)
    b64 = base64.b64encode(buf.getvalue()).decode('ascii')
    return f"data:image/jpeg;base64,{b64}"


# ---------------------------------------------------------------------------
# Core chat handler
# ---------------------------------------------------------------------------

async def _handle_turn(
    user_text: str,
    image: Optional[np.ndarray],
    chat_history: List[Tuple[str, str]],
):
    """Run one full PRA cycle and return updated UI state."""
    detections: List[dict] = []
    annotated = image

    # 1) Perception: run YOLO on the current frame, if present
    if image is not None:
        try:
            event = await RUNTIME.yolo.detect_once(image)
            detections = [d.to_dict() for d in event.detections]
            annotated = _draw_detections(image, detections)
            # Publish the detection on the observation plane so the bg orchestrator
            # also sees it (keeps memory consistent with PRA loop).
            await RUNTIME.redis_client.publish('observation_plane', json.dumps(event.to_dict()))
        except Exception as e:
            logger.error(f"YOLO detection error: {e}")

    # 2) Reasoning: build multimodal prompt
    detections_summary = ", ".join(
        f"{d['label']} ({d['confidence']:.2f})" for d in detections
    ) or "no objects detected"

    user_prompt = (
        f"The user just said: \"{user_text or '(no speech)'}\".\n"
        f"YOLO26 currently detects: {detections_summary}.\n"
        "Reply naturally in 1-3 short sentences. Acknowledge what you can see "
        "if the user is asking about it."
    )

    images_for_llm = [_image_to_data_url(image)] if image is not None else None

    try:
        result = await RUNTIME.engine.reason(
            prompt=user_prompt,
            tools=None,                # direct conversational turn, no tool routing
            images=images_for_llm,
        )
        assistant_text = (result.get('reasoning') or '').strip() or "(no response from model)"
    except Exception as e:
        logger.error(f"GemmaEngine.reason failed: {e}")
        assistant_text = f"[Engine error: {e}]"

    # 3) Expression: TTS the response
    audio_path: Optional[str] = None
    try:
        await RUNTIME.tts.enqueue_tts(assistant_text)
        # wait briefly for the file to materialize
        for _ in range(40):  # up to ~4s
            await asyncio.sleep(0.1)
            p = RUNTIME.tts.last_audio_path
            if p and os.path.exists(p):
                audio_path = p
                break
    except Exception as e:
        logger.error(f"TTS error: {e}")

    chat_history = list(chat_history) + [
        {"role": "user", "content": user_text or "(spoken via mic)"},
        {"role": "assistant", "content": assistant_text},
    ]
    detections_md = "\n".join(
        f"- **{d['label']}** ({d['confidence']:.2f}) bbox={d['bbox']}" for d in detections
    ) or "_(none)_"
    return chat_history, annotated, audio_path, detections_md


def handle_turn_sync(audio_path, typed_text, image, chat_history):
    """Synchronous wrapper invoked by Gradio."""
    chat_history = chat_history or []
    user_text = (typed_text or "").strip()
    if audio_path:
        try:
            transcribed = RUNTIME.stt.transcribe_sync(audio_path).strip()
            if transcribed:
                user_text = (user_text + " " + transcribed).strip() if user_text else transcribed
        except Exception as e:
            logger.error(f"STT failed: {e}")
    if not user_text and image is None:
        return chat_history, image, None, "_(provide audio, text, or an image)_"
    return RUNTIME.submit(_handle_turn(user_text, image, chat_history))


# ---------------------------------------------------------------------------
# Gradio layout
# ---------------------------------------------------------------------------

def build_ui() -> gr.Blocks:
    with gr.Blocks(title="HMMAF Live Test", theme=gr.themes.Soft()) as demo:
        gr.Markdown(
            """
            # HMMAF — Heterogeneous Multi-Modal Agentic Framework (Live)

            Talk to the system or show it something. It will see (YOLO26n), reason (Gemma 4 multimodal),
            remember (World Model + Redis), and respond aloud (SuperTonic2 / edge-tts).
            """
        )
        with gr.Row():
            with gr.Column(scale=1):
                image_in = gr.Image(
                    label="Webcam or uploaded image (this is what YOLO + Gemma see)",
                    sources=["webcam", "upload"],
                    type="numpy",
                    height=360,
                )
                audio_in = gr.Audio(
                    label="Hold to talk (or upload a clip)",
                    sources=["microphone", "upload"],
                    type="filepath",
                )
                typed = gr.Textbox(label="Or type a message", placeholder="What am I holding?")
                send = gr.Button("Send to HMMAF", variant="primary")
                clear = gr.Button("Clear conversation")
            with gr.Column(scale=1):
                chat = gr.Chatbot(label="Conversation", height=420, type="messages")
                tts_out = gr.Audio(label="Spoken response", autoplay=True, type="filepath")
                detections_md = gr.Markdown(label="Latest detections")

        state = gr.State([])

        send.click(
            handle_turn_sync,
            inputs=[audio_in, typed, image_in, state],
            outputs=[chat, image_in, tts_out, detections_md],
        ).then(lambda h: h, inputs=chat, outputs=state).then(
            lambda: ("", None), outputs=[typed, audio_in]
        )

        def _clear():
            return [], [], None, "_(cleared)_"
        clear.click(_clear, outputs=[chat, state, tts_out, detections_md])

    return demo


def main():
    RUNTIME.start()
    ui = build_ui()
    ui.queue(default_concurrency_limit=1).launch(
        server_name=os.environ.get('HMMAF_UI_HOST', '0.0.0.0'),
        server_port=int(os.environ.get('HMMAF_UI_PORT', '7860')),
        share=False,
        show_error=True,
    )


if __name__ == '__main__':
    main()
