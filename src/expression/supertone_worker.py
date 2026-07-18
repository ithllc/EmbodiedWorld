"""Expression layer: SuperTonic2-style TTS + faster-whisper STT.

TTS strategy:
    1. Try to use Supertonic (supertone-inc/supertonic) ONNX models. Supertonic
       is a lightning-fast on-device multilingual TTS by Supertone Inc.
    2. If the ONNX model bundle isn't available (network restricted or weights
       not yet downloaded), transparently fall back to edge-tts so the
       framework keeps working end-to-end.

STT strategy:
    faster-whisper (CTranslate2-optimized Whisper) running CPU 'small' model.
"""
import asyncio
import logging
import os
import threading
import time
import wave
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import soundfile as sf

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger('HMMAF.Expression')


# ---------------------------------------------------------------------------
# Supertonic ONNX TTS engine
# ---------------------------------------------------------------------------

class SupertonicTTSEngine:
    """Wraps the Supertonic 2 ONNX TTS for HMMAF.

    Uses the official `supertonic` Python package (pip-installable). The
    model name `supertonic-2` matches the user's request and supports 5
    languages on-device via ONNX Runtime (CPU).

    Weights auto-download from HuggingFace on first use.
    """

    DEFAULT_MODEL = os.environ.get('SUPERTONIC_MODEL', 'supertonic-2')
    DEFAULT_VOICE_STYLE = os.environ.get('SUPERTONIC_VOICE', 'M1')

    def __init__(self, model: str = DEFAULT_MODEL, voice_style: str = DEFAULT_VOICE_STYLE):
        self.model_name = model
        self.voice_style_name = voice_style
        self._tts = None
        self._style = None
        self._available: Optional[bool] = None
        self._init_lock = threading.Lock()
        self.sample_rate: int = 44100

    def _try_init(self) -> bool:
        try:
            import supertonic  # type: ignore
            self._tts = supertonic.TTS(model=self.model_name, auto_download=True)
            try:
                self._style = self._tts.get_voice_style(self.voice_style_name)
            except Exception:
                # Pick the first available voice if the configured one is missing
                names = list(self._tts.voice_style_names)
                logger.info(f"Voice style '{self.voice_style_name}' not found; using '{names[0]}'.")
                self.voice_style_name = names[0]
                self._style = self._tts.get_voice_style(names[0])
            self.sample_rate = self._tts.sample_rate
            logger.info(f"Supertonic {self.model_name} loaded (voice={self.voice_style_name}, sr={self.sample_rate}).")
            return True
        except Exception as e:
            logger.warning(f"Supertonic init failed ({e}); will fall back to edge-tts.")
            return False

    def is_available(self) -> bool:
        with self._init_lock:
            if self._available is None:
                self._available = self._try_init()
        return self._available

    def synthesize(self, text: str, out_path: str, voice: Optional[str] = None, lang: str = 'en') -> bool:
        if not self.is_available():
            return False
        try:
            style = self._style
            if voice and voice != self.voice_style_name:
                try:
                    style = self._tts.get_voice_style(voice)
                except Exception:
                    pass
            wav, dur = self._tts.synthesize(text=text, voice_style=style, lang=lang)
            # supertonic returns shape (1, N); soundfile wants 1-D or (N, channels)
            audio = np.asarray(wav)
            if audio.ndim == 2 and audio.shape[0] == 1:
                audio = audio[0]
            sf.write(out_path, audio, self.sample_rate)
            logger.info(f"Supertonic synth: {dur[0] if hasattr(dur, '__len__') else dur:.2f}s -> {out_path}")
            return True
        except Exception as e:
            logger.warning(f"Supertonic synthesize failed ({e}); falling back to edge-tts.")
            return False


# ---------------------------------------------------------------------------
# edge-tts fallback
# ---------------------------------------------------------------------------

async def _edge_tts_synthesize(text: str, out_path_mp3: str, voice: str = 'en-US-GuyNeural'):
    import edge_tts
    communicate = edge_tts.Communicate(text, voice)
    await communicate.save(out_path_mp3)


# ---------------------------------------------------------------------------
# Async worker (subscribes to control_plane TEXT_RESPONSE directives)
# ---------------------------------------------------------------------------

class SupertoneWorker:
    """SuperTonic2 expression worker.

    - Tries Supertonic ONNX first, falls back to edge-tts.
    - Subscribes to the Redis 'control_plane' channel and synthesizes any
      directive of the form `TEXT_RESPONSE | <text>` to an MP3/WAV file.
    """

    def __init__(self, redis_client=None, output_dir: str = "./output_audio",
                 supertonic_voice: str = 'M1', edge_voice: str = 'en-US-GuyNeural'):
        self.redis = redis_client
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        self.queue: asyncio.Queue = asyncio.Queue()
        self.task: Optional[asyncio.Task] = None
        self.subscribe_task: Optional[asyncio.Task] = None
        self.running = False
        self.tts = SupertonicTTSEngine(voice_style=supertonic_voice)
        self.supertonic_voice = supertonic_voice
        self.edge_voice = edge_voice
        self._last_path: Optional[str] = None

    @property
    def last_audio_path(self) -> Optional[str]:
        return self._last_path

    async def start(self):
        logger.info('Starting SupertoneWorker ...')
        self.running = True
        self.task = asyncio.create_task(self._worker_loop())
        if self.redis is not None:
            self.subscribe_task = asyncio.create_task(self._subscribe_loop())

    async def stop(self):
        logger.info('Stopping SupertoneWorker ...')
        self.running = False
        for t in (self.task, self.subscribe_task):
            if t is not None:
                t.cancel()
                try:
                    await t
                except asyncio.CancelledError:
                    pass

    async def enqueue_tts(self, text: str, metadata: Optional[Dict[str, Any]] = None):
        logger.info(f'Enqueuing TTS: {text[:80]}')
        await self.queue.put({'text': text, 'metadata': metadata or {}})

    async def _subscribe_loop(self):
        ps = await self.redis.subscribe('control_plane')
        try:
            while self.running:
                msg = ps.get_message(ignore_subscribe_messages=True)
                if msg and isinstance(msg.get('data'), str) and msg['data'].startswith('TEXT_RESPONSE | '):
                    text = msg['data'][len('TEXT_RESPONSE | '):]
                    await self.enqueue_tts(text)
                await asyncio.sleep(0.01)
        except asyncio.CancelledError:
            pass

    async def _worker_loop(self):
        while self.running:
            try:
                item = await asyncio.wait_for(self.queue.get(), timeout=0.5)
            except asyncio.TimeoutError:
                continue
            except asyncio.CancelledError:
                break
            try:
                await self._synthesize(item['text'], item['metadata'])
                self.queue.task_done()
            except Exception as e:
                logger.error(f'Error in SupertoneWorker: {e}')

    async def _synthesize(self, text: str, metadata: Dict[str, Any]):
        ts = int(time.time() * 1000)
        wav_path = os.path.join(self.output_dir, f"tts_{ts}.wav")
        mp3_path = os.path.join(self.output_dir, f"tts_{ts}.mp3")
        loop = asyncio.get_running_loop()

        # Try Supertonic first
        if self.tts.is_available():
            sup_voice = metadata.get('supertonic_voice') or self.supertonic_voice
            ok = await loop.run_in_executor(
                None, lambda: self.tts.synthesize(text, wav_path, voice=sup_voice)
            )
            if ok:
                self._last_path = wav_path
                return

        # Fallback: edge-tts
        try:
            edge_voice = metadata.get('edge_voice') or self.edge_voice
            await _edge_tts_synthesize(text, mp3_path, voice=edge_voice)
            self._last_path = mp3_path
            logger.info(f"edge-tts fallback -> {mp3_path}")
        except Exception as e:
            logger.error(f"All TTS engines failed: {e}")
            self._last_path = None


# ---------------------------------------------------------------------------
# faster-whisper STT
# ---------------------------------------------------------------------------

class WhisperSTT:
    """faster-whisper STT (CPU). Loads the 'small' model lazily on first use."""

    def __init__(self, model_size: str = 'small', device: str = 'cpu', compute_type: str = 'int8'):
        self.model_size = model_size
        self.device = device
        self.compute_type = compute_type
        self._model = None
        self._lock = threading.Lock()

    def _ensure(self):
        if self._model is None:
            with self._lock:
                if self._model is None:
                    from faster_whisper import WhisperModel
                    logger.info(f"Loading faster-whisper {self.model_size} on {self.device} ({self.compute_type}) ...")
                    self._model = WhisperModel(self.model_size, device=self.device, compute_type=self.compute_type)
                    logger.info("faster-whisper loaded.")

    def transcribe_sync(self, audio_path: str, language: str = 'en') -> str:
        self._ensure()
        segments, info = self._model.transcribe(audio_path, language=language, beam_size=1)
        text = "".join(seg.text for seg in segments).strip()
        logger.info(f"STT: '{text}'  (lang={info.language}, conf={info.language_probability:.2f})")
        return text

    async def transcribe(self, audio_path: str, language: str = 'en') -> str:
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, self.transcribe_sync, audio_path, language)


async def _smoke():
    # Standalone sanity check
    w = SupertoneWorker(output_dir="./output_audio")
    await w.start()
    await w.enqueue_tts("Hello from HMMAF live test.")
    await asyncio.sleep(5)
    await w.stop()
    print("Last audio:", w.last_audio_path)


if __name__ == '__main__':
    asyncio.run(_smoke())
