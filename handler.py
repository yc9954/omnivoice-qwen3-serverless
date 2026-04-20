"""
RunPod Serverless handler for Qwen3-TTS streaming voice cloning.

Input (job.input):
    text:           str    — text to synthesize
    ref_audio_b64:  str    — base64-encoded reference WAV bytes
    ref_text:       str    — reference audio transcript
    language:       str    — "Korean" (default) / "English" / ...
    chunk_size:     int    — streaming chunk size (default 12)
    temperature:    float  — sampling temperature (default 0.9)
    top_k:          int    — top-k sampling (default 50)
    repetition_penalty: float — (default 1.05)

Output (yielded chunks):
    first chunk:  {"sr": 24000, "text_chars": N, "type": "meta"}
    audio chunks: {"pcm_b64": <base64 int16 little-endian>, "type": "audio", "idx": i}
    final chunk:  {"type": "done", "chunks": N, "duration": secs, "rtf": float}

Env:
    MODEL_PATH:     default "/workspace/models/Qwen3-TTS-12Hz-1.7B-Base"
    WARMUP_ON_START: "1" to run warmup at init (recommended, costs ~6s on first boot)
"""
import base64
import io
import os
import time
import tempfile
from typing import Generator

import numpy as np
import soundfile as sf
import runpod

MODEL_PATH = os.environ.get("MODEL_PATH", "/workspace/models/Qwen3-TTS-12Hz-1.7B-Base")
WARMUP_ON_START = os.environ.get("WARMUP_ON_START", "1") == "1"

print(f"[init] loading faster-qwen3-tts from {MODEL_PATH}", flush=True)
_t0 = time.time()
import traceback
import torch as _torch
try:
    from faster_qwen3_tts import FasterQwen3TTS
    _dtype = _torch.bfloat16 if _torch.cuda.is_available() else _torch.float32
    _device = "cuda:0" if _torch.cuda.is_available() else "cpu"
    print(f"[init] device={_device}, dtype={_dtype}", flush=True)
    _MODEL = FasterQwen3TTS.from_pretrained(
        MODEL_PATH, device_map=_device, dtype=_dtype, attn_implementation="sdpa",
    )
    print(f"[init] model loaded in {time.time()-_t0:.1f}s", flush=True)
except Exception as _e:
    print(f"[init] FATAL: model load failed: {_e}", flush=True)
    traceback.print_exc()
    raise

if WARMUP_ON_START:
    print("[init] warmup (CUDA graph capture + kernel compilation)...")
    _t0 = time.time()
    _warmup_ref = "/workspace/ref_warmup.wav"
    if not os.path.exists(_warmup_ref):
        sr = 24000
        silent = np.zeros(int(sr * 4.0), dtype=np.float32)
        silent[::200] = 0.01
        sf.write(_warmup_ref, silent, sr)
    try:
        for _ in _MODEL.generate_voice_clone_streaming(
            text="테스트", language="Korean", ref_audio=_warmup_ref,
            ref_text="테스트", chunk_size=12,
        ):
            pass
        print(f"[init] warmup done in {time.time()-_t0:.1f}s")
    except Exception as e:
        print(f"[init] warmup skipped: {e}")


def _float32_to_int16_b64(wav: np.ndarray) -> str:
    """Convert float32 PCM [-1, 1] to 16-bit PCM little-endian, base64 encoded."""
    wav = np.clip(wav, -1.0, 1.0)
    pcm16 = (wav * 32767.0).astype(np.int16)
    return base64.b64encode(pcm16.tobytes()).decode("ascii")


def handler(job) -> Generator[dict, None, None]:
    job_input = job.get("input") or {}
    text = job_input.get("text", "").strip()
    ref_audio_b64 = job_input.get("ref_audio_b64")
    ref_text = job_input.get("ref_text", "")
    language = job_input.get("language", "Korean")
    chunk_size = int(job_input.get("chunk_size", 12))
    temperature = float(job_input.get("temperature", 0.9))
    top_k = int(job_input.get("top_k", 50))
    repetition_penalty = float(job_input.get("repetition_penalty", 1.05))

    if not text:
        yield {"type": "error", "message": "empty text"}
        return
    if not ref_audio_b64:
        yield {"type": "error", "message": "ref_audio_b64 required"}
        return

    # dump ref audio to tempfile (faster-qwen3-tts expects a path)
    try:
        audio_bytes = base64.b64decode(ref_audio_b64)
    except Exception as e:
        yield {"type": "error", "message": f"invalid base64: {e}"}
        return

    tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    tmp.write(audio_bytes)
    tmp.close()

    try:
        t0 = time.time()
        ttfa = None
        n_chunks = 0
        total_samples = 0
        sr = 24000
        yielded_meta = False

        for idx, (wav_chunk, chunk_sr, _info) in enumerate(
            _MODEL.generate_voice_clone_streaming(
                text=text,
                language=language,
                ref_audio=tmp.name,
                ref_text=ref_text,
                chunk_size=chunk_size,
                temperature=temperature,
                top_k=top_k,
                repetition_penalty=repetition_penalty,
            )
        ):
            if not yielded_meta:
                sr = chunk_sr
                yield {"type": "meta", "sr": sr, "text_chars": len(text)}
                yielded_meta = True

            if ttfa is None:
                ttfa = time.time() - t0

            n_chunks += 1
            total_samples += len(wav_chunk)
            yield {
                "type": "audio",
                "idx": idx,
                "pcm_b64": _float32_to_int16_b64(wav_chunk),
                "n_samples": len(wav_chunk),
            }

        elapsed = time.time() - t0
        duration = total_samples / sr if sr else 0.0
        rtf = elapsed / duration if duration > 0 else 0.0
        yield {
            "type": "done",
            "chunks": n_chunks,
            "duration": round(duration, 3),
            "ttfa_ms": round((ttfa or 0) * 1000, 1),
            "elapsed_sec": round(elapsed, 3),
            "rtf": round(rtf, 4),
        }
    except Exception as e:
        yield {"type": "error", "message": f"{type(e).__name__}: {e}"}
    finally:
        try:
            os.unlink(tmp.name)
        except OSError:
            pass


if __name__ == "__main__":
    runpod.serverless.start({
        "handler": handler,
        "return_aggregate_stream": True,
    })
