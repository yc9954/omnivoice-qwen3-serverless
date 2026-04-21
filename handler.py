"""
Qwen3-TTS streaming handler.

Dual-mode: the same image serves either a RunPod Queue endpoint OR a
RunPod Load Balancer endpoint, selected by SERVER_MODE.

  SERVER_MODE=queue (default) → runpod.serverless.start(generator handler)
  SERVER_MODE=http             → FastAPI on $PORT (default 80), streams NDJSON

The queue path goes through RunPod's poll-based stream API (200-3000ms of
platform-side buffering). The http path is direct chunked HTTP and sees
the worker's true TTFA (~500ms).
"""
import sys

_BOOT_LOG = "/tmp/handler_boot.log"
def _boot(msg: str):
    line = f"[boot] {msg}"
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.write(line + "\n"); stream.flush()
        except Exception:
            pass
    try:
        with open(_BOOT_LOG, "a") as _f:
            _f.write(line + "\n")
    except Exception:
        pass

_boot("python started")

try:
    import base64
    import io
    import json
    import os
    import time
    import tempfile
    import traceback
    from typing import Generator, Optional
    _boot("stdlib imports ok")

    import numpy as np
    _boot(f"numpy {np.__version__} ok")

    import soundfile as sf
    _boot(f"soundfile {sf.__version__} ok")

    import torch as _torch
    _boot(f"torch {_torch.__version__} cuda_avail={_torch.cuda.is_available()}")
    if _torch.cuda.is_available():
        _boot(f"cuda device: {_torch.cuda.get_device_name(0)}")

    from faster_qwen3_tts import FasterQwen3TTS
    _boot("faster_qwen3_tts ok")
except Exception as _e:
    _boot(f"FATAL import error: {_e!r}")
    traceback.print_exc(file=sys.stderr)
    raise


MODEL_PATH = os.environ.get("MODEL_PATH", "/workspace/models/Qwen3-TTS-12Hz-1.7B-Base")
WARMUP_ON_START = os.environ.get("WARMUP_ON_START", "1") == "1"
SERVER_MODE = os.environ.get("SERVER_MODE", "queue").lower()

print(f"[init] SERVER_MODE={SERVER_MODE}", flush=True)
print(f"[init] loading faster-qwen3-tts from {MODEL_PATH}", flush=True)
_t0 = time.time()
try:
    _dtype = _torch.bfloat16 if _torch.cuda.is_available() else _torch.float32
    _device = "cuda:0" if _torch.cuda.is_available() else "cpu"
    print(f"[init] device={_device}, dtype={_dtype}", flush=True)
    _MODEL = FasterQwen3TTS.from_pretrained(
        MODEL_PATH, device=_device, dtype=_dtype, attn_implementation="sdpa",
    )
    print(f"[init] model loaded in {time.time()-_t0:.1f}s", flush=True)
except Exception as _e:
    print(f"[init] FATAL: model load failed: {_e!r}", flush=True)
    traceback.print_exc(file=sys.stderr)
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
    wav = np.clip(wav, -1.0, 1.0)
    pcm16 = (wav * 32767.0).astype(np.int16)
    return base64.b64encode(pcm16.tobytes()).decode("ascii")


# ============================================================
# Core generation — shared by both modes
# ============================================================
import re as _re

_SENTENCE_SPLIT = _re.compile(r"(?<=[.!?。！？…])\s+")


def _split_sentences(text: str) -> list:
    """Split on sentence-ending punctuation, keeping it with the preceding
    sentence. Collapses to a single-entry list for short input."""
    parts = [p.strip() for p in _SENTENCE_SPLIT.split(text.strip()) if p.strip()]
    return parts or [text.strip()]


def synthesize_stream(
    text: str,
    ref_audio_path: str,
    ref_text: str = "",
    language: str = "Korean",
    chunk_size: int = 12,
    temperature: float = 0.9,
    top_k: int = 50,
    repetition_penalty: float = 1.05,
    instruct: Optional[str] = None,
    inter_sentence_silence_ms: int = 250,
) -> Generator[dict, None, None]:
    """Yield event dicts: meta → audio × N → done (or error).

    Splits multi-sentence input and synthesizes each sentence separately,
    inserting ``inter_sentence_silence_ms`` of silence between sentences so
    the output has natural sentence-level pauses. Single-sentence input is
    unaffected (one streaming call, no silence)."""
    sentences = _split_sentences(text)
    t0 = time.time()
    ttfa = None
    n_chunks = 0
    total_samples = 0
    sr = 24000
    yielded_meta = False
    try:
        base_kwargs = dict(
            language=language,
            ref_audio=ref_audio_path, ref_text=ref_text,
            chunk_size=chunk_size,
            temperature=temperature, top_k=top_k,
            repetition_penalty=repetition_penalty,
        )
        if instruct:
            base_kwargs["instruct"] = instruct

        for sent_idx, sentence in enumerate(sentences):
            for (wav_chunk, chunk_sr, _info) in _MODEL.generate_voice_clone_streaming(
                text=sentence, **base_kwargs,
            ):
                if not yielded_meta:
                    sr = chunk_sr
                    yield {"type": "meta", "sr": sr, "text_chars": len(text),
                           "sentences": len(sentences)}
                    yielded_meta = True
                if ttfa is None:
                    ttfa = time.time() - t0
                total_samples += len(wav_chunk)
                yield {
                    "type": "audio",
                    "idx": n_chunks,
                    "pcm_b64": _float32_to_int16_b64(wav_chunk),
                    "n_samples": len(wav_chunk),
                }
                n_chunks += 1

            # Silence between sentences, not after the last one
            if sent_idx < len(sentences) - 1 and inter_sentence_silence_ms > 0:
                silence = np.zeros(int(sr * inter_sentence_silence_ms / 1000), dtype=np.float32)
                total_samples += len(silence)
                yield {
                    "type": "audio",
                    "idx": n_chunks,
                    "pcm_b64": _float32_to_int16_b64(silence),
                    "n_samples": len(silence),
                    "silence": True,
                }
                n_chunks += 1

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
            "sentences": len(sentences),
        }
    except Exception as e:
        yield {"type": "error", "message": f"{type(e).__name__}: {e}"}


def _write_ref_tempfile(ref_audio_b64: str) -> str:
    audio_bytes = base64.b64decode(ref_audio_b64)
    tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    tmp.write(audio_bytes)
    tmp.close()
    return tmp.name


# ============================================================
# Mode: QUEUE (runpod.serverless generator handler)
# ============================================================
def handler(job) -> Generator[dict, None, None]:
    job_input = job.get("input") or {}
    text = job_input.get("text", "").strip()
    ref_audio_b64 = job_input.get("ref_audio_b64")
    if not text:
        yield {"type": "error", "message": "empty text"}
        return
    if not ref_audio_b64:
        yield {"type": "error", "message": "ref_audio_b64 required"}
        return
    try:
        ref_path = _write_ref_tempfile(ref_audio_b64)
    except Exception as e:
        yield {"type": "error", "message": f"invalid base64: {e}"}
        return

    try:
        yield from synthesize_stream(
            text=text,
            ref_audio_path=ref_path,
            ref_text=job_input.get("ref_text", ""),
            language=job_input.get("language", "Korean"),
            chunk_size=int(job_input.get("chunk_size", 12)),
            temperature=float(job_input.get("temperature", 0.9)),
            top_k=int(job_input.get("top_k", 50)),
            repetition_penalty=float(job_input.get("repetition_penalty", 1.05)),
            instruct=job_input.get("instruct") or None,
        )
    finally:
        try:
            os.unlink(ref_path)
        except OSError:
            pass


# ============================================================
# Mode: HTTP (FastAPI for Load Balancer endpoints)
# ============================================================
def _build_http_app():
    """Lazily build FastAPI app. Only imported when SERVER_MODE=http."""
    from fastapi import FastAPI, HTTPException
    from fastapi.responses import StreamingResponse
    from pydantic import BaseModel

    app = FastAPI(title="Qwen3-TTS streaming worker")

    class TTSRequest(BaseModel):
        text: str
        ref_audio_b64: str
        ref_text: str = ""
        language: str = "Korean"
        chunk_size: int = 12
        temperature: float = 0.9
        top_k: int = 50
        repetition_penalty: float = 1.05
        instruct: Optional[str] = None

    @app.get("/ping")
    def ping():
        return {"status": "ok", "model_loaded": True, "mode": "http"}

    @app.post("/tts/stream")
    def tts_stream(req: TTSRequest):
        if not req.text.strip():
            raise HTTPException(400, "empty text")
        try:
            ref_path = _write_ref_tempfile(req.ref_audio_b64)
        except Exception as e:
            raise HTTPException(400, f"invalid base64: {e}")

        def event_gen():
            try:
                for ev in synthesize_stream(
                    text=req.text.strip(),
                    ref_audio_path=ref_path,
                    ref_text=req.ref_text,
                    language=req.language,
                    chunk_size=req.chunk_size,
                    temperature=req.temperature,
                    top_k=req.top_k,
                    repetition_penalty=req.repetition_penalty,
                    instruct=req.instruct,
                ):
                    yield (json.dumps(ev) + "\n").encode()
            finally:
                try:
                    os.unlink(ref_path)
                except OSError:
                    pass

        return StreamingResponse(
            event_gen(),
            media_type="application/x-ndjson",
            headers={"X-Accel-Buffering": "no", "Cache-Control": "no-cache"},
        )

    return app


# ============================================================
# Entrypoint
# ============================================================
if __name__ == "__main__":
    if SERVER_MODE == "http":
        import uvicorn
        port = int(os.environ.get("PORT", 80))
        print(f"[init] starting HTTP server on :{port}", flush=True)
        uvicorn.run(_build_http_app(), host="0.0.0.0", port=port, log_level="info")
    else:
        import runpod
        _boot(f"runpod {getattr(runpod, '__version__', '?')} ok")
        print("[init] starting RunPod queue worker", flush=True)
        runpod.serverless.start({
            "handler": handler,
            "return_aggregate_stream": True,
        })
