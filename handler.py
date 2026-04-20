"""
RunPod Serverless handler for Qwen3-TTS streaming voice cloning.
"""
import sys

# --- super-early log so we can tell whether Python even starts ---
_BOOT_LOG = "/tmp/handler_boot.log"
def _boot(msg: str):
    line = f"[boot] {msg}"
    try:
        sys.stdout.write(line + "\n"); sys.stdout.flush()
    except Exception:
        pass
    try:
        sys.stderr.write(line + "\n"); sys.stderr.flush()
    except Exception:
        pass
    try:
        with open(_BOOT_LOG, "a") as _f:
            _f.write(line + "\n")
    except Exception:
        pass

_boot("python started")

# --- wrap every import to catch crashes early ---
try:
    import base64
    import io
    import os
    import time
    import tempfile
    import traceback
    from typing import Generator
    _boot("stdlib imports ok")

    import numpy as np
    _boot(f"numpy {np.__version__} ok")

    import soundfile as sf
    _boot(f"soundfile {sf.__version__} ok")

    import torch as _torch
    _boot(f"torch {_torch.__version__} cuda_avail={_torch.cuda.is_available()}")
    if _torch.cuda.is_available():
        _boot(f"cuda device: {_torch.cuda.get_device_name(0)}")

    import runpod
    _boot(f"runpod {getattr(runpod, '__version__', '?')} ok")

    from faster_qwen3_tts import FasterQwen3TTS
    _boot("faster_qwen3_tts ok")
except Exception as _e:
    _boot(f"FATAL import error: {_e!r}")
    try:
        import traceback as _tb
        _tb.print_exc(file=sys.stderr)
    except Exception:
        pass
    raise

MODEL_PATH = os.environ.get("MODEL_PATH", "/workspace/models/Qwen3-TTS-12Hz-1.7B-Base")
WARMUP_ON_START = os.environ.get("WARMUP_ON_START", "1") == "1"

print(f"[init] loading faster-qwen3-tts from {MODEL_PATH}", flush=True)
_t0 = time.time()
try:
    _dtype = _torch.bfloat16 if _torch.cuda.is_available() else _torch.float32
    _device = "cuda:0" if _torch.cuda.is_available() else "cpu"
    print(f"[init] device={_device}, dtype={_dtype}", flush=True)
    _MODEL = FasterQwen3TTS.from_pretrained(
        MODEL_PATH, device_map=_device, dtype=_dtype, attn_implementation="sdpa",
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
