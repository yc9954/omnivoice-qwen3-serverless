# Qwen3-TTS streaming — RunPod Serverless image
# Base: PyTorch 2.8.0 + CUDA 12.9 — covers Ampere (sm_80) → Ada (sm_89) →
# Hopper (sm_90) → Blackwell (sm_120) in one image.
# Requires RunPod endpoint "Allowed CUDA Versions" set to 12.8 and/or 12.9.
FROM runpod/pytorch:1.0.3-cu1290-torch280-ubuntu2404

ENV PYTHONUNBUFFERED=1 \
    HF_HOME=/root/.cache/huggingface \
    HF_HUB_ENABLE_HF_TRANSFER=1 \
    MODEL_PATH=/opt/model \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    TORCH_CUDA_ARCH_LIST="8.0;8.6;8.9;9.0;12.0" \
    MAX_JOBS=4 \
    WARMUP_ON_START=1

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
        libsndfile1 ffmpeg git && \
    rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN python3 -m pip install --upgrade pip wheel && \
    python3 -m pip install -r requirements.txt

# faster-qwen3-tts from source (PyPI lags). Skips flash-attn; relies on
# torch SDPA + CUDAGraph, which is what the library is actually optimized for.
# Qwen3-TTS model card explicitly notes flash_attention_2 is incompatible.
RUN git clone --depth=1 https://github.com/andimarafioti/faster-qwen3-tts.git /tmp/fqtts && \
    python3 -m pip install -e /tmp/fqtts

# bake the model (~4.3GB) into the image so cold start skips the HF download.
# HF_HUB_ENABLE_HF_TRANSFER=1 above gives ~3-5x faster pulls.
RUN python3 -c "\
from huggingface_hub import snapshot_download; \
snapshot_download(repo_id='Qwen/Qwen3-TTS-12Hz-1.7B-Base', local_dir='/opt/model')"

COPY handler.py /app/handler.py

CMD ["sh", "-c", "echo '[boot] sh entered'; python3 --version; echo '[boot] launching handler'; exec python3 -u /app/handler.py"]
