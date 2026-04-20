# Qwen3-TTS streaming — RunPod Serverless image
# Base: PyTorch 2.7.1 + CUDA 12.6.3 — driver 560+ required (broad RunPod worker coverage)
# Tradeoff: cu128 has better Blackwell support but some serverless hosts lag driver versions
FROM runpod/pytorch:0.7.0-cu1263-torch271-ubuntu2204

ENV PYTHONUNBUFFERED=1 \
    HF_HOME=/root/.cache/huggingface \
    MODEL_PATH=/opt/model \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    TORCH_CUDA_ARCH_LIST=12.0 \
    MAX_JOBS=4

WORKDIR /app

# system deps
RUN apt-get update && apt-get install -y --no-install-recommends \
        libsndfile1 ffmpeg git && \
    rm -rf /var/lib/apt/lists/*

# python deps (cacheable layer)
COPY requirements.txt .
RUN python3 -m pip install --upgrade pip wheel ninja packaging && \
    python3 -m pip install -r requirements.txt

# install faster-qwen3-tts from source (PyPI lags)
RUN git clone --depth=1 https://github.com/andimarafioti/faster-qwen3-tts.git /tmp/fqtts && \
    python3 -m pip install -e /tmp/fqtts

# install flash-attn (sm_120 / Blackwell). Fall back to skip if wheel/build fails so the
# container still runs (faster-qwen3-tts has a manual attention fallback).
ARG SKIP_FLASH_ATTN=1
RUN if [ "$SKIP_FLASH_ATTN" = "1" ]; then \
      echo "SKIP_FLASH_ATTN=1, skipping"; \
    else \
      python3 -m pip install --no-build-isolation flash-attn==2.7.4.post1 \
        || echo "flash-attn install failed — will fall back to manual attention at runtime" ; \
    fi

# bake the model (~4.3GB) into the image so cold start skips the download
RUN python -c "\
from huggingface_hub import snapshot_download; \
snapshot_download(repo_id='Qwen/Qwen3-TTS-12Hz-1.7B-Base', local_dir='/opt/model')"

# handler code (kept as last cheap layer for iteration)
COPY handler.py /app/handler.py

ENV WARMUP_ON_START=1

# sh wrapper gives us early boot signal before Python runs
CMD ["sh", "-c", "echo '[boot] sh entered'; which python3 || echo 'python3 MISSING'; python3 --version; echo '[boot] launching handler'; exec python3 -u /app/handler.py"]
