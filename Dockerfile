# Qwen3-TTS streaming — RunPod Serverless image
# Base: RunPod PyTorch 2.8 + CUDA 12.8.1 (matches RunPod worker driver), Ubuntu 22.04
# Earlier cu1290 caused nvidia-container-cli to fail on hosts limited to CUDA 12.8
FROM runpod/pytorch:1.0.3-cu1281-torch280-ubuntu2204

ENV PYTHONUNBUFFERED=1 \
    HF_HOME=/root/.cache/huggingface \
    MODEL_PATH=/opt/model \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

# system deps for audio IO (libsndfile, ffmpeg for wider codec support)
RUN apt-get update && apt-get install -y --no-install-recommends \
        libsndfile1 ffmpeg git && \
    rm -rf /var/lib/apt/lists/*

# python deps — install first (cache friendly)
COPY requirements.txt .
RUN pip install --upgrade pip wheel && \
    pip install -r requirements.txt

# install faster-qwen3-tts from source (PyPI lags)
RUN git clone --depth=1 https://github.com/andimarafioti/faster-qwen3-tts.git /tmp/fqtts && \
    pip install -e /tmp/fqtts

# bake the model into the image (~4.3GB) — eliminates cold-start download
RUN python -c "\
from huggingface_hub import snapshot_download; \
snapshot_download(repo_id='Qwen/Qwen3-TTS-12Hz-1.7B-Base', local_dir='/opt/model')"

# handler code
COPY handler.py /app/handler.py

ENV WARMUP_ON_START=1

# wrapper: verify python works + capture any early crash
CMD ["sh", "-c", "echo '[boot] sh entered'; which python3; which python; python3 --version; echo '[boot] launching handler'; exec python3 -u /app/handler.py"]
