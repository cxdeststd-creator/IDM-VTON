FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
WORKDIR /app

# -----------------------------
# System deps
# -----------------------------
RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    git \
    wget \
    unzip \
    libgl1 \
    libglib2.0-0 \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

RUN ln -s /usr/bin/python3 /usr/bin/python

# -----------------------------
# Python deps (FINAL LOCK)
# -----------------------------
RUN pip install --no-cache-dir \
    torch==2.1.0+cu118 \
    torchvision==0.16.0+cu118 \
    torchaudio==2.1.0+cu118 \
    --extra-index-url https://download.pytorch.org/whl/cu118

RUN pip install --no-cache-dir \
    diffusers==0.25.1 \
    transformers==4.36.2 \
    accelerate==0.25.0 \
    safetensors \
    huggingface_hub \
    xformers \
    opencv-python \
    pillow \
    numpy \
    scipy \
    tqdm \
    matplotlib \
    onnxruntime \
    runpod

# -----------------------------
# Clone IDM-VTON
# -----------------------------
RUN git clone https://github.com/yisol/IDM-VTON.git
WORKDIR /app/IDM-VTON

# -----------------------------
# Download checkpoints (FIXED)
# -----------------------------
RUN apt-get update && apt-get install -y ca-certificates && rm -rf /var/lib/apt/lists/*

RUN mkdir -p ckpt && \
    wget -q --show-progress -O densepose.zip \
      https://huggingface.co/yisol/IDM-VTON/resolve/main/ckpt/densepose.zip && \
    wget -q --show-progress -O humanparsing.zip \
      https://huggingface.co/yisol/IDM-VTON/resolve/main/ckpt/humanparsing.zip && \
    wget -q --show-progress -O openpose.zip \
      https://huggingface.co/yisol/IDM-VTON/resolve/main/ckpt/openpose.zip && \
    unzip -o densepose.zip -d ckpt && \
    unzip -o humanparsing.zip -d ckpt && \
    unzip -o openpose.zip -d ckpt && \
    rm densepose.zip humanparsing.zip openpose.zip

# -----------------------------
# Fix BROKEN ONNX (CRITICAL)
# -----------------------------
RUN python - <<EOF
from huggingface_hub import hf_hub_download
hf_hub_download(
    repo_id="yisol/IDM-VTON",
    filename="ckpt/humanparsing/parsing_atr.onnx",
    force_download=True,
    local_dir=".",
    local_dir_use_symlinks=False
)
print("ONNX fixed")
EOF

# -----------------------------
# RunPod serverless
# -----------------------------
COPY handler.py /app/IDM-VTON/handler.py

CMD ["python", "-u", "/usr/local/lib/python3.10/dist-packages/runpod/serverless/start.py"]
