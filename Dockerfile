FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
WORKDIR /app

# -------------------------------------------------
# System packages
# -------------------------------------------------
RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    git \
    libgl1 \
    libglib2.0-0 \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

RUN ln -s /usr/bin/python3 /usr/bin/python

# -------------------------------------------------
# Pip upgrade
# -------------------------------------------------
RUN pip install --upgrade pip

# -------------------------------------------------
# TORCH (LOCKED – FIRST)
# -------------------------------------------------
RUN pip install --no-cache-dir \
    torch==2.1.0+cu118 \
    torchvision==0.16.0+cu118 \
    torchaudio==2.1.0+cu118 \
    --extra-index-url https://download.pytorch.org/whl/cu118

# -------------------------------------------------
# Core libs (NO TORCH OVERRIDE)
# -------------------------------------------------
RUN pip install --no-cache-dir --no-deps \
    diffusers==0.25.1 \
    transformers==4.36.2 \
    accelerate==0.25.0

# -------------------------------------------------
# Remaining deps
# -------------------------------------------------
RUN pip install --no-cache-dir \
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

# -------------------------------------------------
# IDM-VTON
# -------------------------------------------------
RUN git clone https://github.com/yisol/IDM-VTON.git

COPY handler.py /app/IDM-VTON/handler.py

# -------------------------------------------------
# RunPod serverless
# -------------------------------------------------
CMD ["python", "-u", "/usr/local/lib/python3.10/dist-packages/runpod/serverless/start.py"]
