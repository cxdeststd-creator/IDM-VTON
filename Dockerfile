FROM runpod/pytorch:2.1.0-cuda11.8.0-runtime

ENV DEBIAN_FRONTEND=noninteractive
WORKDIR /app

# -----------------------------
# System dependencies
# -----------------------------
RUN apt-get update && apt-get install -y \
    git \
    wget \
    unzip \
    libgl1 \
    libglib2.0-0 \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# -----------------------------
# Python dependencies (FINAL)
# -----------------------------
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
# Download checkpoints
# -----------------------------
RUN mkdir -p ckpt && \
    wget -q https://huggingface.co/yisol/IDM-VTON/resolve/main/ckpt/densepose.zip -O densepose.zip && \
    wget -q https://huggingface.co/yisol/IDM-VTON/resolve/main/ckpt/humanparsing.zip -O humanparsing.zip && \
    wget
