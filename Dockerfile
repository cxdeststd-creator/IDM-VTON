FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PIP_NO_CACHE_DIR=1
ENV PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y \
    python3.10 \
    python3-pip \
    git \
    wget \
    unzip \
    ffmpeg \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

RUN ln -sf /usr/bin/python3.10 /usr/bin/python

# pip güncelle
RUN python -m pip install --upgrade pip setuptools wheel

# ---------- TORCH (ASLA deps ALMA) ----------
RUN pip install torch==2.1.0+cu118 \
    torchvision==0.16.0+cu118 \
    torchaudio==2.1.0+cu118 \
    --extra-index-url https://download.pytorch.org/whl/cu118 \
    --no-deps

# ---------- ZORUNLU DÜŞÜK SEVİYE DEPS ----------
RUN pip install \
    numpy==1.26.4 \
    pillow==10.2.0 \
    opencv-python==4.9.0.80 \
    psutil \
    regex \
    tokenizers==0.15.2 \
    importlib-metadata \
    protobuf==3.20.3 \
    matplotlib \
    onnxruntime-gpu==1.16.3

# ---------- HF STACK (NO-DEPS) ----------
RUN pip install \
    diffusers==0.25.1 \
    transformers==4.36.2 \
    accelerate==0.25.0 \
    huggingface-hub==0.19.4 \
    safetensors \
    einops \
    xformers==0.0.22.post7 \
    --no-deps

# ---------- IDM-VTON ----------
WORKDIR /app
RUN git clone https://github.com/yisol/IDM-VTON.git

# ---------- CKPT ----------
WORKDIR /app/IDM-VTON
RUN mkdir -p ckpt && \
    wget -O densepose.zip https://huggingface.co/yisol/IDM-VTON/resolve/main/ckpt/densepose.zip && \
    wget -O humanparsing.zip https://huggingface.co/yisol/IDM-VTON/resolve/main/ckpt/humanparsing.zip && \
    wget -O openpose.zip https://huggingface.co/yisol/IDM-VTON/resolve/main/ckpt/openpose.zip && \
    unzip -o densepose.zip -d ckpt && \
    unzip -o humanparsing.zip -d ckpt && \
    unzip -o openpose.zip -d ckpt && \
    rm *.zip

CMD ["python", "handler.py"]
