FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PIP_NO_CACHE_DIR=1

# 1. build-essential ve python3-dev EKLENDİ (Insightface derlemek için şart)
RUN apt-get update && apt-get install -y \
    python3.10 \
    python3-dev \
    python3-pip \
    build-essential \
    git \
    curl \
    unzip \
    ffmpeg \
    libgl1 \
    libglib2.0-0 \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

RUN ln -sf /usr/bin/python3.10 /usr/bin/python
RUN python -m pip install --upgrade pip setuptools wheel

# ---- TORCH (NO-DEPS) ----
RUN pip install torch==2.1.0+cu118 \
    torchvision==0.16.0+cu118 \
    torchaudio==2.1.0+cu118 \
    --extra-index-url https://download.pytorch.org/whl/cu118 \
    --no-deps

# ---- CORE DEPS ----
# runpod, insightface ve scipy EKLENDİ
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
    onnxruntime-gpu==1.16.3 \
    runpod \
    scipy \
    insightface

# ---- HF STACK (NO-DEPS) ----
# diffusers versiyonunu 0.24.0'a çektim çünkü IDM-VTON kodu yenisiyle çalışmaz
RUN pip install \
    diffusers==0.24.0 \
    transformers==4.36.2 \
    accelerate==0.25.0 \
    huggingface-hub==0.19.4 \
    safetensors \
    einops \
    xformers==0.0.22.post7 \
    --no-deps

# ---- IDM-VTON CODE ----
WORKDIR /app
RUN git clone https://github.com/yisol/IDM-VTON.git

WORKDIR /app/IDM-VTON

# !!! İŞTE UNUTTUĞUN SATIR BU !!!
# Bilgisayarındaki handler.py dosyasını konteynerin içine atıyoruz
COPY handler.py .

CMD ["python", "-u", "handler.py"]
