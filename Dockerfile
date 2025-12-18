# 1. Base Image (CUDA destekli)
FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04

# 2. Ortam Değişkenleri
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PIP_NO_CACHE_DIR=1

# 3. Sistem Paketleri
RUN apt-get update && apt-get install -y \
    python3.10 \
    python3-dev \
    python3-pip \
    build-essential \
    git \
    curl \
    ffmpeg \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Python Linkleme
RUN ln -sf /usr/bin/python3.10 /usr/bin/python
RUN python -m pip install --upgrade pip setuptools wheel

# 4. TORCH (CUDA 11.8 Uyumlu)
RUN pip install torch==2.1.0+cu118 \
    torchvision==0.16.0+cu118 \
    torchaudio==2.1.0+cu118 \
    --extra-index-url https://download.pytorch.org/whl/cu118 \
    --no-deps

# 5. Temel Kütüphaneler (Runpod ve Insightface ekli)
# transformers'ı burada kuruyoruz ki aşağıdaki import'lar çalışsın
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

# 6. Diffusers ve HuggingFace
RUN pip install \
    diffusers==0.25.0 \
    transformers==4.36.2 \
    accelerate==0.25.0 \
    huggingface-hub==0.19.4 \
    safetensors \
    einops \
    xformers==0.0.22.post7 \
    --no-deps

# 7. REPONU ÇEKİYORUZ
WORKDIR /app
RUN git clone https://github.com/cxdeststd-creator/IDM-VTON.git

# Repo içine giriyoruz
WORKDIR /app/IDM-VTON

# ⚠️ ÖNEMLİ ADIM:
# Bilgisayarındaki son düzenlediğin (ensure_ckpts fonksiyonlu) handler.py dosyasını
# Docker'ın içine kopyalıyoruz. Bu dosyanın Dockerfile ile yan yana olduğundan emin ol.
COPY handler.py .

# 🔥 KRİTİK HAMLE: BUILD SIRASINDA İNDİRME 🔥
# Bu komut, handler.py içindeki ensure_ckpts() fonksiyonunu çalıştırır.
# Modelleri (15GB+) indirip imajın içine kaydeder.
# NOT: Bu adım internet hızına göre 10-20 dakika sürebilir. Bekle, kapatma.
RUN python -c "from handler import ensure_ckpts; ensure_ckpts()"

# 8. Başlatma
CMD ["python", "-u", "handler.py"]
