FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PIP_NO_CACHE_DIR=1

# 1. Sistem Paketleri
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

# 2. TORCH (CUDA 11.8)
RUN pip install torch==2.1.0+cu118 \
    torchvision==0.16.0+cu118 \
    torchaudio==2.1.0+cu118 \
    --extra-index-url https://download.pytorch.org/whl/cu118 \
    --no-deps

# 3. Temel Kütüphaneler
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

# 4. Diffusers ve HuggingFace (Versiyonlar Önemli)
RUN pip install \
    diffusers==0.24.0 \
    transformers==4.36.2 \
    accelerate==0.25.0 \
    huggingface-hub==0.19.4 \
    safetensors \
    einops \
    xformers==0.0.22.post7 \
    --no-deps

# 5. Kodu İndirme
WORKDIR /app
RUN git clone https://github.com/yisol/IDM-VTON.git

WORKDIR /app/IDM-VTON

# ==========================================================
# 🛑 KOD AMELİYATI (SED KOMUTLARI) - BURASI HATALARI ÇÖZER
# ==========================================================

# Düzeltme 1: ImageProjection Hatası
# "diffusers.models" içinden ImageProjection kalktı, onu "models.embeddings" içine yönlendiriyoruz.
RUN sed -i 's/from diffusers.models import AutoencoderKL, ImageProjection, UNet2DConditionModel/from diffusers.models import AutoencoderKL, UNet2DConditionModel; from diffusers.models.embeddings import ImageProjection/g' src/tryon_pipeline.py

# Düzeltme 2: FusedAttnProcessor2_0 Hatası (Sırada bekleyen hata buydu, peşinen çözdük)
# Bu isim değişti, eski ismi yeni isme (AttnProcessor2_0) eşitliyoruz.
RUN sed -i 's/from diffusers.models.attention_processor import FusedAttnProcessor2_0/from diffusers.models.attention_processor import AttnProcessor2_0 as FusedAttnProcessor2_0/g' src/tryon_pipeline.py

# ==========================================================

# 6. Handler Dosyasını İçeri Atma
COPY handler.py .

CMD ["python", "-u", "handler.py"]
