# 1. Base Image (RunPod)
FROM runpod/pytorch:2.2.0-py3.10-cuda12.1.1-devel-ubuntu22.04

WORKDIR /app

# Ortam Değişkenleri
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

# 2. Sistem Kütüphaneleri
RUN apt-get update && apt-get install -y \
    git \
    wget \
    cmake \
    protobuf-compiler \
    libgl1-mesa-glx \
    libglib2.0-0 \
    build-essential \
    python3-dev \
    && rm -rf /var/lib/apt/lists/*

# 3. TEMİZLİK VE KURULUM (Hatanın Çözümü Burası) 🧹
# Önce sistemdeki tüm torch ailesini siliyoruz (torchaudio çakışmasın diye).
# Sonra temiz bir şekilde 2.0.1 ve 0.15.2 kuruyoruz.
RUN pip uninstall -y torch torchvision torchaudio && \
    pip install torch==2.0.1 torchvision==0.15.2 --index-url https://download.pytorch.org/whl/cu118 --no-cache-dir

# 4. Kütüphaneleri Kur
COPY requirements.txt .

# requirements.txt içinden torch'u siliyoruz ki bizim kurduğumuzu bozmasın.
RUN sed -i '/torch/d' requirements.txt && \
    pip install --upgrade pip && \
    pip install --no-cache-dir --ignore-installed -r requirements.txt

# RunPod/HF kütüphaneleri
RUN pip install runpod huggingface_hub

# 5. Builder (Model İndirme)
COPY builder.py .
RUN python3 builder.py

# 6. Kalan Dosyalar
COPY . .

# 7. Başlat
CMD [ "python", "-u", "handler.py" ]
