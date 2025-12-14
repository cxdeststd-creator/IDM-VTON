# RunPod'un hazır PyTorch imajı
FROM runpod/pytorch:2.2.0-py3.10-cuda12.1.1-devel-ubuntu22.04

WORKDIR /app

# 1. SİSTEM ARAÇLARI (Burayı güçlendirdik)
# cmake ve protobuf-compiler ekledik, build hatalarını çözer
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

# 2. Dosyaları Kopyala
COPY . .

# 3. PYTHON KÜTÜPHANELERİ (Hata toleranslı mod)
RUN pip install --upgrade pip
# --ignore-installed bayrağıyla çakışmaları zorla geçiyoruz
RUN pip install --no-cache-dir --ignore-installed -r requirements.txt
RUN pip install runpod

# 4. Handler'ı başlat
CMD [ "python", "-u", "handler.py" ]
