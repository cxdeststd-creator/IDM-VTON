# 1. Base Image (RunPod)
FROM runpod/pytorch:2.2.0-py3.10-cuda12.1.1-devel-ubuntu22.04

WORKDIR /app

# Ortam Değişkenleri
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

# 2. Sistem Araçları
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

# 3. İŞTE ÇÖZÜM BURASI: Önceki belaları siliyoruz 🧹
# Sistemde gelen uyumsuz torchaudio'yu ve torch'u kökten siliyoruz.
RUN pip uninstall -y torch torchvision torchaudio

# 4. Temiz Kurulum: Sadece uyumlu olanları kuruyoruz 🔨
# IDM-VTON için altın standart: Torch 2.0.1 + Vision 0.15.2
RUN pip install torch==2.0.1 torchvision==0.15.2 --index-url https://download.pytorch.org/whl/cu118 --no-cache-dir

# 5. requirements.txt Ayarı
COPY requirements.txt .
# Dosyanın içinde 'torch' varsa siliyoruz ki bizim kurduğumuzu bozmasın.
RUN sed -i '/torch/d' requirements.txt && \
    pip install --upgrade pip && \
    pip install --no-cache-dir --ignore-installed -r requirements.txt

# RunPod kütüphanesi şart
RUN pip install runpod huggingface_hub

# 6. Modelleri Build Sırasında İndir (Timeout Yememek İçin) ⬇️
COPY builder.py .
RUN python3 builder.py

# 7. Kodları Kopyala
COPY . .

# 8. Başlat
CMD [ "python", "-u", "handler.py" ]
