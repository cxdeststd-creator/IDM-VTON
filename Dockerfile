# 1. Base Image
FROM runpod/pytorch:2.2.0-py3.10-cuda12.1.1-devel-ubuntu22.04

WORKDIR /app
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

# 2. Sistem Araçları
RUN apt-get update && apt-get install -y \
    git wget cmake protobuf-compiler libgl1-mesa-glx libglib2.0-0 build-essential python3-dev \
    && rm -rf /var/lib/apt/lists/*

# 3. TEMİZLİK: Çakışan Kütüphaneleri Sil (Dependency Conflict Çözümü) 🧹
RUN pip uninstall -y torch torchvision torchaudio

# 4. KURULUM: Uyumlu Versiyonlar (Torch 2.0.1 + Vision 0.15.2) 🔨
RUN pip install torch==2.0.1 torchvision==0.15.2 --index-url https://download.pytorch.org/whl/cu118 --no-cache-dir

# 5. Requirements Ayarı
COPY requirements.txt .
# İçindeki torch'u silip kalanı kuruyoruz ki bizim kurduğumuzu bozmasın
RUN sed -i '/torch/d' requirements.txt && \
    pip install --upgrade pip && \
    pip install --no-cache-dir --ignore-installed -r requirements.txt

# RunPod kütüphanesi
RUN pip install runpod huggingface_hub protobuf

# 6. MODELLERİ İNDİR (Builder Script) ⬇️
COPY builder.py .
RUN python3 builder.py

# 7. Kodları Kopyala
COPY . .

# ==========================================================
# 8. KRİTİK TAMİR (Bunu yapmazsak Pod AÇILMAZ!) 🛠️
# ==========================================================
# Orijinal koddaki hatalı importları temizliyoruz.
RUN find src -name "*.py" -exec sed -i 's/, PixArtAlphaTextProjection//g' {} + && \
    find src -name "*.py" -exec sed -i 's/, RMSNorm//g' {} + && \
    find src -name "*.py" -exec sed -i 's/, AdaLayerNormContinuous//g' {} +

# 9. Başlat
CMD [ "python", "-u", "handler.py" ]
