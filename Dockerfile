# 1. RunPod'un en sağlam PyTorch imajı (Senin verdiğin base)
FROM runpod/pytorch:2.2.0-py3.10-cuda12.1.1-devel-ubuntu22.04

# Çalışma klasörü
WORKDIR /app

# Ortam Değişkenleri (Hata almamak için)
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

# 2. SİSTEM GEREKSİNİMLERİ
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

# 3. KRİTİK ADIM: İstediğin Torch Versiyonunu Zorla Çakıyoruz 🔨
# Base imajda gelen torch 2.2'yi siler, senin istediğin 2.0.1'i kurar.
# "nms does not exist" hatasını çözen satır burası.
RUN pip install torch==2.0.1 torchvision==0.15.2 --index-url https://download.pytorch.org/whl/cu118 --no-cache-dir --force-reinstall

# 4. PYTHON KÜTÜPHANELERİNİ KURMA
COPY requirements.txt .

# requirements.txt içinde torch varsa sil, yoksa bizim kurduğumuzu bozar.
# Sonra geri kalanları kur.
RUN sed -i '/torch/d' requirements.txt && \
    pip install --upgrade pip && \
    pip install --no-cache-dir --ignore-installed -r requirements.txt

# RunPod ve HuggingFace (Gerekirse)
RUN pip install runpod huggingface_hub

# 5. MODEL İNDİRME AŞAMASI (BUILDER)
COPY builder.py .
# Bu scriptin içindeki indirme fonksiyonları çalışacak
RUN python3 builder.py

# 6. KALAN TÜM DOSYALARI KOPYALA
COPY . .

# 7. BAŞLAT
CMD [ "python", "-u", "handler.py" ]
