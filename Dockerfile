# 1. RunPod'un en sağlam PyTorch imajını kullanıyoruz
FROM runpod/pytorch:2.2.0-py3.10-cuda12.1.1-devel-ubuntu22.04

# Çalışma klasörü
WORKDIR /app

# 2. SİSTEM GEREKSİNİMLERİ (Önce bunları kurmazsak pip install patlar)
# OpenCV ve derleme araçları için gerekli kütüphaneler
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

# 3. PYTHON KÜTÜPHANELERİNİ KURMA
# Önce requirements.txt'yi kopyalıyoruz
COPY requirements.txt .

# pip güncelle ve kütüphaneleri kur
RUN pip install --upgrade pip
RUN pip install --no-cache-dir --ignore-installed -r requirements.txt
# builder.py çalışsın diye bunları garanti olsun diye tekrar ekliyoruz
RUN pip install runpod huggingface_hub

# 4. MODEL İNDİRME AŞAMASI (BUILDER)
# builder.py dosyasını kopyalayıp çalıştırıyoruz
# Bu aşama build süresini uzatır ama açılışı (cold start) 10 kat hızlandırır
COPY builder.py .
RUN python3 builder.py

# 5. KALAN TÜM DOSYALARI KOPYALA
# (Handler, src klasörü vs. hepsi buraya)
COPY . .

# 6. BAŞLAT
# -u parametresini koyuyoruz ki logları anlık görelim
CMD [ "python", "-u", "handler.py" ]
