# RunPod'un en stabil PyTorch imajını kullanıyoruz
FROM runpod/pytorch:2.2.0-py3.10-cuda12.1.1-devel-ubuntu22.04

# Çalışma klasörünü ayarla
WORKDIR /app

# --- FİKS BURADA: Önce sistem araçlarını kuruyoruz ---
# Bu satır opencv ve diğer kütüphanelerin "build failed" hatasını çözer
RUN apt-get update && apt-get install -y \
    git \
    wget \
    libgl1-mesa-glx \
    libglib2.0-0 \
    build-essential \
    python3-dev \
    && rm -rf /var/lib/apt/lists/*

# Dosyaları kopyala
COPY . .

# Şimdi Python kütüphanelerini kur (Artık hata vermez)
RUN pip install --no-cache-dir --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install runpod

# Başlat
CMD [ "python", "-u", "handler.py" ]
