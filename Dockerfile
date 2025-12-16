# 1. Base Image
FROM python:3.10

# Çalışma dizini
WORKDIR /app

# 2. Sistem Kütüphaneleri
RUN apt-get update --fix-missing && apt-get install -y \
    build-essential \
    libgl1 \
    libglib2.0-0 \
    git \
    && rm -rf /var/lib/apt/lists/*

# Pip güncelle
RUN pip install --upgrade pip

# 3. TORCH SABİTLEME
RUN pip install torch==2.0.1 torchvision==0.15.2 --index-url https://download.pytorch.org/whl/cu118 --no-cache-dir

# 4. Requirements Dosyasını Kopyala ve Kur
COPY requirements.txt .
RUN pip install -r requirements.txt --no-cache-dir

# ==========================================================
# 5. HATA DÜZELTİCİLER (BURASI ÇOK ÖNEMLİ)
# ==========================================================

# Düzeltme 1: NumPy Hatası için (Exit Code 100/Segfault önleyici)
RUN pip install "numpy<2" --force-reinstall --no-cache-dir

# Düzeltme 2: HuggingFace 'cached_download' Hatası için
# Yeni sürümlerde bu fonksiyon kaldırıldı, o yüzden sürümü düşürüyoruz.
RUN pip install "huggingface_hub<0.25.0" --force-reinstall --no-cache-dir

# ==========================================================

# 6. Dosyaları kopyala
COPY . .

# 7. Başlatma
CMD [ "python", "-u", "handler.py" ]
