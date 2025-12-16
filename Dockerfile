# 1. Slim yerine TAM sürüm kullanıyoruz (Hata riskini azaltır, biraz daha büyük boyuttur ama çalışır)
FROM python:3.10

# Çalışma dizini
WORKDIR /app

# 2. SİSTEM KÜTÜPHANELERİ
# "libgl1-mesa-glx" yerine "libgl1" yazıyoruz (Yeni Debian sürümü uyumu için)
# "--fix-missing" ekledik ki internet koparsa tekrar denesin.
RUN apt-get update --fix-missing && apt-get install -y \
    build-essential \
    libgl1 \
    libglib2.0-0 \
    git \
    && rm -rf /var/lib/apt/lists/*

# Pip güncelle
RUN pip install --upgrade pip

# 3. TORCH SABİTLEME (Aynı mantık, değişmedi)
RUN pip install torch==2.0.1 torchvision==0.15.2 --index-url https://download.pytorch.org/whl/cu118 --no-cache-dir

# 4. Requirements Dosyasını Kopyala ve Kur
COPY requirements.txt .
# onnxruntime-gpu yazdığından emin ol
RUN pip install -r requirements.txt --no-cache-dir

# 5. Dosyaları kopyala
COPY . .

# 6. Başlat
CMD ["python", "app.py"]
