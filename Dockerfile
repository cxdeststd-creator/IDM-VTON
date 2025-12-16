# 1. Base Image (Tam sürüm)
FROM python:3.10

# Çalışma dizini
WORKDIR /app

# 2. Sistem Kütüphaneleri (Hata riskini sıfıra indirmek için --fix-missing ile)
RUN apt-get update --fix-missing && apt-get install -y \
    build-essential \
    libgl1 \
    libglib2.0-0 \
    git \
    && rm -rf /var/lib/apt/lists/*

# Pip güncelle
RUN pip install --upgrade pip

# 3. TORCH SABİTLEME (Değişmedi)
RUN pip install torch==2.0.1 torchvision==0.15.2 --index-url https://download.pytorch.org/whl/cu118 --no-cache-dir

# 4. BAĞIMLILIKLARI YÜKLEME
COPY requirements.txt .
# Önce requirements dosyasını kuruyoruz
RUN pip install -r requirements.txt --no-cache-dir

# 5. KRİTİK DÜZELTME: NUMPY DOWNGRADE
# Requirements ne yüklerse yüklesin, biz zorla NumPy'ı 1.x sürümüne çekiyoruz.
# Bu satır o hatayı %100 çözer.
RUN pip install "numpy<2" --force-reinstall --no-cache-dir

# 6. Dosyaları kopyala
COPY . .

# 7. BAŞLATMA KOMUTU (Düzeltildi)
# -u parametresi logları anlık görmeni sağlar (unbuffered)
CMD [ "python", "-u", "handler.py" ]
