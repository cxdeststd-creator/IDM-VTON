# Python 3.10 tabanlı hafif imaj (Gereksiz çöplerden arınmış)
FROM python:3.10-slim

# Çalışma dizinini ayarla
WORKDIR /app

# 1. SİSTEM BAĞIMLILIKLARI
# OpenCV ve Insightface'in çalışması (ve derlenmesi) için bunlar ŞART.
# Yoksa "libGL.so not found" veya "gcc failed" hatası alırsın.
RUN apt-get update && apt-get install -y \
    build-essential \
    libgl1-mesa-glx \
    libglib2.0-0 \
    git \
    && rm -rf /var/lib/apt/lists/*

# 2. KRİTİK ADIM: TORCH VE VISION'I ELLE SABİTLEME
# requirements.txt'den ÖNCE bunları kuruyoruz ki pip kafasına göre 2.9.1 indirmesin.
RUN pip install torch==2.0.1 torchvision==0.15.2 --index-url https://download.pytorch.org/whl/cu118 --no-cache-dir

# 3. KALAN PAKETLERİ YÜKLEME
# (requirements.txt içinden torch ve torchvision satırlarını sildiğinden emin ol)
COPY requirements.txt .
RUN pip install -r requirements.txt --no-cache-dir

# 4. Uygulama dosyalarını kopyala
COPY . .

# (Opsiyonel) Eğer app.py çalıştıracaksan:
CMD ["python", "app.py"]
