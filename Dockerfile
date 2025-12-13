# RunPod'un hazır PyTorch imajını kullan (Hızlı açılır)
FROM runpod/pytorch:2.2.0-py3.10-cuda12.1.1-devel-ubuntu22.04

# Çalışma klasörüne geç
WORKDIR /app

# Gerekli sistem araçlarını kur
RUN apt-get update && apt-get install -y git wget

# IDM-VTON Reposunu çek (Eğer senin kendi kodun varsa burayı silip COPY . . yapabilirsin)
RUN git clone https://github.com/yisol/IDM-VTON.git .

# Python kütüphanelerini kur
RUN pip install -r requirements.txt
RUN pip install runpod  # Serverless için şart!

# Modelleri İndir (Bu kısım çok önemli, yoksa her açılışta indirir yavaşlar)
# (Buraya huggingface-cli veya wget ile model indirme komutlarını eklemen lazım)
# Örn: RUN wget https://huggingface.co/yisol/IDM-VTON/resolve/main/unet/diffusion_pytorch_model.bin ...

# Handler kodunu kopyala
COPY handler.py /app/handler.py

# Başlat
CMD [ "python", "-u", "/app/handler.py" ]
