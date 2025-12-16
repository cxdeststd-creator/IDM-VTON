# ===============================
# 1. CUDA BASE (ŞART)
# ===============================
FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04

# ===============================
# 2. SYSTEM
# ===============================
RUN apt-get update && apt-get install -y \
    python3.10 \
    python3.10-dev \
    python3-pip \
    git \
    libgl1 \
    libglib2.0-0 \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

RUN ln -s /usr/bin/python3.10 /usr/bin/python
RUN python -m pip install --upgrade pip

WORKDIR /app

# ===============================
# 3. TORCH (CUDA UYUMLU)
# ===============================
RUN pip install torch==2.0.1 torchvision==0.15.2 \
    --index-url https://download.pytorch.org/whl/cu118

# ===============================
# 4. REQUIREMENTS
# ===============================
COPY requirements.txt .
RUN pip install -r requirements.txt --no-cache-dir

# ===============================
# 5. IDM-VTON REPO
# ===============================
RUN git clone https://github.com/yisol/IDM-VTON.git
WORKDIR /app/IDM-VTON

# ===============================
# 6. ZORUNLU PATCH (ImageProjection)
# ===============================
RUN sed -i \
's/from diffusers.models import AutoencoderKL, ImageProjection, UNet2DConditionModel/from diffusers.models import AutoencoderKL, UNet2DConditionModel; from diffusers.models.embeddings import ImageProjection/' \
src/tryon_pipeline.py

# ===============================
# 7. HANDLER
# ===============================
COPY handler.py /app/IDM-VTON/handler.py

# ===============================
# 8. START
# ===============================
CMD ["python", "-u", "handler.py"]
