# ======================================
# 1. CUDA BASE
# ======================================
FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04

# ======================================
# 2. SYSTEM
# ======================================
RUN apt-get update && apt-get install -y \
    python3.10 \
    python3.10-dev \
    python3-pip \
    git \
    wget \
    unzip \
    libgl1 \
    libglib2.0-0 \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

RUN ln -s /usr/bin/python3.10 /usr/bin/python
RUN python -m pip install --upgrade pip

WORKDIR /app

# ======================================
# 3. TORCH (CUDA)
# ======================================
RUN pip install torch==2.0.1 torchvision==0.15.2 \
    --index-url https://download.pytorch.org/whl/cu118

# ======================================
# 4. PYTHON DEPS
# ======================================
COPY requirements.txt .
RUN pip install -r requirements.txt --no-cache-dir

# ======================================
# 5. IDM-VTON CODE
# ======================================
RUN git clone https://github.com/yisol/IDM-VTON.git
WORKDIR /app/IDM-VTON

# ======================================
# 6. DIFFUSERS PATCH (ImageProjection)
# ======================================
RUN sed -i \
's/from diffusers.models import AutoencoderKL, ImageProjection, UNet2DConditionModel/from diffusers.models import AutoencoderKL, UNet2DConditionModel; from diffusers.models.embeddings import ImageProjection/' \
src/tryon_pipeline.py

# ======================================
# 7. MODEL WEIGHTS (AUTODOWNLOAD)
# ======================================
# ---- CKPT ----
RUN mkdir -p ckpt && \
    wget -q https://huggingface.co/yisol/IDM-VTON/resolve/main/ckpt/densepose.zip -O densepose.zip && \
    wget -q https://huggingface.co/yisol/IDM-VTON/resolve/main/ckpt/humanparsing.zip -O humanparsing.zip && \
    wget -q https://huggingface.co/yisol/IDM-VTON/resolve/main/ckpt/openpose.zip -O openpose.zip && \
    unzip densepose.zip -d ckpt && \
    unzip humanparsing.zip -d ckpt && \
    unzip openpose.zip -d ckpt && \
    rm *.zip

# ======================================
# 8. HANDLER
# ======================================
COPY handler.py /app/IDM-VTON/handler.py

# ======================================
# 9. START
# ======================================
CMD ["python", "-u", "handler.py"]
