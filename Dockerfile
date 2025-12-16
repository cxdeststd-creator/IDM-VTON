FROM runpod/pytorch:2.1.0-cuda12.1.1-runtime

ENV DEBIAN_FRONTEND=noninteractive
WORKDIR /app

# -----------------------------
# System dependencies
# -----------------------------
RUN apt-get update && apt-get install -y \
    git \
    wget \
    unzip \
    libgl1 \
    libglib2.0-0 \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# -----------------------------
# Python dependencies (FINAL)
# -----------------------------
RUN pip install --no-cache-dir \
    diffusers==0.25.1 \
    transformers==4.36.2 \
    accelerate==0.25.0 \
    safetensors \
    huggingface_hub \
    xformers \
    opencv-python \
    pillow \
    numpy \
    scipy \
    tqdm \
    matplotlib \
    onnxruntime \
    runpod

# -----------------------------
# Clone IDM-VTON
# -----------------------------
RUN git clone https://github.com/yisol/IDM-VTON.git
WORKDIR /app/IDM-VTON

# -----------------------------
# Download checkpoints
# -----------------------------
RUN mkdir -p ckpt && \
    wget -q https://huggingface.co/yisol/IDM-VTON/resolve/main/ckpt/densepose.zip -O densepose.zip && \
    wget -q https://huggingface.co/yisol/IDM-VTON/resolve/main/ckpt/humanparsing.zip -O humanparsing.zip && \
    wget -q https://huggingface.co/yisol/IDM-VTON/resolve/main/ckpt/openpose.zip -O openpose.zip && \
    unzip densepose.zip -d ckpt && \
    unzip humanparsing.zip -d ckpt && \
    unzip openpose.zip -d ckpt && \
    rm *.zip

# -----------------------------
# 🔥 FORCE FIX BROKEN ONNX
# -----------------------------
RUN python - <<EOF
from huggingface_hub import hf_hub_download
hf_hub_download(
    repo_id="yisol/IDM-VTON",
    filename="ckpt/humanparsing/parsing_atr.onnx",
    force_download=True,
    local_dir=".",
    local_dir_use_symlinks=False
)
print("ONNX parsing_atr fixed")
EOF

# -----------------------------
# RunPod serverless
# -----------------------------
WORKDIR /app
COPY handler.py /app/IDM-VTON/handler.py

CMD ["python", "-u", "/usr/local/lib/python3.10/dist-packages/runpod/serverless/start.py"]
