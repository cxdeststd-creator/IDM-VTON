# handler.py
import os
import io
import sys
import base64
import shutil
import numpy as np
import torch
from PIL import Image, ImageOps
import runpod
from torchvision import transforms
from huggingface_hub import snapshot_download

# -------------------------------------------------
# GLOBAL
# -------------------------------------------------
BASE_REPO = "yisol/IDM-VTON"
MODEL_LOADED = False
model = {}

sys.path.append(os.getcwd())

# -------------------------------------------------
# DOWNLOAD MODELS (DOĞRU KAYNAKLAR)
# -------------------------------------------------
def download_models():
    print("⬇️ IDM-VTON çekiliyor...")
    snapshot_download(
        repo_id="yisol/IDM-VTON",
        local_dir="ckpt",
        allow_patterns=[
            "unet/*",
            "scheduler/*",
            "vae/*",
            "humanparsing/*",
            "densepose/*",
            "openpose/*"
        ],
        local_dir_use_symlinks=True
    )

    print("⬇️ SDXL tokenizer + text encoder çekiliyor...")
    snapshot_download(
        repo_id="stabilityai/stable-diffusion-xl-base-1.0",
        local_dir="ckpt",
        allow_patterns=[
            "tokenizer/*",
            "tokenizer_2/*",
            "text_encoder/*",
            "text_encoder_2/*"
        ],
        local_dir_use_symlinks=True
    )

    print("⬇️ Image encoder çekiliyor...")
    snapshot_download(
        repo_id="laion/CLIP-ViT-H-14-laion2B-s32B-b79K",
        local_dir="image_encoder",
        allow_patterns=["*.json", "*.safetensors", "*.txt"],
        local_dir_use_symlinks=True
    )

    print("⬇️ UNet garm çekiliyor...")
    snapshot_download(
        repo_id="stabilityai/stable-diffusion-xl-base-1.0",
        local_dir="unet_garm",
        allow_patterns=["unet/*", "*.json"],
        local_dir_use_symlinks=True
    )

# -------------------------------------------------
# LOAD MODEL
# -------------------------------------------------
def load_model():
    global MODEL_LOADED, model
    if MODEL_LOADED:
        return model

    download_models()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"🚀 Model yükleniyor | Device: {device}")

    from transformers import (
        AutoTokenizer,
        CLIPTextModel,
        CLIPTextModelWithProjection,
        CLIPVisionModelWithProjection,
        CLIPImageProcessor,
    )
    from preprocess.humanparsing.run_parsing import Parsing
    from preprocess.openpose.run_openpose import OpenPose
    from src.tryon_pipeline import StableDiffusionXLInpaintPipeline
    from src.unet_hacked_tryon import UNet2DConditionModel
    from src.unet_hacked_garmnet import UNet2DConditionModel as UNetGarm

    parsing = Parsing(0)
    openpose = OpenPose(0)

    tokenizer = AutoTokenizer.from_pretrained("ckpt", subfolder="tokenizer", use_fast=False)
    tokenizer_2 = AutoTokenizer.from_pretrained("ckpt", subfolder="tokenizer_2", use_fast=False)

    text_encoder = CLIPTextModel.from_pretrained(
        "ckpt", subfolder="text_encoder", torch_dtype=torch.float16
    ).to(device)

    text_encoder_2 = CLIPTextModelWithProjection.from_pretrained(
        "ckpt", subfolder="text_encoder_2", torch_dtype=torch.float16
    ).to(device)

    image_encoder = CLIPVisionModelWithProjection.from_pretrained(
        "image_encoder", torch_dtype=torch.float16
    ).to(device)

    feature_extractor = CLIPImageProcessor.from_pretrained("image_encoder")

    unet = UNet2DConditionModel.from_pretrained(
        "ckpt", subfolder="unet", torch_dtype=torch.float16
    ).to(device)

    unet_garm = UNetGarm.from_pretrained(
        "unet_garm/unet", torch_dtype=torch.float16, use_safetensors=True
    ).to(device)

    pipe = StableDiffusionXLInpaintPipeline.from_pretrained(
        "ckpt",
        unet=unet,
        unet_encoder=unet_garm,
        image_encoder=image_encoder,
        feature_extractor=feature_extractor,
        text_encoder=text_encoder,
        text_encoder_2=text_encoder_2,
        tokenizer=tokenizer,
        tokenizer_2=tokenizer_2,
        torch_dtype=torch.float16,
        use_safetensors=True,
    ).to(device)

    model = {
        "pipe": pipe,
        "parsing": parsing,
        "openpose": openpose,
        "device": device,
    }

    MODEL_LOADED = True
    print("✅ MODEL HAZIR")
    return model

# -------------------------------------------------
# HELPERS
# -------------------------------------------------
def b64_to_img(b64):
    if "," in b64:
        b64 = b64.split(",")[1]
    img = Image.open(io.BytesIO(base64.b64decode(b64))).convert("RGB")
    return ImageOps.exif_transpose(img)

def img_to_b64(img):
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=95)
    return base64.b64encode(buf.getvalue()).decode()

def smart_resize(img, w, h):
    img = ImageOps.exif_transpose(img)
    img.thumbnail((w, h), Image.Resampling.LANCZOS)
    bg = Image.new("RGB", (w, h), (0, 0, 0))
    bg.paste(img, ((w - img.width)//2, (h - img.height)//2))
    return bg

# -------------------------------------------------
# HANDLER
# -------------------------------------------------
def handler(job):
    data = job["input"]
    mdl = load_model()

    human = smart_resize(b64_to_img(data["human_image"]), 768, 1024)
    garment = smart_resize(b64_to_img(data["garment_image"]), 768, 1024)

    steps = data.get("steps", 30)
    seed = data.get("seed", 42)

    with torch.no_grad():
        # Parsing
        try:
            parse = mdl["parsing"](np.array(human))
            parse = np.array(parse)
            mask = (parse == 4) | (parse == 6) | (parse == 7)
            mask_img = Image.fromarray((mask * 255).astype(np.uint8))
        except:
            mask_img = Image.new("L", human.size, 0)

        # OpenPose
        try:
            raw_pose = mdl["openpose"](np.array(human)[:, :, ::-1])
            pose = raw_pose["pose_img"] if isinstance(raw_pose, dict) else raw_pose
        except:
            pose = Image.new("RGB", human.size, (0, 0, 0))

        result = mdl["pipe"](
            prompt="clothes",
            image=human,
            mask_image=mask_img,
            ip_adapter_image=garment,
            pose=pose,
            num_inference_steps=steps,
            guidance_scale=2.0,
            generator=torch.Generator(mdl["device"]).manual_seed(seed),
            height=1024,
            width=768,
        ).images[0]

    return {"output": img_to_b64(result)}

runpod.serverless.start({"handler": handler})
