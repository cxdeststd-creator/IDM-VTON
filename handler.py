import os
import io
import base64
import numpy as np
import torch
import shutil
import sys
import gc
from PIL import Image, ImageOps
import runpod
from torchvision import transforms 
from huggingface_hub import snapshot_download

# --- 1. DOSYA TAMİRİ (KESME ve BİÇME) ---
def force_fix_file_with_slicer():
    target_file = "src/unet_hacked_garmnet.py"
    if not os.path.exists(target_file):
        return

    print("🔧 [v51] Dosya yamalanıyor (OTOMATİK BOYUTLANDIRICI)...")
    with open(target_file, "r") as f:
        lines = f.readlines()

    if any("FIXED_BY_GEMINI_V51" in line for line in lines):
        return

    new_lines = []
    fixed = False
    search_text = 'if "text_embeds" not in added_cond_kwargs:'
    
    for line in lines:
        if search_text in line and not fixed:
            indent = line.split('if')[0]
            new_lines.append(f'{indent}# FIXED_BY_GEMINI_V51 (THE SLICER)\n')
            new_lines.append(f'{indent}if added_cond_kwargs is None: added_cond_kwargs = {{}}\n')
            
            # --- KRİTİK MÜDAHALE: ENCODER HIDDEN STATES ---
            new_lines.append(f'{indent}if encoder_hidden_states is not None:\n')
            new_lines.append(f'{indent}    curr_dim = encoder_hidden_states.shape[-1]\n')
            
            # DURUM 1: Veri ÇOK BÜYÜK (3072 geldiyse -> 2048'e KES)
            new_lines.append(f'{indent}    if curr_dim > 2048:\n')
            new_lines.append(f'{indent}        encoder_hidden_states = encoder_hidden_states[:, :, :2048]\n')
            
            # DURUM 2: Veri KÜÇÜK (640 geldiyse -> 2048'e DOLDUR)
            new_lines.append(f'{indent}    elif curr_dim < 2048:\n')
            new_lines.append(f'{indent}        pad = 2048 - curr_dim\n')
            new_lines.append(f'{indent}        encoder_hidden_states = torch.nn.functional.pad(encoder_hidden_states, (0, pad))\n')

            # TEXT EMBEDS (Bunu da garantiye alıp 1280 yapıyoruz)
            new_lines.append(f'{indent}if "text_embeds" not in added_cond_kwargs:\n')
            new_lines.append(f'{indent}    added_cond_kwargs["text_embeds"] = torch.zeros((1, 1280), device=sample.device, dtype=sample.dtype)\n')
            
            new_lines.append(f'{indent}if "time_ids" not in added_cond_kwargs:\n')
            new_lines.append(f'{indent}    added_cond_kwargs["time_ids"] = torch.zeros((1, 6), device=sample.device, dtype=sample.dtype)\n')
            
            fixed = True
        new_lines.append(line)
            
    if fixed:
        with open(target_file, "w") as f:
            f.writelines(new_lines)
        print("✅ Dosya başarıyla kaydedildi (Slicer Aktif).")

# --- 2. HAZIRLIK ---
print("⬇️ Sistem Hazırlanıyor...")
torch.cuda.empty_cache()

snapshot_download(repo_id="yisol/IDM-VTON", local_dir="ckpt", allow_patterns=["*.py", "unet/*", "tokenizer/**", "tokenizer_2/**", "*.json", "*.txt"], local_dir_use_symlinks=True)

# Yamayı uygula
force_fix_file_with_slicer()

# --- 3. IMPORTLAR ---
sys.path.append(os.getcwd())

try:
    from transformers import (
        CLIPImageProcessor, 
        CLIPVisionModelWithProjection,
        CLIPTextModel,
        CLIPTextModelWithProjection,
        AutoTokenizer
    )
except ImportError:
    pass

from preprocess.humanparsing.run_parsing import Parsing
from preprocess.openpose.run_openpose import OpenPose
from src.tryon_pipeline import StableDiffusionXLInpaintPipeline
from src.unet_hacked_tryon import UNet2DConditionModel
from src.unet_hacked_garmnet import UNet2DConditionModel as UNetGarm

MODEL_LOADED = False
model = {}

def load_model():
    global MODEL_LOADED, model
    if MODEL_LOADED: return model

    print("🚀 Modeller Yükleniyor...")
    snapshot_download(repo_id="yisol/IDM-VTON", local_dir="ckpt", allow_patterns=["text_encoder/*", "text_encoder_2/*", "vae/*", "scheduler/*", "humanparsing/*", "densepose/*", "openpose/*"], local_dir_use_symlinks=True)
    snapshot_download(repo_id="laion/CLIP-ViT-H-14-laion2B-s32B-b79K", local_dir="image_encoder", allow_patterns=["*.json", "*.safetensors", "*.txt"], local_dir_use_symlinks=True)
    snapshot_download(repo_id="stabilityai/stable-diffusion-xl-base-1.0", local_dir="unet_garm", allow_patterns=["unet/*", "*.json"], local_dir_use_symlinks=True)

    src_pose = "ckpt/openpose/ckpts/body_pose_model.pth"
    dst_pose = "preprocess/openpose/ckpts/body_pose_model.pth"
    if os.path.exists(src_pose) and not os.path.exists(dst_pose):
        os.makedirs(os.path.dirname(dst_pose), exist_ok=True)
        try: shutil.copy(src_pose, dst_pose)
        except: pass

    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    tokenizer = AutoTokenizer.from_pretrained("ckpt", subfolder="tokenizer", use_fast=False)
    tokenizer_2 = AutoTokenizer.from_pretrained("ckpt", subfolder="tokenizer_2", use_fast=False)
    
    text_encoder = CLIPTextModel.from_pretrained("ckpt", subfolder="text_encoder", torch_dtype=torch.float16).to(device)
    text_encoder_2 = CLIPTextModelWithProjection.from_pretrained("ckpt", subfolder="text_encoder_2", torch_dtype=torch.float16).to(device)
    image_encoder = CLIPVisionModelWithProjection.from_pretrained("image_encoder", torch_dtype=torch.float16).to(device)
    feature_extractor = CLIPImageProcessor.from_pretrained("image_encoder", torch_dtype=torch.float16)
    
    unet = UNet2DConditionModel.from_pretrained("ckpt", subfolder="unet", torch_dtype=torch.float16).to(device)
    
    garm_path = "unet_garm/unet" if os.path.exists("unet_garm/unet") else "unet_garm"
    unet_encoder = UNetGarm.from_pretrained(garm_path, torch_dtype=torch.float16, use_safetensors=True).to(device)

    pipe = StableDiffusionXLInpaintPipeline.from_pretrained(
        "ckpt", unet=unet, unet_encoder=unet_encoder, image_encoder=image_encoder,
        feature_extractor=feature_extractor, text_encoder=text_encoder, text_encoder_2=text_encoder_2,
        tokenizer=tokenizer, tokenizer_2=tokenizer_2, torch_dtype=torch.float16, use_safetensors=True,
    ).to(device)
    pipe.register_modules(unet_encoder=unet_encoder)

    parsing = Parsing(0)
    openpose = OpenPose(0)

    model = {"pipe": pipe, "parsing": parsing, "openpose": openpose, "device": device}
    MODEL_LOADED = True
    print("✅ Sistem Hazır! (v51)")
    return model

def b64_to_img(b64):
    if "," in b64: b64 = b64.split(",")[1]
    return ImageOps.exif_transpose(Image.open(io.BytesIO(base64.b64decode(b64))).convert("RGB"))

def img_to_b64(img):
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=95)
    return base64.b64encode(buf.getvalue()).decode()

def smart_resize(img, width, height):
    img = ImageOps.exif_transpose(img)
    img.thumbnail((width, height), Image.Resampling.LANCZOS)
    background = Image.new('RGB', (width, height), (0, 0, 0))
    background.paste(img, ((width - img.width) // 2, (height - img.height) // 2))
    return background

def handler(job):
    gc.collect()
    torch.cuda.empty_cache()
    
    print("🚀 HANDLER ÇALIŞIYOR (v51)")
    data = job["input"]
    try:
        mdl = load_model()
        human = smart_resize(b64_to_img(data["human_image"]), 768, 1024)
        garment = smart_resize(b64_to_img(data["garment_image"]), 768, 1024)
        steps = data.get("steps", 30)
        seed = data.get("seed", 42)
        
        with torch.no_grad():
            try:
                parse_pil = mdl["parsing"](human)
                parse_arr = np.array(parse_pil)
                if parse_arr.ndim > 2: parse_arr = parse_arr.squeeze()
                mask_arr = (parse_arr == 4) | (parse_arr == 6) | (parse_arr == 7)
                mask_image = Image.fromarray((mask_arr * 255).astype(np.uint8))
            except:
                mask_image = Image.new("L", human.
