import os
import io
import base64
import numpy as np
import torch
import shutil
import sys
import gc
# import types <-- ARTIK YOK! Yama yapmıyoruz.
from PIL import Image, ImageOps, ImageDraw
import runpod
from torchvision import transforms 
from huggingface_hub import snapshot_download

# --- 1. AŞAMA: TAM TEMİZLİK VE HAZIRLIK ---
print("🧹 Sistem Temizleniyor (Fabrika Ayarları)...")
torch.cuda.empty_cache()
gc.collect()

# Eski, bozuk, yamalı src klasörü varsa sil. Tertipler insin.
if os.path.exists("src"):
    shutil.rmtree("src")
    print("🗑️ Eski 'src' klasörü silindi.")

print("⬇️ Orijinal Dosyalar İndiriliyor...")
# UNet ve Python scriptlerini taze indir
snapshot_download(repo_id="yisol/IDM-VTON", local_dir="ckpt", allow_patterns=["src/*", "unet/*", "tokenizer/**", "tokenizer_2/**", "*.json", "*.txt"], local_dir_use_symlinks=True)

# src klasörünü ana dizine taşı (Importlar için şart)
if os.path.exists("ckpt/src") and not os.path.exists("src"):
    shutil.move("ckpt/src", "src")
    print("✅ 'src' klasörü yerine taşındı.")

# --- IMPORTLAR ---
sys.path.append(os.getcwd())
try:
    from transformers import (CLIPImageProcessor, CLIPVisionModelWithProjection, CLIPTextModel, CLIPTextModelWithProjection, AutoTokenizer)
except: pass

# ARTIK ORİJİNAL DOSYALARI KULLANIYORUZ
# (Hacked isimli olsalar da artık içleri temiz)
from preprocess.humanparsing.run_parsing import Parsing
from preprocess.openpose.run_openpose import OpenPose
from src.tryon_pipeline import StableDiffusionXLInpaintPipeline
from src.unet_hacked_tryon import UNet2DConditionModel
from src.unet_hacked_garmnet import UNet2DConditionModel as UNetGarm

MODEL_LOADED = False
model = {}

# --- ARTIK MONKEY PATCH YOK! MODEL ORİJİNAL HALİYLE ÇALIŞACAK ---

def load_model():
    global MODEL_LOADED, model
    if MODEL_LOADED: return model

    print("🚀 Modeller Yükleniyor (Saf Mod)...")
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
    
    # Orijinal UNet'leri yüklüyoruz
    unet = UNet2DConditionModel.from_pretrained("ckpt", subfolder="unet", torch_dtype=torch.float16).to(device)
    garm_path = "unet_garm/unet" if os.path.exists("unet_garm/unet") else "unet_garm"
    unet_encoder = UNetGarm.from_pretrained(garm_path, torch_dtype=torch.float16, use_safetensors=True).to(device)

    # --- YAMA KISMI SİLİNDİ ---

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
    print("✅ Sistem Hazır! (v58)")
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
    
    print("🚀 HANDLER İŞ ALDI (v58 - ORİJİNAL MOD)")
    data = job["input"]
    try:
        mdl = load_model()
        human = smart_resize(b64_to_img(data["human_image"]), 768, 1024)
        garment = smart_resize(b64_to_img(data["garment_image"]), 768, 1024)
        garment_des = data.get("description", "clothes") 
        steps = data.get("steps", 30)
        seed = data.get("seed", 42)
        
        device = mdl["device"]
        pipe = mdl["pipe"]

        with torch.no_grad():
            # 1. Maske (Parsing Resize Fix - Bu kalmalı, faydalı)
            try:
                human_small = human.resize((384, 512))
                parse_pil = mdl["parsing"](human_small)
                parse_pil = parse_pil.resize((768, 1024), Image.NEAREST)
                parse_arr = np.array(parse_pil)
                if parse_arr.ndim > 2: parse_arr = parse_arr.squeeze()
                mask_arr = (parse_arr == 4) | (parse_arr == 6) | (parse_arr == 7)
                if np.sum(mask_arr) < 100: raise Exception("Empty Mask")
                mask_image = Image.fromarray((mask_arr * 255).astype(np.uint8))
            except:
                print("⚠️ Maske bulunamadı, fallback uygulanıyor.")
                mask_image = Image.new("L", human.size, 0)
                draw = ImageDraw.Draw(mask_image)
                w, h = human.size
                draw.rectangle([w*0.2, h*0.2, w*0.8, h*0.8], fill=255)
                
            # 2. OpenPose
            human_cv = np.array(human)[:, :, ::-1].copy() 
            pose = None
            try:
                raw = mdl["openpose"](human_cv)
                if isinstance(raw, dict) and "pose_img" in raw: pose = raw["pose_img"]
                elif isinstance(raw, Image.Image): pose = raw
            except: pass
            if pose is None: pose = Image.new("RGB", human.size, (0,0,0))
            
            pose_tensor = transforms.ToTensor()(pose).unsqueeze(0).to(device, torch.float16)
            cloth_tensor = transforms.ToTensor()(garment).unsqueeze(0).to(device, torch.float16)

            # 3. Prompt Logic (App.py'den alınan doğru mantık)
            prompt = "model is wearing " + garment_des
            negative_prompt = "monochrome, lowres, bad anatomy, worst quality, low quality"
            
            (prompt_embeds, negative_prompt_embeds, pooled_prompt_embeds, negative_pooled_prompt_embeds) = pipe.encode_prompt(
                prompt, num_images_per_prompt=1, do_classifier_free_guidance=True, negative_prompt=negative_prompt,
            )

            prompt_cloth = "a photo of " + garment_des
            (prompt_embeds_c, _, _, _) = pipe.encode_prompt(
                prompt_cloth, num_images_per_prompt=1, do_classifier_free_guidance=False, negative_prompt=negative_prompt,
            )
            
            # Pipeline Çalıştırma (Orijinal haliyle)
            output_tuple = pipe(
                prompt_embeds=prompt_embeds,
                negative_prompt_embeds=negative_prompt_embeds,
                pooled_prompt_embeds=pooled_prompt_embeds,
                negative_pooled_prompt_embeds=negative_pooled_prompt_embeds,
                num_inference_steps=steps,
                guidance_scale=2.0,
                generator=torch.Generator(device).manual_seed(seed),
                strength=1.0,
                pose_img=pose_tensor,
                text_embeds_cloth=prompt_embeds_c,
                cloth=cloth_tensor,
                mask_image=mask_image,
                image=human, 
                height=1024,
                width=768,
                ip_adapter_image=garment,
            )
            
            final_image = output_tuple[0][0]
        
        torch.cuda.empty_cache()
        return {"output": img_to_b64(final_image)}
        
    except Exception as e:
        import traceback
        trace = traceback.format_exc()
        return {"error": str(e), "trace": trace}

print("🔌 RunPod Başlatılıyor...")
runpod.serverless.start({"handler": handler})
