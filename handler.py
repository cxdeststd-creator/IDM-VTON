import os
import io
import json
import base64
import shutil
import numpy as np
import torch
from PIL import Image
from huggingface_hub import hf_hub_download
import runpod

# Global Değişkenler
MODEL_LOADED = False
model = {}
BASE_REPO = "yisol/IDM-VTON"

# -------------------------------------------------
# 1. DOSYA İNDİRME (Eksik Parçaları Tamamlama)
# -------------------------------------------------
def ensure_ckpts():
    print("⬇️ Model dosyaları (UNet Garm dahil) indiriliyor...")
    
    tasks = [
        # --- 1. UNET GARM (StabilityAI'dan alıp ismini değiştiriyoruz) ---
        {
            "repo_id": "stabilityai/stable-diffusion-xl-base-1.0",
            "remote": "unet/config.json",
            "locals": ["unet_garm/config.json"] # Klasör adını unet_garm yapıyoruz
        },
        {
            "repo_id": "stabilityai/stable-diffusion-xl-base-1.0",
            "remote": "unet/diffusion_pytorch_model.fp16.safetensors",
            "locals": ["unet_garm/diffusion_pytorch_model.safetensors"] 
        },

        # --- 2. IMAGE ENCODER (Laion'dan) ---
        {
            "repo_id": "laion/CLIP-ViT-H-14-laion2B-s32B-b79K",
            "remote": "config.json",
            "locals": ["image_encoder/config.json"]
        },
        {
            "repo_id": "laion/CLIP-ViT-H-14-laion2B-s32B-b79K",
            "remote": "model.safetensors", 
            "locals": ["image_encoder/model.safetensors"]
        },

        # --- 3. IP ADAPTER (h94'ten) ---
        {
            "repo_id": "h94/IP-Adapter",
            "remote": "sdxl_models/ip-adapter-plus_sdxl_vit-h.bin",
            "locals": ["ip_adapter/adapter_model.bin"]
        },

        # --- 4. OPENPOSE (Yisol - Çift Dikiş) ---
        {
            "repo_id": "yisol/IDM-VTON",
            "remote": "openpose/ckpts/body_pose_model.pth",
            "locals": [
                "ckpt/openpose/ckpts/body_pose_model.pth",
                "preprocess/openpose/ckpts/body_pose_model.pth"
            ]
        },
        
        # --- 5. DENSEPOSE & HUMANPARSING (Yisol) ---
        {
            "repo_id": "yisol/IDM-VTON",
            "remote": "densepose/model_final_162be9.pkl",
            "locals": ["ckpt/densepose/densepose_model.pkl"]
        },
        {
            "repo_id": "yisol/IDM-VTON",
            "remote": "humanparsing/parsing_atr.onnx",
            "locals": ["ckpt/humanparsing/parsing_atr.onnx"]
        },
        {
            "repo_id": "yisol/IDM-VTON",
            "remote": "humanparsing/parsing_lip.onnx",
            "locals": ["ckpt/humanparsing/parsing_lip.onnx"]
        }
    ]

    for task in tasks:
        repo_id = task["repo_id"]
        remote_path = task["remote"]
        local_paths = task["locals"]

        try:
            print(f"⏳ İndiriliyor ({repo_id}): {remote_path}")
            downloaded_file_path = hf_hub_download(
                repo_id=repo_id,
                filename=remote_path
            )
        except Exception as e:
            print(f"❌ HATA: {remote_path} indirilemedi! Detay: {e}")
            raise e

        for local_path in local_paths:
            os.makedirs(os.path.dirname(local_path), exist_ok=True)
            shutil.copy(downloaded_file_path, local_path)
            print(f"✅ Hazır: {local_path}")

# -------------------------------------------------
# 2. MODEL LOAD (DÜZELTİLEN KISIM BURASI)
# -------------------------------------------------
def load_model():
    global MODEL_LOADED, model

    if MODEL_LOADED:
        return model

    # Önce dosyaları indir
    ensure_ckpts()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Importları burada yapıyoruz (Dosya yapına uygun)
    from preprocess.humanparsing.run_parsing import Parsing
    from preprocess.openpose.run_openpose import OpenPose
    from src.tryon_pipeline import StableDiffusionXLInpaintPipeline
    from src.unet_hacked_tryon import UNet2DConditionModel
    from src.unet_hacked_garmnet import UNet2DConditionModel as UNetGarm

    print("🔄 Modeller yükleniyor...")
    
    parsing = Parsing(0)
    openpose = OpenPose(0)

    # Ana UNet (Hala Yisol'dan çekebilir, sorun yok)
    unet = UNet2DConditionModel.from_pretrained(
        BASE_REPO,
        subfolder="unet",
        torch_dtype=torch.float16
    )

    # --- KRİTİK DÜZELTME BURADA YAPILDI ---
    # Eski hali: BASE_REPO ve subfolder="unet_garm" kullanıyordu (ve patlıyordu).
    # Yeni hali: Direkt yerel klasör adını ("unet_garm") veriyoruz.
    print("🔄 UNet Garm yerel klasörden yükleniyor...")
    unet_garm = UNetGarm.from_pretrained(
        "unet_garm",  # <-- Yerel klasör adı
        torch_dtype=torch.float16,
        use_safetensors=True # Çünkü safetensors indirdik
    )

    # Pipeline oluşturma
    print("🔄 Pipeline oluşturuluyor...")
    pipe = StableDiffusionXLInpaintPipeline.from_pretrained(
        BASE_REPO,
        unet=unet,
        torch_dtype=torch.float16,
        use_safetensors=True
    ).to(device)

    # GarmNet'i pipeline'a monte et
    pipe.unet_garm = unet_garm

    model = {
        "pipe": pipe,
        "parsing": parsing,
        "openpose": openpose,
        "device": device
    }

    MODEL_LOADED = True
    print("✅ Tüm modeller başarıyla yüklendi!")
    return model

# -------------------------------------------------
# HELPERS
# -------------------------------------------------
def b64_to_img(b64):
    if "," in b64:
        b64 = b64.split(",")[1]
    return Image.open(io.BytesIO(base64.b64decode(b64))).convert("RGB")

def img_to_b64(img):
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=95)
    return base64.b64encode(buf.getvalue()).decode()

# -------------------------------------------------
# HANDLER
# -------------------------------------------------
def handler(job):
    data = job["input"]

    # Gelen resimleri işle
    human = b64_to_img(data["human_image"]).resize((768, 1024))
    garment = b64_to_img(data["garment_image"]).resize((768, 1024))

    steps = data.get("steps", 30)
    seed = data.get("seed", 42)

    # Modeli al
    mdl = load_model()

    # İşlem yap
    with torch.no_grad():
        human_np = np.array(human)
        
        # Preprocess
        parsing = mdl["parsing"](human_np)
        pose = mdl["openpose"](human_np)

        # Try-On
        result = mdl["pipe"](
            prompt="clothes",
            image=human,
            garment=garment,
            pose=pose,
            num_inference_steps=steps,
            guidance_scale=2.0,
            generator=torch.Generator(mdl["device"]).manual_seed(seed),
            height=1024,
            width=768
        ).images[0]

    return {"output": img_to_b64(result)}

runpod.serverless.start({"handler": handler})
