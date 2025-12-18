# builder.py - Sadece Build sırasında çalışır
import os
import shutil
from huggingface_hub import hf_hub_download

def download_models():
    print("⬇️ BUILDER: Model dosyaları indiriliyor...")
    
    tasks = [
        # --- 1. UNET GARM (StabilityAI -> Yerel unet_garm) ---
        {
            "repo_id": "stabilityai/stable-diffusion-xl-base-1.0",
            "remote": "unet/config.json",
            "locals": ["unet_garm/config.json"]
        },
        {
            "repo_id": "stabilityai/stable-diffusion-xl-base-1.0",
            "remote": "unet/diffusion_pytorch_model.fp16.safetensors",
            "locals": ["unet_garm/diffusion_pytorch_model.safetensors"] 
        },

        # --- 2. IMAGE ENCODER & FEATURE EXTRACTOR (Laion) ---
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
        {
            "repo_id": "laion/CLIP-ViT-H-14-laion2B-s32B-b79K",
            "remote": "preprocessor_config.json",   
            "locals": ["image_encoder/preprocessor_config.json"]
        },

        # --- 3. IP ADAPTER (h94) ---
        {
            "repo_id": "h94/IP-Adapter",
            "remote": "sdxl_models/ip-adapter-plus_sdxl_vit-h.bin",
            "locals": ["ip_adapter/adapter_model.bin"]
        },

        # --- 4. OPENPOSE (Yisol) ---
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
            # Build sırasında hata olursa işlem dursun
            raise e

        for local_path in local_paths:
            os.makedirs(os.path.dirname(local_path), exist_ok=True)
            shutil.copy(downloaded_file_path, local_path)
            print(f"✅ Hazır: {local_path}")

if __name__ == "__main__":
    download_models()
