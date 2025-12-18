# builder.py
import os
import shutil
from huggingface_hub import hf_hub_download

def download_models():
    print("⬇️ BUILDER: Modeller indiriliyor...")
    
    tasks = [
        # 1. UNET GARM
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
        # 2. IMAGE ENCODER
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
        # 3. IP ADAPTER
        {
            "repo_id": "h94/IP-Adapter",
            "remote": "sdxl_models/ip-adapter-plus_sdxl_vit-h.bin",
            "locals": ["ip_adapter/adapter_model.bin"]
        },
        # 4. OPENPOSE
        {
            "repo_id": "yisol/IDM-VTON",
            "remote": "openpose/ckpts/body_pose_model.pth",
            "locals": ["ckpt/openpose/ckpts/body_pose_model.pth", "preprocess/openpose/ckpts/body_pose_model.pth"]
        },
        # 5. DENSEPOSE & PARSING
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
        try:
            print(f"⏳ İndiriliyor: {task['remote']}")
            path = hf_hub_download(repo_id=task["repo_id"], filename=task["remote"])
            for local in task["locals"]:
                os.makedirs(os.path.dirname(local), exist_ok=True)
                shutil.copy(path, local)
        except Exception as e:
            print(f"❌ HAATA: {e}")
            raise e

if __name__ == "__main__":
    download_models()
