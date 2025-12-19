import os
import shutil
from huggingface_hub import hf_hub_download

def download_models():
    print("⬇️ BUILDER BAŞLIYOR: Eksik Text Encoder parçaları indiriliyor...")
    
    # İndirilecek dosyalar listesi
    tasks = [
        # --- MEVCUT PARÇALAR (Zaten Vardı) ---
        {"repo_id": "yisol/IDM-VTON", "remote": "vae/config.json", "locals": ["ckpt/vae/config.json"]},
        {"repo_id": "yisol/IDM-VTON", "remote": "vae/diffusion_pytorch_model.safetensors", "locals": ["ckpt/vae/diffusion_pytorch_model.safetensors"]},
        {"repo_id": "yisol/IDM-VTON", "remote": "unet/config.json", "locals": ["ckpt/unet/config.json"]},
        {"repo_id": "yisol/IDM-VTON", "remote": "unet/diffusion_pytorch_model.safetensors", "locals": ["ckpt/unet/diffusion_pytorch_model.safetensors"]},
        {"repo_id": "yisol/IDM-VTON", "remote": "scheduler/scheduler_config.json", "locals": ["ckpt/scheduler/scheduler_config.json"]},
        {"repo_id": "stabilityai/stable-diffusion-xl-base-1.0", "remote": "unet/config.json", "locals": ["unet_garm/config.json"]},
        {"repo_id": "stabilityai/stable-diffusion-xl-base-1.0", "remote": "unet/diffusion_pytorch_model.fp16.safetensors", "locals": ["unet_garm/diffusion_pytorch_model.safetensors"]},
        {"repo_id": "laion/CLIP-ViT-H-14-laion2B-s32B-b79K", "remote": "config.json", "locals": ["image_encoder/config.json"]},
        {"repo_id": "laion/CLIP-ViT-H-14-laion2B-s32B-b79K", "remote": "model.safetensors", "locals": ["image_encoder/model.safetensors"]},
        {"repo_id": "laion/CLIP-ViT-H-14-laion2B-s32B-b79K", "remote": "preprocessor_config.json", "locals": ["image_encoder/preprocessor_config.json"]},
        {"repo_id": "h94/IP-Adapter", "remote": "sdxl_models/ip-adapter-plus_sdxl_vit-h.bin", "locals": ["ip_adapter/adapter_model.bin"]},
        {"repo_id": "yisol/IDM-VTON", "remote": "openpose/ckpts/body_pose_model.pth", "locals": ["ckpt/openpose/ckpts/body_pose_model.pth", "preprocess/openpose/ckpts/body_pose_model.pth"]},
        {"repo_id": "yisol/IDM-VTON", "remote": "densepose/model_final_162be9.pkl", "locals": ["ckpt/densepose/densepose_model.pkl"]},
        {"repo_id": "yisol/IDM-VTON", "remote": "humanparsing/parsing_atr.onnx", "locals": ["ckpt/humanparsing/parsing_atr.onnx"]},
        {"repo_id": "yisol/IDM-VTON", "remote": "humanparsing/parsing_lip.onnx", "locals": ["ckpt/humanparsing/parsing_lip.onnx"]},

        # --- EKSİK PARÇALAR (BUNLARI EKLİYORUZ!) ---
        # 1. TOKENIZER
        {"repo_id": "yisol/IDM-VTON", "remote": "tokenizer/tokenizer_config.json", "locals": ["ckpt/tokenizer/tokenizer_config.json"]},
        {"repo_id": "yisol/IDM-VTON", "remote": "tokenizer/vocab.json", "locals": ["ckpt/tokenizer/vocab.json"]},
        {"repo_id": "yisol/IDM-VTON", "remote": "tokenizer/merges.txt", "locals": ["ckpt/tokenizer/merges.txt"]},
        {"repo_id": "yisol/IDM-VTON", "remote": "tokenizer/special_tokens_map.json", "locals": ["ckpt/tokenizer/special_tokens_map.json"]},

        # 2. TOKENIZER_2
        {"repo_id": "yisol/IDM-VTON", "remote": "tokenizer_2/tokenizer_config.json", "locals": ["ckpt/tokenizer_2/tokenizer_config.json"]},
        {"repo_id": "yisol/IDM-VTON", "remote": "tokenizer_2/vocab.json", "locals": ["ckpt/tokenizer_2/vocab.json"]},
        {"repo_id": "yisol/IDM-VTON", "remote": "tokenizer_2/merges.txt", "locals": ["ckpt/tokenizer_2/merges.txt"]},
        {"repo_id": "yisol/IDM-VTON", "remote": "tokenizer_2/special_tokens_map.json", "locals": ["ckpt/tokenizer_2/special_tokens_map.json"]},

        # 3. TEXT_ENCODER
        {"repo_id": "yisol/IDM-VTON", "remote": "text_encoder/config.json", "locals": ["ckpt/text_encoder/config.json"]},
        {"repo_id": "yisol/IDM-VTON", "remote": "text_encoder/model.safetensors", "locals": ["ckpt/text_encoder/model.safetensors"]},

        # 4. TEXT_ENCODER_2
        {"repo_id": "yisol/IDM-VTON", "remote": "text_encoder_2/config.json", "locals": ["ckpt/text_encoder_2/config.json"]},
        {"repo_id": "yisol/IDM-VTON", "remote": "text_encoder_2/model.safetensors", "locals": ["ckpt/text_encoder_2/model.safetensors"]},
    ]

    for task in tasks:
        try:
            print(f"⏳ İndiriliyor: {task['remote']}")
            path = hf_hub_download(repo_id=task["repo_id"], filename=task['remote'])
            for local in task["locals"]:
                os.makedirs(os.path.dirname(local), exist_ok=True)
                shutil.copy(path, local)
        except Exception as e:
            print(f"❌ HATA: {task['remote']} indirilemedi! Detay: {e}")
            # Text encoderlar hayati, indirmezse hata verip dursun build
            if "tokenizer" in task['remote'] or "text_encoder" in task['remote']:
                raise e

    print("✅ TÜM MODELLER (TEXT ENCODERLAR DAHİL) İNDİRİLDİ.")

if __name__ == "__main__":
    download_models()
