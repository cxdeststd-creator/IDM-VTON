import os
import shutil
import subprocess
from huggingface_hub import hf_hub_download

def run_command(command):
    """Linux komutunu zorla çalıştırır"""
    try:
        subprocess.run(command, shell=True, check=True)
    except subprocess.CalledProcessError as e:
        print(f"❌ KOMUT HATASI: {e}")
        raise e

def force_download(url, local_path):
    """
    1. Varsa siler.
    2. WGET ile zorla indirir.
    """
    # 1. TEMİZLİK: Dosya varsa acıma, sil.
    if os.path.exists(local_path):
        print(f"🗑️ Eski dosya siliniyor: {local_path}")
        os.remove(local_path)
    
    # Klasörü yarat
    os.makedirs(os.path.dirname(local_path), exist_ok=True)

    print(f"⬇️ ZORLA İNDİRİLİYOR: {local_path}")
    
    # 2. İNDİRME: wget komutu (Cache yok, LFS hatası yok)
    # -O: Dosyayı şuraya kaydet
    cmd = f"wget -O '{local_path}' '{url}'"
    run_command(cmd)
    
    # Kontrol: İndi mi?
    if os.path.getsize(local_path) < 10000:
        raise Exception(f"🚨 Dosya indi ama boyutu çok küçük! ({os.path.getsize(local_path)} bytes). Link bozuk olabilir.")

def download_models():
    print("🚧 BUILDER BAŞLIYOR: BALYOZ MODU 🚧")

    # --- 1. BOZUK OLAN ONNX DOSYALARI (Zorla İndirilecekler) ---
    # Bu linkler HuggingFace'in "Raw" linkleridir. Git LFS araya giremez.
    
    force_download(
        "https://huggingface.co/yisol/IDM-VTON/resolve/main/humanparsing/parsing_atr.onnx?download=true",
        "ckpt/humanparsing/parsing_atr.onnx"
    )

    force_download(
        "https://huggingface.co/yisol/IDM-VTON/resolve/main/humanparsing/parsing_lip.onnx?download=true",
        "ckpt/humanparsing/parsing_lip.onnx"
    )

    force_download(
        "https://huggingface.co/yisol/IDM-VTON/resolve/main/densepose/model_final_162be9.pkl?download=true",
        "ckpt/densepose/densepose_model.pkl"
    )

    # --- 2. DİĞER STANDART DOSYALAR (Bunlar zaten çalışıyordu, elleşme) ---
    print("🔄 Diğer standart modeller kontrol ediliyor...")
    
    tasks = [
        {"repo_id": "yisol/IDM-VTON", "remote": "openpose/ckpts/body_pose_model.pth", "locals": ["ckpt/openpose/ckpts/body_pose_model.pth", "preprocess/openpose/ckpts/body_pose_model.pth"]},
        
        # ANA MOTORLAR
        {"repo_id": "yisol/IDM-VTON", "remote": "vae/config.json", "locals": ["ckpt/vae/config.json"]},
        {"repo_id": "yisol/IDM-VTON", "remote": "vae/diffusion_pytorch_model.safetensors", "locals": ["ckpt/vae/diffusion_pytorch_model.safetensors"]},
        {"repo_id": "yisol/IDM-VTON", "remote": "unet/config.json", "locals": ["ckpt/unet/config.json"]},
        {"repo_id": "yisol/IDM-VTON", "remote": "unet/diffusion_pytorch_model.safetensors", "locals": ["ckpt/unet/diffusion_pytorch_model.safetensors"]},
        {"repo_id": "yisol/IDM-VTON", "remote": "scheduler/scheduler_config.json", "locals": ["ckpt/scheduler/scheduler_config.json"]},

        # TEXT ENCODERLAR
        {"repo_id": "yisol/IDM-VTON", "remote": "tokenizer/tokenizer_config.json", "locals": ["ckpt/tokenizer/tokenizer_config.json"]},
        {"repo_id": "yisol/IDM-VTON", "remote": "tokenizer/vocab.json", "locals": ["ckpt/tokenizer/vocab.json"]},
        {"repo_id": "yisol/IDM-VTON", "remote": "tokenizer/merges.txt", "locals": ["ckpt/tokenizer/merges.txt"]},
        {"repo_id": "yisol/IDM-VTON", "remote": "tokenizer/special_tokens_map.json", "locals": ["ckpt/tokenizer/special_tokens_map.json"]},
        {"repo_id": "yisol/IDM-VTON", "remote": "tokenizer_2/tokenizer_config.json", "locals": ["ckpt/tokenizer_2/tokenizer_config.json"]},
        {"repo_id": "yisol/IDM-VTON", "remote": "tokenizer_2/vocab.json", "locals": ["ckpt/tokenizer_2/vocab.json"]},
        {"repo_id": "yisol/IDM-VTON", "remote": "tokenizer_2/merges.txt", "locals": ["ckpt/tokenizer_2/merges.txt"]},
        {"repo_id": "yisol/IDM-VTON", "remote": "tokenizer_2/special_tokens_map.json", "locals": ["ckpt/tokenizer_2/special_tokens_map.json"]},
        {"repo_id": "yisol/IDM-VTON", "remote": "text_encoder/config.json", "locals": ["ckpt/text_encoder/config.json"]},
        {"repo_id": "yisol/IDM-VTON", "remote": "text_encoder/model.safetensors", "locals": ["ckpt/text_encoder/model.safetensors"]},
        {"repo_id": "yisol/IDM-VTON", "remote": "text_encoder_2/config.json", "locals": ["ckpt/text_encoder_2/config.json"]},
        {"repo_id": "yisol/IDM-VTON", "remote": "text_encoder_2/model.safetensors", "locals": ["ckpt/text_encoder_2/model.safetensors"]},
        
        # IP-ADAPTER & GARM
        {"repo_id": "h94/IP-Adapter", "remote": "sdxl_models/ip-adapter-plus_sdxl_vit-h.bin", "locals": ["ip_adapter/adapter_model.bin"]},
        {"repo_id": "laion/CLIP-ViT-H-14-laion2B-s32B-b79K", "remote": "config.json", "locals": ["image_encoder/config.json"]},
        {"repo_id": "laion/CLIP-ViT-H-14-laion2B-s32B-b79K", "remote": "model.safetensors", "locals": ["image_encoder/model.safetensors"]},
        {"repo_id": "laion/CLIP-ViT-H-14-laion2B-s32B-b79K", "remote": "preprocessor_config.json", "locals": ["image_encoder/preprocessor_config.json"]},
        {"repo_id": "stabilityai/stable-diffusion-xl-base-1.0", "remote": "unet/config.json", "locals": ["unet_garm/config.json"]},
        {"repo_id": "stabilityai/stable-diffusion-xl-base-1.0", "remote": "unet/diffusion_pytorch_model.fp16.safetensors", "locals": ["unet_garm/diffusion_pytorch_model.safetensors"]},
    ]

    for task in tasks:
        try:
            path = hf_hub_download(repo_id=task["repo_id"], filename=task['remote'])
            for local in task["locals"]:
                os.makedirs(os.path.dirname(local), exist_ok=True)
                if not os.path.exists(local):
                    shutil.copy(path, local)
        except Exception as e:
            print(f"❌ HATA: {task['remote']} indirilemedi! Detay: {e}")
            raise e

    print("✅ BALYOZ OPERASYONU TAMAMLANDI. MODELLER KAYA GİBİ.")

if __name__ == "__main__":
    download_models()
