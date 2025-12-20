import os
import shutil
import requests # <--- BUNA İHTİYACIMIZ VAR (Standart kütüphanedir)
from huggingface_hub import hf_hub_download

# --- YENİ FONKSİYON: URL'den Direkt İndirici ---
def download_direct(url, save_path):
    print(f"🔗 Direkt İndiriliyor: {save_path} \n   -> Kaynak: {url}")
    
    # Klasörü yarat
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    # Varsa ve boyutu küçükse (LFS pointer ise) sil
    if os.path.exists(save_path) and os.path.getsize(save_path) < 5000:
        print("🗑️ Bozuk/Küçük dosya tespit edildi, siliniyor...")
        os.remove(save_path)
    
    # Zaten büyük dosya varsa indirme (Cache mantığı)
    if os.path.exists(save_path) and os.path.getsize(save_path) > 100000:
        print("✅ Dosya zaten sağlam, atlanıyor.")
        return

    # Dosyayı indir
    with requests.get(url, stream=True) as r:
        r.raise_for_status()
        with open(save_path, 'wb') as f:
            for chunk in r.iter_content(chunk_size=8192): 
                f.write(chunk)
    print("✅ İndirme Tamamlandı.")

def download_models():
    print("⬇️ BUILDER BAŞLIYOR: ONNX Dosyaları Manuel İndiriliyor...")
    
    # --- 1. BOZUK OLAN DOSYALAR (DİREKT LİNK İLE) ---
    # HuggingFace'in "resolve/main" linkleri LFS'yi bypass eder, direkt dosyayı verir.
    
    # Parsing ATR
    download_direct(
        "https://huggingface.co/yisol/IDM-VTON/resolve/main/humanparsing/parsing_atr.onnx?download=true", 
        "ckpt/humanparsing/parsing_atr.onnx"
    )
    
    # Parsing LIP
    download_direct(
        "https://huggingface.co/yisol/IDM-VTON/resolve/main/humanparsing/parsing_lip.onnx?download=true", 
        "ckpt/humanparsing/parsing_lip.onnx"
    )
    
    # DensePose (Bu da bazen sorun çıkarır, elle indirelim)
    download_direct(
        "https://huggingface.co/yisol/IDM-VTON/resolve/main/densepose/model_final_162be9.pkl?download=true",
        "ckpt/densepose/densepose_model.pkl"
    )

    # --- 2. DİĞER STANDART DOSYALAR (Bunlarda sorun yoktu, HF ile devam) ---
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
            # print(f"⏳ Kontrol ediliyor: {task['remote']}") # Log kirliliği olmasın
            path = hf_hub_download(repo_id=task["repo_id"], filename=task['remote'])
            for local in task["locals"]:
                os.makedirs(os.path.dirname(local), exist_ok=True)
                if not os.path.exists(local):
                    shutil.copy(path, local)
        except Exception as e:
            print(f"❌ HATA: {task['remote']} indirilemedi! Detay: {e}")
            raise e

    print("✅ TÜM MODELLER (ONNX DAHİL) SAĞLAM İNDİRİLDİ.")

if __name__ == "__main__":
    download_models()
