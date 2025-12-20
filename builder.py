import os
import shutil
from huggingface_hub import hf_hub_download

def download_models():
    print("⬇️ BUILDER BAŞLIYOR: Bozuk ONNX dosyaları onarılıyor...")
    
    # 1. TEMİZLİK: Eğer humanparsing klasörü varsa SİL (Çünkü içindeki dosya bozuk)
    parsing_path = "ckpt/humanparsing"
    if os.path.exists(parsing_path):
        print(f"🗑️ Bozuk olma ihtimaline karşı klasör siliniyor: {parsing_path}")
        shutil.rmtree(parsing_path)
    
    # Klasörleri yeniden oluştur
    os.makedirs(parsing_path, exist_ok=True)

    # İndirilecek dosyalar listesi
    tasks = [
        # --- BOZUK OLAN DOSYALAR (Bunları kesin indiriyoruz) ---
        {"repo_id": "yisol/IDM-VTON", "remote": "humanparsing/parsing_atr.onnx", "locals": ["ckpt/humanparsing/parsing_atr.onnx"]},
        {"repo_id": "yisol/IDM-VTON", "remote": "humanparsing/parsing_lip.onnx", "locals": ["ckpt/humanparsing/parsing_lip.onnx"]},

        # --- DİĞER GEREKLİ DOSYALAR (Eksikse indirir) ---
        {"repo_id": "yisol/IDM-VTON", "remote": "densepose/model_final_162be9.pkl", "locals": ["ckpt/densepose/densepose_model.pkl"]},
        {"repo_id": "yisol/IDM-VTON", "remote": "openpose/ckpts/body_pose_model.pth", "locals": ["ckpt/openpose/ckpts/body_pose_model.pth", "preprocess/openpose/ckpts/body_pose_model.pth"]},
        
        # --- ANA MODELLER (.bin veya .safetensors fark etmez, burayı kendi tercihine göre düzenle) ---
        # Eğer .bin kullanıyorsan onları buraya ekle, ben standartları bırakıyorum:
        {"repo_id": "yisol/IDM-VTON", "remote": "vae/config.json", "locals": ["ckpt/vae/config.json"]},
        {"repo_id": "yisol/IDM-VTON", "remote": "vae/diffusion_pytorch_model.safetensors", "locals": ["ckpt/vae/diffusion_pytorch_model.safetensors"]},
        {"repo_id": "yisol/IDM-VTON", "remote": "unet/config.json", "locals": ["ckpt/unet/config.json"]},
        {"repo_id": "yisol/IDM-VTON", "remote": "unet/diffusion_pytorch_model.safetensors", "locals": ["ckpt/unet/diffusion_pytorch_model.safetensors"]},
        
        # TOKENIZER VE TEXT ENCODERLAR (Hayati Önemli)
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
        
        # IP-Adapter & Image Encoder
        {"repo_id": "h94/IP-Adapter", "remote": "sdxl_models/ip-adapter-plus_sdxl_vit-h.bin", "locals": ["ip_adapter/adapter_model.bin"]},
        {"repo_id": "laion/CLIP-ViT-H-14-laion2B-s32B-b79K", "remote": "config.json", "locals": ["image_encoder/config.json"]},
        {"repo_id": "laion/CLIP-ViT-H-14-laion2B-s32B-b79K", "remote": "model.safetensors", "locals": ["image_encoder/model.safetensors"]},
        {"repo_id": "laion/CLIP-ViT-H-14-laion2B-s32B-b79K", "remote": "preprocessor_config.json", "locals": ["image_encoder/preprocessor_config.json"]},
        
        # GarmNet
        {"repo_id": "stabilityai/stable-diffusion-xl-base-1.0", "remote": "unet/config.json", "locals": ["unet_garm/config.json"]},
        {"repo_id": "stabilityai/stable-diffusion-xl-base-1.0", "remote": "unet/diffusion_pytorch_model.fp16.safetensors", "locals": ["unet_garm/diffusion_pytorch_model.safetensors"]},
    ]

    for task in tasks:
        # Önce dosya var mı kontrol et, varsa ve boyutu çok küçükse (bozuksa) sil
        for local in task["locals"]:
            if os.path.exists(local) and os.path.getsize(local) < 2000: # 2KB'dan küçükse kesin bozuktur (LFS pointer)
                print(f"⚠️ Dosya çok küçük (Bozuk olabilir), siliniyor: {local}")
                os.remove(local)

        try:
            print(f"⏳ İndiriliyor: {task['remote']}")
            path = hf_hub_download(repo_id=task["repo_id"], filename=task['remote'])
            for local in task["locals"]:
                os.makedirs(os.path.dirname(local), exist_ok=True)
                if not os.path.exists(local): # Sadece yoksa kopyala
                    shutil.copy(path, local)
        except Exception as e:
            print(f"❌ HATA: {task['remote']} indirilemedi! Detay: {e}")
            raise e # Hata varsa dur, devam etme

    print("✅ ONARIM TAMAMLANDI. MODELLER HAZIR.")

if __name__ == "__main__":
    download_models()
