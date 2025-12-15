# builder.py
import os
from huggingface_hub import snapshot_download

def download_model():
    # Model ID
    model_id = "yisol/IDM-VTON"
    
    print(f"⏳ {model_id} indiriliyor... (Sadece dosyalar çekilecek)")
    
    # Modeli cache klasörüne indir
    # allow_patterns ile sadece ihtiyacımız olanları çekiyoruz ki gereksiz şişmesin
    snapshot_download(
        repo_id=model_id,
        ignore_patterns=["*.msgpack", "*.h5", "*.safetensors"], # PyTorch bin dosyalarını alalım
        allow_patterns=["*.bin", "*.json", "*.txt", "*.png", "*config.json"]
    )
    
    print("✅ Model dosyaları başarıyla cache'lendi!")

if __name__ == "__main__":
    download_model()
