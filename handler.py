import runpod
import torch
import base64
import io
import os
from PIL import Image
from pathlib import Path

# Model imports (IDM-VTON için)
from diffusers import StableDiffusionInpaintPipeline
from transformers import CLIPTextModel, CLIPTokenizer
import numpy as np

print("=" * 50)
print("🚀 IDM-VTON Handler Başlatılıyor...")
print("=" * 50)

# GPU kontrolü
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"📍 Device: {device}")

# Model yükleme (global - bir kere yüklenir)
MODEL_LOADED = False
pipe = None

def load_model():
    """Model yükleme fonksiyonu"""
    global pipe, MODEL_LOADED
    
    if MODEL_LOADED:
        print("✅ Model zaten yüklü!")
        return
    
    try:
        print("📦 Model yükleniyor...")
        
        # IDM-VTON veya alternatif model
        # Buraya kendi modelinizin yükleme kodunu ekleyin
        
        # Örnek: Stable Diffusion Inpainting (basit alternatif)
        model_id = "stabilityai/stable-diffusion-2-inpainting"
        pipe = StableDiffusionInpaintPipeline.from_pretrained(
            model_id,
            torch_dtype=torch.float16 if device == "cuda" else torch.float32,
            safety_checker=None,
            requires_safety_checker=False
        )
        pipe.to(device)
        pipe.enable_attention_slicing()
        
        MODEL_LOADED = True
        print("✅ Model hazır!")
        
    except Exception as e:
        print(f"❌ Model yükleme hatası: {e}")
        raise

# Başlangıçta model yükle
load_model()

def base64_to_image(base64_string):
    """Base64 string'i PIL Image'a çevirir"""
    try:
        # data:image/jpeg;base64, prefix'ini temizle
        if ',' in base64_string:
            base64_string = base64_string.split(',')[1]
        
        img_data = base64.b64decode(base64_string)
        img = Image.open(io.BytesIO(img_data)).convert("RGB")
        return img
    except Exception as e:
        print(f"❌ Base64 decode hatası: {e}")
        raise

def image_to_base64(img):
    """PIL Image'ı base64 string'e çevirir"""
    try:
        buffered = io.BytesIO()
        img.save(buffered, format="JPEG", quality=95)
        img_str = base64.b64encode(buffered.getvalue()).decode()
        return img_str
    except Exception as e:
        print(f"❌ Image encode hatası: {e}")
        raise

def process_tryon(human_img, garment_img, garment_des="clothing item"):
    """
    Virtual try-on işlemini yapar
    
    Args:
        human_img: PIL Image - Kişinin fotoğrafı
        garment_img: PIL Image - Kıyafet fotoğrafı
        garment_des: str - Kıyafet açıklaması
    
    Returns:
        PIL Image - Try-on sonucu
    """
    try:
        print(f"🎨 Try-on işlemi başladı...")
        print(f"   Human: {human_img.size}, Garment: {garment_img.size}")
        
        # Görselleri resize et (model için optimize boyut)
        target_size = (512, 768)  # width, height
        human_img_resized = human_img.resize(target_size, Image.LANCZOS)
        garment_img_resized = garment_img.resize(target_size, Image.LANCZOS)
        
        # BURAYA GERÇEK IDM-VTON MODELİNİZİ EKLEYIN!
        # Şimdilik basit bir demo implementasyonu:
        
        # Örnek: Basit overlay (gerçek model yerine)
        # Gerçek IDM-VTON kullanıyorsanız aşağıdaki kodu değiştirin:
        
        with torch.no_grad():
            # Prompt oluştur
            prompt = f"A person wearing {garment_des}, full body, high quality, detailed"
            
            # Basit mask oluştur (üst gövde bölgesi)
            mask = Image.new("L", target_size, 255)
            
            # Model çalıştır
            if pipe is not None:
                result = pipe(
                    prompt=prompt,
                    image=human_img_resized,
                    mask_image=mask,
                    negative_prompt="blurry, low quality, distorted",
                    num_inference_steps=30,
                    guidance_scale=7.5,
                    strength=0.8
                ).images[0]
            else:
                # Fallback: Basit blend
                result = Image.blend(human_img_resized, garment_img_resized, 0.5)
        
        print("✅ Try-on tamamlandı!")
        return result
        
    except Exception as e:
        print(f"❌ Try-on hatası: {e}")
        # Hata durumunda orijinal görseli döndür
        return human_img

def handler(event):
    """
    RunPod handler fonksiyonu
    
    Expected input format:
    {
        "input": {
            "human_img": "base64_string",
            "garm_img": "base64_string",
            "garment_des": "description (optional)",
            "is_checked": true,
            "is_checked_crop": false,
            "denoise_steps": 30,
            "seed": 42
        }
    }
    """
    try:
        print("\n" + "=" * 50)
        print("📥 Yeni istek alındı!")
        print("=" * 50)
        
        # Input'ları al
        job_input = event.get('input', {})
        
        human_img_b64 = job_input.get('human_img')
        garm_img_b64 = job_input.get('garm_img')
        garment_des = job_input.get('garment_des', 'clothing item')
        denoise_steps = job_input.get('denoise_steps', 30)
        seed = job_input.get('seed', 42)
        
        # Validasyon
        if not human_img_b64 or not garm_img_b64:
            raise ValueError("human_img ve garm_img gerekli!")
        
        print(f"📋 Parametreler:")
        print(f"   - Garment desc: {garment_des}")
        print(f"   - Denoise steps: {denoise_steps}")
        print(f"   - Seed: {seed}")
        
        # Seed ayarla
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
        
        # Base64'ten PIL Image'a çevir
        print("🔄 Görseller decode ediliyor...")
        human_img = base64_to_image(human_img_b64)
        garm_img = base64_to_image(garm_img_b64)
        
        # Try-on işlemi
        print("🎨 Try-on işlemi başlatılıyor...")
        result_img = process_tryon(human_img, garm_img, garment_des)
        
        # Sonucu base64'e çevir
        print("📤 Sonuç encode ediliyor...")
        result_b64 = image_to_base64(result_img)
        
        print("✅ İşlem başarıyla tamamlandı!")
        print("=" * 50 + "\n")
        
        return {
            "output": result_b64,
            "status": "success"
        }
        
    except Exception as e:
        error_msg = f"Handler hatası: {str(e)}"
        print(f"❌ {error_msg}")
        import traceback
        traceback.print_exc()
        
        return {
            "error": error_msg,
            "status": "failed"
        }

# RunPod serverless başlat
if __name__ == "__main__":
    print("\n" + "=" * 50)
    print("🎯 RunPod Serverless Handler Hazır!")
    print("=" * 50 + "\n")
    
    runpod.serverless.start({"handler": handler})
