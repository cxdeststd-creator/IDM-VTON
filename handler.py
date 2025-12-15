import runpod
import torch
import base64
import io
import os
import cv2
import numpy as np
from PIL import Image
from pathlib import Path

# --- GLOBAL AYARLAR ---
device = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_LOADED = False
model = {}

print("=" * 50)
print(f"🚀 IDM-VTON Handler Başlatılıyor... (Device: {device})")
print("=" * 50)

def load_idm_vton_model():
    """
    IDM-VTON modelini yükler.
    Hata olursa gizlemez, direkt patlar (ki loglarda görelim).
    """
    global model, MODEL_LOADED
    
    if MODEL_LOADED:
        return

    print("📦 IDM-VTON Modelleri RAM'e yükleniyor...")

    try:
        # 1. Gerekli kütüphaneleri import et (Repo yapısı bozuksa burada patlar)
        from preprocess.humanparsing.run_parsing import Parsing
        from preprocess.openpose.run_openpose import OpenPose
        from src.tryon_pipeline import StableDiffusionXLInpaintPipeline as TryonPipeline
        from src.unet_hacked_garmnet import UNet2DConditionModel as UNet2DConditionModel_ref
        from src.unet_hacked_tryon import UNet2DConditionModel
        
        # 2. Model Yolları (Cache'den çekecek)
        base_path = 'yisol/IDM-VTON'
        
        # 3. Yardımcı Modelleri Yükle (Parsing & OpenPose)
        # Checkpointleri indirmediyse burada hata verebilir, o yüzden builder.py önemliydi.
        parsing_model = Parsing(0) # GPU 0
        openpose_model = OpenPose(0)
        
        # 4. UNet Modellerini Yükle
        print("⏳ UNet modelleri yükleniyor...")
        unet = UNet2DConditionModel.from_pretrained(
            base_path,
            subfolder="unet",
            torch_dtype=torch.float16,
        )
        unet.requires_grad_(False)
        
        unet_garm = UNet2DConditionModel_ref.from_pretrained(
            base_path,
            subfolder="unet_garm",
            torch_dtype=torch.float16,
        )
        unet_garm.requires_grad_(False)
        
        # 5. Ana Pipeline'ı Kur
        print("⏳ Pipeline kuruluyor...")
        pipe = TryonPipeline.from_pretrained(
            base_path,
            unet=unet,
            torch_dtype=torch.float16,
            use_safetensors=True, 
            variant="fp16" # Eğer fp16 yoksa bunu silip deneyebiliriz ama genelde var
        ).to(device)
        
        pipe.unet_garm = unet_garm
        
        # 6. Global değişkene ata
        model['pipe'] = pipe
        model['parsing'] = parsing_model
        model['openpose'] = openpose_model
        
        MODEL_LOADED = True
        print("✅ IDM-VTON Model başarıyla yüklendi!")

    except Exception as e:
        print("\n" + "!"*50)
        print(f"❌ KRİTİK HATA: Model Yüklenemedi!")
        print(f"Hata Detayı: {str(e)}")
        import traceback
        traceback.print_exc()
        print("!"*50 + "\n")
        # Burada raise diyoruz ki container hatayı yutmasın, restart atsın.
        raise e

# Container başlarken modeli yükle (Cold Start burada olur)
load_idm_vton_model()

def base64_to_image(base64_string):
    """Base64 string'i PIL Image'a çevirir"""
    if ',' in base64_string:
        base64_string = base64_string.split(',')[1]
    img_data = base64.b64decode(base64_string)
    image = Image.open(io.BytesIO(img_data))
    return image.convert("RGB")

def image_to_base64(img):
    """PIL Image'ı Base64 string'e çevirir"""
    buffered = io.BytesIO()
    img.save(buffered, format="JPEG", quality=95)
    return base64.b64encode(buffered.getvalue()).decode('utf-8')

def handler(event):
    """RunPod'dan gelen isteği işleyen fonksiyon"""
    try:
        print("\n📥 Yeni istek geldi!")
        job_input = event.get("input")
        
        if not job_input:
            return {"error": "Input verisi boş!"}

        # Parametreleri al
        human_img_b64 = job_input.get("human_image") # API'de human_image bekliyoruz
        garm_img_b64 = job_input.get("garment_image") # API'de garment_image bekliyoruz
        garment_des = job_input.get("description", "clothing")
        denoise_steps = job_input.get("steps", 30)
        seed = job_input.get("seed", 42)
        
        if not human_img_b64 or not garm_img_b64:
            return {"error": "human_image ve garment_image zorunludur!"}

        # Resimleri Decode Et
        human_img = base64_to_image(human_img_b64)
        garm_img = base64_to_image(garm_img_b64)
        
        # Boyutlandırma (IDM-VTON 768x1024 sever)
        human_img = human_img.resize((768, 1024))
        garm_img = garm_img.resize((768, 1024))

        # Seed ayarla
        generator = torch.Generator(device).manual_seed(seed)

        # İşlemi Başlat
        print("🎨 Görüntü işleniyor (Parsing + OpenPose)...")
        
        with torch.no_grad():
            # 1. Parsing ve Pose
            human_img_np = np.array(human_img)
            parsing_result = model['parsing'](human_img_np)
            pose_result = model['openpose'](human_img_np)
            
            # 2. Try-on Pipeline
            print(f"🔄 Difüzyon modeli çalışıyor ({denoise_steps} steps)...")
            
            # (mask parametresi optional, auto-masking kullanıyoruz)
            result_images = model['pipe'](
                prompt=garment_des,
                image=human_img,
                garment=garm_img,
                pose=pose_result,
                num_inference_steps=denoise_steps,
                guidance_scale=2.0,
                generator=generator,
                height=1024,
                width=768,
            ).images

        final_image = result_images[0]
        
        # Sonucu Encode Et
        output_b64 = image_to_base64(final_image)
        
        print("✅ İşlem Başarılı!")
        return {"output": output_b64}

    except Exception as e:
        print(f"❌ Handler Hatası: {str(e)}")
        import traceback
        traceback.print_exc()
        return {"error": str(e)}

if __name__ == "__main__":
    runpod.serverless.start({"handler": handler})
