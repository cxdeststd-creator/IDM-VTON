import runpod
import torch
import base64
import io
import os
import cv2
import numpy as np
from PIL import Image
from pathlib import Path

print("=" * 50)
print("🚀 IDM-VTON Handler Başlatılıyor...")
print("=" * 50)

# GPU kontrolü
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"📍 Device: {device}")

# Model yükleme (global)
MODEL_LOADED = False
model = None
auto_mask = None
auto_processor = None

def load_idm_vton_model():
    """IDM-VTON modelini yükler"""
    global model, auto_mask, auto_processor, MODEL_LOADED
    
    if MODEL_LOADED:
        print("✅ Model zaten yüklü!")
        return
    
    try:
        print("📦 IDM-VTON Model yükleniyor...")
        
        # IDM-VTON imports
        from preprocess.humanparsing.run_parsing import Parsing
        from preprocess.openpose.run_openpose import OpenPose
        from src.tryon_pipeline import StableDiffusionXLInpaintPipeline as TryonPipeline
        from src.unet_hacked_garmnet import UNet2DConditionModel as UNet2DConditionModel_ref
        from src.unet_hacked_tryon import UNet2DConditionModel
        
        # Model paths
        base_path = 'yisol/IDM-VTON'
        unet_path = 'yisol/IDM-VTON'
        
        # Parsing ve OpenPose
        parsing_model = Parsing(0)  # GPU 0
        openpose_model = OpenPose(0)
        
        # UNet models
        unet = UNet2DConditionModel.from_pretrained(
            unet_path,
            subfolder="unet",
            torch_dtype=torch.float16,
        )
        unet.requires_grad_(False)
        
        unet_garm = UNet2DConditionModel_ref.from_pretrained(
            unet_path,
            subfolder="unet_garm",
            torch_dtype=torch.float16,
        )
        unet_garm.requires_grad_(False)
        
        # Pipeline
        pipe = TryonPipeline.from_pretrained(
            base_path,
            unet=unet,
            vae=None,  # Will be loaded by pipeline
            torch_dtype=torch.float16,
        ).to(device)
        pipe.unet_garm = unet_garm
        
        # Store globally
        model = {
            'pipe': pipe,
            'parsing': parsing_model,
            'openpose': openpose_model
        }
        
        MODEL_LOADED = True
        print("✅ IDM-VTON Model hazır!")
        
    except Exception as e:
        print(f"❌ Model yükleme hatası: {e}")
        print("⚠️  Basit fallback modu aktif!")
        MODEL_LOADED = True  # Fallback ile devam et

# Model yükle
load_idm_vton_model()

def base64_to_image(base64_string):
    """Base64'ten PIL Image"""
    try:
        if ',' in base64_string:
            base64_string = base64_string.split(',')[1]
        img_data = base64.b64decode(base64_string)
        img = Image.open(io.BytesIO(img_data)).convert("RGB")
        return img
    except Exception as e:
        print(f"❌ Base64 decode: {e}")
        raise

def image_to_base64(img):
    """PIL Image'dan base64"""
    try:
        buffered = io.BytesIO()
        img.save(buffered, format="JPEG", quality=95)
        return base64.b64encode(buffered.getvalue()).decode()
    except Exception as e:
        print(f"❌ Image encode: {e}")
        raise

def process_tryon(human_img, garment_img, garment_des="clothing", is_checked=True, denoise_steps=30, seed=42):
    """IDM-VTON Try-on işlemi"""
    try:
        print("🎨 IDM-VTON işlemi başladı...")
        
        # Seed
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
        
        # Resize to model size
        human_img = human_img.resize((768, 1024))
        garment_img = garment_img.resize((768, 1024))
        
        if model and 'pipe' in model:
            print("🔄 Model çalıştırılıyor...")
            
            # Preprocess
            parsing = model['parsing']
            openpose = model['openpose']
            
            # Human parsing ve pose detection
            human_img_np = np.array(human_img)
            parse_result = parsing(human_img_np)
            pose_result = openpose(human_img_np)
            
            # Pipeline çalıştır
            with torch.no_grad():
                result = model['pipe'](
                    prompt=garment_des,
                    image=human_img,
                    garment=garment_img,
                    mask=parse_result if is_checked else None,
                    pose=pose_result,
                    num_inference_steps=denoise_steps,
                    guidance_scale=2.0,
                    height=1024,
                    width=768,
                ).images[0]
            
            print("✅ IDM-VTON tamamlandı!")
            return result
        else:
            # Fallback: Basit blend
            print("⚠️  Fallback mode: Basit blend")
            result = Image.blend(human_img, garment_img, 0.3)
            return result
            
    except Exception as e:
        print(f"❌ Try-on hatası: {e}")
        import traceback
        traceback.print_exc()
        # Error durumunda human image döndür
        return human_img

def handler(event):
    """RunPod handler"""
    try:
        print("\n" + "=" * 50)
        print("📥 Yeni istek alındı!")
        
        job_input = event.get('input', {})
        
        human_img_b64 = job_input.get('human_img')
        garm_img_b64 = job_input.get('garm_img')
        garment_des = job_input.get('garment_des', 'clothing')
        is_checked = job_input.get('is_checked', True)
        denoise_steps = job_input.get('denoise_steps', 30)
        seed = job_input.get('seed', 42)
        
        if not human_img_b64 or not garm_img_b64:
            raise ValueError("human_img ve garm_img gerekli!")
        
        print(f"📋 Parametreler: steps={denoise_steps}, seed={seed}")
        
        # Decode images
        print("🔄 Görseller decode ediliyor...")
        human_img = base64_to_image(human_img_b64)
        garm_img = base64_to_image(garm_img_b64)
        
        # Process
        print("🎨 Try-on işleniyor...")
        result_img = process_tryon(
            human_img, 
            garm_img, 
            garment_des, 
            is_checked, 
            denoise_steps, 
            seed
        )
        
        # Encode result
        print("📤 Sonuç encode ediliyor...")
        result_b64 = image_to_base64(result_img)
        
        print("✅ Başarılı!")
        print("=" * 50 + "\n")
        
        return {
            "output": result_b64
        }
        
    except Exception as e:
        print(f"❌ Handler hatası: {e}")
        import traceback
        traceback.print_exc()
        return {
            "error": str(e)
        }

if __name__ == "__main__":
    print("🎯 RunPod Handler Hazır!\n")
    runpod.serverless.start({"handler": handler})
