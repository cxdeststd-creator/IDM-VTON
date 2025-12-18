# handler.py - EXIF ve AKILLI RESİZE FİX
import os
import io
import base64
import numpy as np
import torch
from PIL import Image, ImageOps
import runpod
from transformers import CLIPImageProcessor, CLIPVisionModelWithProjection

# Global Değişkenler
MODEL_LOADED = False
model = {}
BASE_REPO = "yisol/IDM-VTON"

def load_model():
    global MODEL_LOADED, model
    if MODEL_LOADED: return model

    # ensure_ckpts() YOK - Builder halletti.
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"🔄 Modeller yükleniyor... Device: {device}")

    from preprocess.humanparsing.run_parsing import Parsing
    from preprocess.openpose.run_openpose import OpenPose
    from src.tryon_pipeline import StableDiffusionXLInpaintPipeline
    from src.unet_hacked_tryon import UNet2DConditionModel
    from src.unet_hacked_garmnet import UNet2DConditionModel as UNetGarm

    # Yardımcı Modeller
    parsing = Parsing(0)
    openpose = OpenPose(0)

    # Ana Modeller
    image_encoder = CLIPVisionModelWithProjection.from_pretrained("image_encoder", torch_dtype=torch.float16)
    feature_extractor = CLIPImageProcessor.from_pretrained("image_encoder", torch_dtype=torch.float16)
    unet = UNet2DConditionModel.from_pretrained(BASE_REPO, subfolder="unet", torch_dtype=torch.float16)
    unet_garm = UNetGarm.from_pretrained("unet_garm", torch_dtype=torch.float16, use_safetensors=True)

    pipe = StableDiffusionXLInpaintPipeline.from_pretrained(
        BASE_REPO,
        unet=unet,
        image_encoder=image_encoder,
        feature_extractor=feature_extractor,
        torch_dtype=torch.float16,
        use_safetensors=True,
    ).to(device)
    
    pipe.unet_garm = unet_garm

    model = {"pipe": pipe, "parsing": parsing, "openpose": openpose, "device": device}
    MODEL_LOADED = True
    print("✅ Sistem Hazır!")
    return model

# --- YENİ HELPER FONKSİYONLAR ---

def smart_resize(img, width, height):
    """
    Resmi sündürmeden (aspect ratio bozmadan) hedef boyuta getirir.
    Gerekirse kenarları siyahla doldurur.
    """
    # Önce EXIF bilgisini düzelt (Yan duran fotoları dikelt)
    img = ImageOps.exif_transpose(img)
    
    # Orantılı resize
    img.thumbnail((width, height), Image.Resampling.LANCZOS)
    
    # Yeni siyah tuval oluştur
    background = Image.new('RGB', (width, height), (0, 0, 0))
    
    # Resmi ortaya yapıştır
    offset_x = (width - img.width) // 2
    offset_y = (height - img.height) // 2
    background.paste(img, (offset_x, offset_y))
    
    return background

def b64_to_img(b64):
    if "," in b64: b64 = b64.split(",")[1]
    image = Image.open(io.BytesIO(base64.b64decode(b64))).convert("RGB")
    # Yükleme anında da EXIF kontrolü yapalım
    return ImageOps.exif_transpose(image)

def img_to_b64(img):
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=95)
    return base64.b64encode(buf.getvalue()).decode()

# --- HANDLER ---
def handler(job):
    data = job["input"]
    
    # 1. Resimleri Yükle ve Akıllı Boyutlandır
    # Eğer resmi sündürürsek (basıklaştırırsa) model insanı tanıyamaz.
    raw_human = b64_to_img(data["human_image"])
    human = smart_resize(raw_human, 768, 1024)

    raw_garment = b64_to_img(data["garment_image"])
    garment = smart_resize(raw_garment, 768, 1024)
    
    steps = data.get("steps", 30)
    seed = data.get("seed", 42)
    
    mdl = load_model()

    with torch.no_grad():
        # --- A. PARSING & MASKE ---
        print("Run Parsing...")
        try:
            parse_pil = mdl["parsing"](human)
            parse_arr = np.array(parse_pil)
            # 4:Üst, 6:Ceket, 7:Elbise
            mask_arr = (parse_arr == 4) | (parse_arr == 6) | (parse_arr == 7)
            mask_image = Image.fromarray((mask_arr * 255).astype(np.uint8))
        except Exception as e:
            print(f"⚠️ Maske hatası: {e}")
            mask_image = Image.new("L", human.size, 0)

        # --- B. OPENPOSE ---
        print("Run OpenPose...")
        # OpenPose BGR ister, RGB değil!
        human_cv = np.array(human)[:, :, ::-1].copy() 
        
        try:
            pose = mdl["openpose"](human_cv)
        except Exception as e:
            print(f"⚠️ OpenPose Hatası: {e}")
            pose = None

        if pose is None:
            print("⚠️ UYARI: İnsan bulunamadı, boş pose veriliyor.")
            pose = Image.new("RGB", human.size, (0,0,0))

        # --- C. PIPELINE ---
        print("Run Pipeline...")
        result = mdl["pipe"](
            prompt="clothes",
            image=human,
            mask_image=mask_image,       
            ip_adapter_image=garment,    
            pose=pose,                   
            num_inference_steps=steps,
            guidance_scale=2.0,
            generator=torch.Generator(mdl["device"]).manual_seed(seed),
            height=1024,
            width=768
        ).images[0]

    return {"output": img_to_b64(result)}

runpod.serverless.start({"handler": handler})
