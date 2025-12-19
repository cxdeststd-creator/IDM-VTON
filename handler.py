import os
import io
import base64
import numpy as np
import torch
from PIL import Image, ImageOps
import runpod
from torchvision import transforms 
from transformers import (
    CLIPImageProcessor, 
    CLIPVisionModelWithProjection,
    CLIPTextModel,
    CLIPTextModelWithProjection,
    AutoTokenizer
)

MODEL_LOADED = False
model = {}
BASE_REPO = "yisol/IDM-VTON"

def load_model():
    global MODEL_LOADED, model
    if MODEL_LOADED: return model

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"🔄 Modeller YERELDEN (ckpt/) yükleniyor... Device: {device}")

    from preprocess.humanparsing.run_parsing import Parsing
    from preprocess.openpose.run_openpose import OpenPose
    from src.tryon_pipeline import StableDiffusionXLInpaintPipeline
    from src.unet_hacked_tryon import UNet2DConditionModel
    from src.unet_hacked_garmnet import UNet2DConditionModel as UNetGarm

    parsing = Parsing(0)
    openpose = OpenPose(0)

    # --- 1. METİN MODÜLLERİ (YERELDEN) ---
    # Artık "ckpt/..." klasöründen çekiyoruz, internet yok!
    tokenizer = AutoTokenizer.from_pretrained("ckpt/tokenizer")
    tokenizer_2 = AutoTokenizer.from_pretrained("ckpt/tokenizer_2")
    
    text_encoder = CLIPTextModel.from_pretrained("ckpt/text_encoder", torch_dtype=torch.float16).to(device)
    text_encoder_2 = CLIPTextModelWithProjection.from_pretrained("ckpt/text_encoder_2", torch_dtype=torch.float16).to(device)

    # 2. Görüntü Modülleri
    # Bunları zaten builder indiriyordu ama klasör yollarını yerel yapalım garanti olsun
    image_encoder = CLIPVisionModelWithProjection.from_pretrained("image_encoder", torch_dtype=torch.float16).to(device)
    feature_extractor = CLIPImageProcessor.from_pretrained("image_encoder", torch_dtype=torch.float16)
    
    # 3. UNet Modülleri
    uunet = UNet2DConditionModel.from_pretrained("ckpt/unet", torch_dtype=torch.float16, use_safetensors=True).to(device)
    unet_encoder = UNetGarm.from_pretrained("unet_garm", torch_dtype=torch.float16, use_safetensors=True).to(device)

    # 4. Pipeline Kurulumu (FULL PAKET)
    # BASE_REPO yerine "ckpt" veya lokal yol veriyoruz ki internete bakmasın
    pipe = StableDiffusionXLInpaintPipeline.from_pretrained(
        "ckpt", 
        unet=unet,
        unet_encoder=unet_encoder,
        image_encoder=image_encoder,
        feature_extractor=feature_extractor,
        text_encoder=text_encoder,
        text_encoder_2=text_encoder_2,
        tokenizer=tokenizer,
        tokenizer_2=tokenizer_2,
        torch_dtype=torch.float16,
        use_safetensors=True,
    ).to(device)
    
    pipe.register_modules(unet_encoder=unet_encoder)

    model = {"pipe": pipe, "parsing": parsing, "openpose": openpose, "device": device}
    MODEL_LOADED = True
    print("✅ Sistem Hazır! (Full Local Load)")
    return model

# --- HELPER ---
def b64_to_img(b64):
    if "," in b64: b64 = b64.split(",")[1]
    return ImageOps.exif_transpose(Image.open(io.BytesIO(base64.b64decode(b64))).convert("RGB"))

def img_to_b64(img):
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=95)
    return base64.b64encode(buf.getvalue()).decode()

def smart_resize(img, width, height):
    img = ImageOps.exif_transpose(img)
    img.thumbnail((width, height), Image.Resampling.LANCZOS)
    background = Image.new('RGB', (width, height), (0, 0, 0))
    background.paste(img, ((width - img.width) // 2, (height - img.height) // 2))
    return background

# --- HANDLER ---
def handler(job):
    print("🚀 GÜNCEL KOD BAŞLADI v14 (Builder Fix)")
    data = job["input"]
    
    human = smart_resize(b64_to_img(data["human_image"]), 768, 1024)
    garment = smart_resize(b64_to_img(data["garment_image"]), 768, 1024)
    
    steps = data.get("steps", 30)
    seed = data.get("seed", 42)
    mdl = load_model()

    with torch.no_grad():
        # 1. PARSING
        try:
            parse_pil = mdl["parsing"](human)
            parse_arr = np.array(parse_pil)
            if parse_arr.ndim > 2: parse_arr = parse_arr.squeeze()
            mask_arr = (parse_arr == 4) | (parse_arr == 6) | (parse_arr == 7)
            mask_image = Image.fromarray((mask_arr * 255).astype(np.uint8))
        except:
            mask_image = Image.new("L", human.size, 0)

        # 2. OPENPOSE
        human_cv = np.array(human)[:, :, ::-1].copy() 
        pose = None
        try:
            raw = mdl["openpose"](human_cv)
            if isinstance(raw, dict) and "pose_img" in raw:
                pose = raw["pose_img"]
            elif isinstance(raw, Image.Image):
                pose = raw
        except:
            pass
            
        if pose is None:
            print("⚠️ OpenPose boş döndü, siyah pose kullanılıyor.")
            pose = Image.new("RGB", human.size, (0,0,0))
        
        # --- TENSOR HAZIRLIK ---
        print("🔄 Tensor Dönüşümleri...")
        device = mdl["device"]
        dtype = torch.float16
        
        pose_tensor = transforms.ToTensor()(pose).unsqueeze(0).to(device, dtype=dtype)
        cloth_tensor = transforms.ToTensor()(garment).unsqueeze(0).to(device, dtype=dtype)
        
        # 3. PIPELINE
        result = mdl["pipe"](
            prompt="clothes",
            image=human,
            mask_image=mask_image,
            ip_adapter_image=garment, 
            cloth=cloth_tensor,       
            pose_img=pose_tensor,
            num_inference_steps=steps,
            guidance_scale=2.0,
            generator=torch.Generator(device).manual_seed(seed),
            height=1024,
            width=768
        ).images[0]

    return {"output": img_to_b64(result)}

runpod.serverless.start({"handler": handler})
