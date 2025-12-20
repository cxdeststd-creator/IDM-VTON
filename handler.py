import os
import io
import base64
import numpy as np
import torch
import subprocess
import shutil
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
BASE_PATH = "ckpt"

def check_and_fix_model(local_path, url, min_size_mb=100):
    """
    Dosyayı kontrol eder, bozuksa CURL ile indirir.
    """
    needs_download = False
    
    # 1. Kontrol
    if os.path.exists(local_path):
        size_mb = os.path.getsize(local_path) / (1024 * 1024)
        if size_mb < min_size_mb:
            print(f"⚠️ BOZUK DOSYA: {local_path} ({size_mb:.2f} MB). Siliniyor...")
            try: os.remove(local_path)
            except: pass
            needs_download = True
        else:
            print(f"✅ Dosya Sağlam: {local_path} ({size_mb:.2f} MB)")
    else:
        print(f"🔍 Dosya Yok: {local_path}")
        needs_download = True

    # 2. İndirme
    if needs_download:
        print(f"⬇️ İNDİRİLİYOR (CURL): {local_path}")
        os.makedirs(os.path.dirname(local_path), exist_ok=True)
        try:
            subprocess.run(["curl", "-L", "-o", local_path, url], check=True)
            print(f"🎉 İndirme Tamamlandı!")
        except Exception as e:
            print(f"❌ İNDİRME HATASI: {e}")
            raise e

def load_model():
    global MODEL_LOADED, model
    if MODEL_LOADED: return model

    print("🔧 SİSTEM KONTROLÜ VE ONARIMI BAŞLIYOR...")

    # --- 1. HUMAN PARSING (ONNX) ---
    check_and_fix_model(
        "ckpt/humanparsing/parsing_atr.onnx",
        "https://huggingface.co/yisol/IDM-VTON/resolve/main/humanparsing/parsing_atr.onnx?download=true",
        min_size_mb=200 
    )
    check_and_fix_model(
        "ckpt/humanparsing/parsing_lip.onnx",
        "https://huggingface.co/yisol/IDM-VTON/resolve/main/humanparsing/parsing_lip.onnx?download=true",
        min_size_mb=50
    )

    # --- 2. DENSEPOSE ---
    check_and_fix_model(
        "ckpt/densepose/densepose_model.pkl",
        "https://huggingface.co/yisol/IDM-VTON/resolve/main/densepose/model_final_162be9.pkl?download=true",
        min_size_mb=50
    )

    # --- 3. OPENPOSE (KRİTİK DÜZELTME) ---
    # OpenPose kodu bazen "preprocess/..." klasörüne bazen "ckpt/..." klasörüne bakıyor.
    # İşimi garantiye alıp ana dosyayı indiriyorum, sonra diğer yere kopyalıyorum.
    
    target_path = "ckpt/openpose/ckpts/body_pose_model.pth"
    backup_path = "preprocess/openpose/ckpts/body_pose_model.pth"
    url = "https://huggingface.co/yisol/IDM-VTON/resolve/main/openpose/ckpts/body_pose_model.pth?download=true"

    # Önce ana hedefi kontrol et/indir
    check_and_fix_model(target_path, url, min_size_mb=150)

    # Sonra diğer klasöre kopyala (OpenPose kütüphanesi oraya bakıyorsa patlamasın)
    if not os.path.exists(backup_path) or os.path.getsize(backup_path) < 150 * 1024 * 1024:
        print(f"🔄 OpenPose yedeği kopyalanıyor: {backup_path}")
        os.makedirs(os.path.dirname(backup_path), exist_ok=True)
        shutil.copy(target_path, backup_path)
    # -------------------------------------

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"🔄 Modeller Yükleniyor... Device: {device}")

    from preprocess.humanparsing.run_parsing import Parsing
    from preprocess.openpose.run_openpose import OpenPose
    from src.tryon_pipeline import StableDiffusionXLInpaintPipeline
    from src.unet_hacked_tryon import UNet2DConditionModel
    from src.unet_hacked_garmnet import UNet2DConditionModel as UNetGarm

    # Artık dosya sağlam, hata vermeyecek
    parsing = Parsing(0)
    openpose = OpenPose(0)

    # Modülleri yükle
    tokenizer = AutoTokenizer.from_pretrained("ckpt/tokenizer")
    tokenizer_2 = AutoTokenizer.from_pretrained("ckpt/tokenizer_2")
    text_encoder = CLIPTextModel.from_pretrained("ckpt/text_encoder", torch_dtype=torch.float16).to(device)
    text_encoder_2 = CLIPTextModelWithProjection.from_pretrained("ckpt/text_encoder_2", torch_dtype=torch.float16).to(device)

    image_encoder = CLIPVisionModelWithProjection.from_pretrained("image_encoder", torch_dtype=torch.float16).to(device)
    feature_extractor = CLIPImageProcessor.from_pretrained("image_encoder", torch_dtype=torch.float16)
    
    unet = UNet2DConditionModel.from_pretrained("ckpt/unet", torch_dtype=torch.float16).to(device)
    unet_encoder = UNetGarm.from_pretrained("unet_garm", torch_dtype=torch.float16, use_safetensors=True).to(device)

    pipe = StableDiffusionXLInpaintPipeline.from_pretrained(
        BASE_PATH, 
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
    print("✅ Sistem Hazır! (OpenPose Fix Applied)")
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
    print("🚀 HANDLER ÇALIŞIYOR (v17)")
    data = job["input"]
    
    human = smart_resize(b64_to_img(data["human_image"]), 768, 1024)
    garment = smart_resize(b64_to_img(data["garment_image"]), 768, 1024)
    
    steps = data.get("steps", 30)
    seed = data.get("seed", 42)
    
    try:
        mdl = load_model()
    except Exception as e:
        import traceback
        return {"error": f"Yükleme Hatası: {str(e)}", "trace": traceback.format_exc()}

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
            pose = Image.new("RGB", human.size, (0,0,0))
        
        # 3. PIPELINE
        device = mdl["device"]
        dtype = torch.float16
        
        pose_tensor = transforms.ToTensor()(pose).unsqueeze(0).to(device, dtype=dtype)
        cloth_tensor = transforms.ToTensor()(garment).unsqueeze(0).to(device, dtype=dtype)
        
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
