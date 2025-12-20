import os
import io
import base64
import numpy as np
import torch
import subprocess
import shutil
import time
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

# --- EVRENSEL İNDİRİCİ ---
def ensure_file(local_path, url, min_size_mb=0):
    is_valid = False
    if os.path.exists(local_path):
        size_mb = os.path.getsize(local_path) / (1024 * 1024)
        # Sadece büyük dosyalar için boyut kontrolü yap (Configler 1KB olabilir, onları silmesin)
        if min_size_mb > 0 and size_mb < min_size_mb:
            print(f"⚠️ DOSYA EKSİK/BOZUK: {local_path} ({size_mb:.2f} MB). Siliniyor...")
            try: os.remove(local_path)
            except: pass
        else:
            is_valid = True

    if not is_valid:
        print(f"⬇️ İNDİRİLİYOR: {local_path}")
        os.makedirs(os.path.dirname(local_path), exist_ok=True)
        try:
            subprocess.run(["curl", "-L", "-o", local_path, "--retry", "3", url], check=True)
            print(f"🎉 İndi: {local_path}")
        except Exception as e:
            print(f"❌ İNDİRME HATASI: {e} -> {url}")
            raise e

def load_model():
    global MODEL_LOADED, model
    if MODEL_LOADED: return model

    print("🔧 SİSTEM FULL CONFIG TARAMASI (v24)...")
    base_url = "https://huggingface.co/yisol/IDM-VTON/resolve/main"

    # ==========================================
    # 1. KÜÇÜK CONFIG DOSYALARI (EKSİKSİZ LİSTE)
    # ==========================================
    # Hata veren tokenizer_config.json dahil HEPSİ burada.
    
    config_files = [
        # TOKENIZER 1
        ("ckpt/tokenizer/tokenizer_config.json", f"{base_url}/tokenizer/tokenizer_config.json?download=true"),
        ("ckpt/tokenizer/special_tokens_map.json", f"{base_url}/tokenizer/special_tokens_map.json?download=true"),
        ("ckpt/tokenizer/vocab.json", f"{base_url}/tokenizer/vocab.json?download=true"),
        ("ckpt/tokenizer/merges.txt", f"{base_url}/tokenizer/merges.txt?download=true"),

        # TOKENIZER 2
        ("ckpt/tokenizer_2/tokenizer_config.json", f"{base_url}/tokenizer_2/tokenizer_config.json?download=true"),
        ("ckpt/tokenizer_2/special_tokens_map.json", f"{base_url}/tokenizer_2/special_tokens_map.json?download=true"),
        ("ckpt/tokenizer_2/vocab.json", f"{base_url}/tokenizer_2/vocab.json?download=true"),
        ("ckpt/tokenizer_2/merges.txt", f"{base_url}/tokenizer_2/merges.txt?download=true"),

        # TEXT ENCODER 1 & 2 CONFIGS
        ("ckpt/text_encoder/config.json", f"{base_url}/text_encoder/config.json?download=true"),
        ("ckpt/text_encoder_2/config.json", f"{base_url}/text_encoder_2/config.json?download=true"),

        # VAE & SCHEDULER CONFIGS
        ("ckpt/vae/config.json", f"{base_url}/vae/config.json?download=true"),
        ("ckpt/scheduler/scheduler_config.json", f"{base_url}/scheduler/scheduler_config.json?download=true"),

        # UNET CONFIG
        ("ckpt/unet/config.json", f"{base_url}/unet/config.json?download=true"),
    ]

    print(f"📦 {len(config_files)} adet Config dosyası kontrol ediliyor...")
    for local, url in config_files:
        ensure_file(local, url, 0) # 0 MB limit, çünkü bunlar küçük dosyalar


    # ==========================================
    # 2. BÜYÜK MODELLER (AĞIR ABİLER)
    # ==========================================
    
    # Text Encoders
    ensure_file("ckpt/text_encoder/model.safetensors", f"{base_url}/text_encoder/model.safetensors?download=true", 200)
    ensure_file("ckpt/text_encoder_2/model.safetensors", f"{base_url}/text_encoder_2/model.safetensors?download=true", 200)
    
    # VAE
    ensure_file("ckpt/vae/diffusion_pytorch_model.safetensors", f"{base_url}/vae/diffusion_pytorch_model.safetensors?download=true", 100)
    
    # UNET (Sadece .bin kullanıyoruz, safetensors belasını siliyoruz)
    if os.path.exists("ckpt/unet/diffusion_pytorch_model.safetensors"):
        print("🧹 Bozuk safetensors temizleniyor...")
        os.remove("ckpt/unet/diffusion_pytorch_model.safetensors")
    
    # UNET .bin indir (1GB altıysa bozuktur)
    ensure_file("ckpt/unet/diffusion_pytorch_model.bin", f"{base_url}/unet/diffusion_pytorch_model.bin?download=true", 1000)

    # Image Encoder (LAION)
    laion_url = "https://huggingface.co/laion/CLIP-ViT-H-14-laion2B-s32B-b79K/resolve/main"
    ensure_file("image_encoder/config.json", f"{laion_url}/config.json?download=true", 0)
    ensure_file("image_encoder/preprocessor_config.json", f"{laion_url}/preprocessor_config.json?download=true", 0)
    ensure_file("image_encoder/model.safetensors", f"{laion_url}/model.safetensors?download=true", 500)
    
    # UNET GARM (SDXL Base)
    sdxl_url = "https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0/resolve/main/unet"
    ensure_file("unet_garm/config.json", f"{sdxl_url}/config.json?download=true", 0)
    ensure_file("unet_garm/diffusion_pytorch_model.safetensors", f"{sdxl_url}/diffusion_pytorch_model.fp16.safetensors?download=true", 1000)

    # 3. YAN MODELLER (Parsing, OpenPose, DensePose)
    ensure_file("ckpt/humanparsing/parsing_atr.onnx", f"{base_url}/humanparsing/parsing_atr.onnx?download=true", 200)
    ensure_file("ckpt/humanparsing/parsing_lip.onnx", f"{base_url}/humanparsing/parsing_lip.onnx?download=true", 50)
    ensure_file("ckpt/densepose/densepose_model.pkl", f"{base_url}/densepose/model_final_162be9.pkl?download=true", 50)
    
    # OpenPose Fix
    target_pose = "ckpt/openpose/ckpts/body_pose_model.pth"
    ensure_file(target_pose, f"{base_url}/openpose/ckpts/body_pose_model.pth?download=true", 150)
    
    # Preprocess yedeği
    backup_pose = "preprocess/openpose/ckpts/body_pose_model.pth"
    if not os.path.exists(backup_pose):
        os.makedirs(os.path.dirname(backup_pose), exist_ok=True)
        shutil.copy(target_pose, backup_pose)

    # --- YÜKLEME ---
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"🔄 Modeller Yükleniyor... Device: {device}")

    from preprocess.humanparsing.run_parsing import Parsing
    from preprocess.openpose.run_openpose import OpenPose
    from src.tryon_pipeline import StableDiffusionXLInpaintPipeline
    from src.unet_hacked_tryon import UNet2DConditionModel
    from src.unet_hacked_garmnet import UNet2DConditionModel as UNetGarm

    parsing = Parsing(0)
    try: openpose = OpenPose(0)
    except: 
        if os.path.exists(target_pose): os.remove(target_pose)
        raise RuntimeError("OpenPose yüklenemedi. Dosya silindi. Tekrar deneyin.")

    # Tokenizer configleri artık diskte VAR. Hata vermeyecek.
    tokenizer = AutoTokenizer.from_pretrained("ckpt/tokenizer")
    tokenizer_2 = AutoTokenizer.from_pretrained("ckpt/tokenizer_2")
    
    text_encoder = CLIPTextModel.from_pretrained("ckpt/text_encoder", torch_dtype=torch.float16).to(device)
    text_encoder_2 = CLIPTextModelWithProjection.from_pretrained("ckpt/text_encoder_2", torch_dtype=torch.float16).to(device)
    
    image_encoder = CLIPVisionModelWithProjection.from_pretrained("image_encoder", torch_dtype=torch.float16).to(device)
    feature_extractor = CLIPImageProcessor.from_pretrained("image_encoder", torch_dtype=torch.float16)
    
    # UNET (use_safetensors=False ile .bin formatına zorluyoruz)
    unet = UNet2DConditionModel.from_pretrained("ckpt/unet", torch_dtype=torch.float16, use_safetensors=False).to(device)
    
    unet_encoder = UNetGarm.from_pretrained("unet_garm", torch_dtype=torch.float16, use_safetensors=True).to(device)

    pipe = StableDiffusionXLInpaintPipeline.from_pretrained(
        BASE_PATH, # ckpt klasörü (içinde model_index.json var!)
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
    print("✅ Sistem Hazır! (v24 - ALL CONFIGS)")
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
    print("🚀 HANDLER ÇALIŞIYOR (v24)")
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
        try:
            parse_pil = mdl["parsing"](human)
            parse_arr = np.array(parse_pil)
            if parse_arr.ndim > 2: parse_arr = parse_arr.squeeze()
            mask_arr = (parse_arr == 4) | (parse_arr == 6) | (parse_arr == 7)
            mask_image = Image.fromarray((mask_arr * 255).astype(np.uint8))
        except:
            mask_image = Image.new("L", human.size, 0)

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
