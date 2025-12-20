import os
import io
import base64
import numpy as np
import torch
import shutil
from PIL import Image, ImageOps
import runpod
from torchvision import transforms 
from huggingface_hub import snapshot_download

from transformers import (
    CLIPImageProcessor, 
    CLIPVisionModelWithProjection,
    CLIPTextModel,
    CLIPTextModelWithProjection,
    AutoTokenizer
)
from preprocess.humanparsing.run_parsing import Parsing
from preprocess.openpose.run_openpose import OpenPose
from src.tryon_pipeline import StableDiffusionXLInpaintPipeline
from src.unet_hacked_tryon import UNet2DConditionModel
from src.unet_hacked_garmnet import UNet2DConditionModel as UNetGarm

MODEL_LOADED = False
model = {}

def download_everything():
    """
    Cache varsa oradan okur (Symlink), yoksa indirir.
    SameFileError hatasını çözer.
    """
    print("⬇️ MODELLER BAĞLANIYOR (SYMLINK)...")
    
    # 1. ANA MODEL (IDM-VTON) -> ckpt klasörüne
    # local_dir_use_symlinks=True yaptık. Kopyalama hatası vermez, direkt bağlar.
    snapshot_download(
        repo_id="yisol/IDM-VTON",
        local_dir="ckpt",
        local_dir_use_symlinks=True  # <--- İŞTE ÇÖZÜM BURASI
    )
    
    # 2. IMAGE ENCODER (LAION)
    snapshot_download(
        repo_id="laion/CLIP-ViT-H-14-laion2B-s32B-b79K",
        local_dir="image_encoder",
        local_dir_use_symlinks=True
    )

    # 3. UNET GARM (SDXL Base)
    # Sadece UNet kısmını alıyoruz
    snapshot_download(
        repo_id="stabilityai/stable-diffusion-xl-base-1.0",
        local_dir="unet_garm",
        allow_patterns=["unet/*"], 
        local_dir_use_symlinks=True
    )
        
    # 4. OPENPOSE YEDEKLEME
    # Burası kopyalama yapmak zorunda çünkü klasör yapısı farklı.
    # Önce hedefi siliyoruz ki "SameFileError" vermesin.
    src_pose = "ckpt/openpose/ckpts/body_pose_model.pth"
    dst_pose = "preprocess/openpose/ckpts/body_pose_model.pth"
    
    # Kaynak gerçek dosya değil de symlink ise onun gerçek yolunu bulalım
    if os.path.islink(src_pose):
        src_pose = os.path.realpath(src_pose)

    if os.path.exists(src_pose):
        # Eğer hedef zaten varsa ve aynısıysa dokunma
        if not os.path.exists(dst_pose):
            print("🔄 OpenPose yedeği alınıyor...")
            os.makedirs(os.path.dirname(dst_pose), exist_ok=True)
            try:
                shutil.copy(src_pose, dst_pose)
            except shutil.SameFileError:
                pass # Zaten aynıysa sorun yok devam et

    print("✅ Dosya bağlantıları tamamlandı.")

def load_model():
    global MODEL_LOADED, model
    if MODEL_LOADED: return model

    # İNDİRME/BAĞLAMA
    download_everything()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"🚀 Modeller YERELDEN (ckpt) Yükleniyor... Device: {device}")

    parsing = Parsing(0)
    openpose = OpenPose(0)

    # HuggingFace kütüphanesi symlinkleri otomatik tanır ve açar.
    
    tokenizer = AutoTokenizer.from_pretrained("ckpt", subfolder="tokenizer")
    tokenizer_2 = AutoTokenizer.from_pretrained("ckpt", subfolder="tokenizer_2")
    
    text_encoder = CLIPTextModel.from_pretrained("ckpt", subfolder="text_encoder", torch_dtype=torch.float16).to(device)
    text_encoder_2 = CLIPTextModelWithProjection.from_pretrained("ckpt", subfolder="text_encoder_2", torch_dtype=torch.float16).to(device)
    
    image_encoder = CLIPVisionModelWithProjection.from_pretrained("image_encoder", torch_dtype=torch.float16).to(device)
    feature_extractor = CLIPImageProcessor.from_pretrained("image_encoder", torch_dtype=torch.float16)
    
    # UNet'ler
    unet = UNet2DConditionModel.from_pretrained("ckpt", subfolder="unet", torch_dtype=torch.float16).to(device)
    
    # Garm Unet path kontrolü
    garm_path = "unet_garm/unet" if os.path.exists("unet_garm/unet") else "unet_garm"
    unet_encoder = UNetGarm.from_pretrained(garm_path, torch_dtype=torch.float16, use_safetensors=True).to(device)

    # PIPELINE
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
    print("✅ Sistem Hazır! (v28 - Symlink Mode)")
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
    print("🚀 HANDLER ÇALIŞIYOR (v28)")
    data = job["input"]
    
    try:
        human = smart_resize(b64_to_img(data["human_image"]), 768, 1024)
        garment = smart_resize(b64_to_img(data["garment_image"]), 768, 1024)
        steps = data.get("steps", 30)
        seed = data.get("seed", 42)
    
        mdl = load_model()
    except Exception as e:
        import traceback
        return {"error": f"Yükleme/Giriş Hatası: {str(e)}", "trace": traceback.format_exc()}

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
        except: pass
            
        if pose is None: pose = Image.new("RGB", human.size, (0,0,0))
        
        device = mdl["device"]
        pose_tensor = transforms.ToTensor()(pose).unsqueeze(0).to(device, torch.float16)
        cloth_tensor = transforms.ToTensor()(garment).unsqueeze(0).to(device, torch.float16)
        
        result = mdl["pipe"](
            prompt="clothes", image=human, mask_image=mask_image, ip_adapter_image=garment, 
            cloth=cloth_tensor, pose_img=pose_tensor, num_inference_steps=steps, guidance_scale=2.0,
            generator=torch.Generator(device).manual_seed(seed), height=1024, width=768
        ).images[0]

    return {"output": img_to_b64(result)}

runpod.serverless.start({"handler": handler})
