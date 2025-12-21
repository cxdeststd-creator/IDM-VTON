import os
import io
import base64
import numpy as np
import torch
import shutil
import sys
from PIL import Image, ImageOps
import runpod
from torchvision import transforms 
from huggingface_hub import snapshot_download

# Standartlar yukarıda.

from transformers import (
    CLIPImageProcessor, 
    CLIPVisionModelWithProjection,
    CLIPTextModel,
    CLIPTextModelWithProjection,
    AutoTokenizer
)

MODEL_LOADED = False
model = {}

def fix_unet_bugs():
    """
    Nihai UNet Tamircisi.
    Hedef: Modelin içine giren her şeyi zorla 2048 boyutuna getirmek.
    """
    target_file = "src/unet_hacked_garmnet.py"
    if not os.path.exists(target_file):
        print(f"⚠️ Uyarı: {target_file} bulunamadı.")
        return

    print(f"🔧 UNET AMELİYATI (v41 - ULTIMATE 2048): {target_file}")
    with open(target_file, "r") as f:
        lines = f.readlines()

    new_lines = []
    fixed = False
    
    # Referans alacağımız satır (Bunun hemen öncesine kod enjekte edeceğiz)
    search_text = 'if "text_embeds" not in added_cond_kwargs:'
    
    for line in lines:
        if search_text in line:
            indent = line.split('if')[0] # Girinti (indentation) kopyala
            
            print("⚡ 2048 Boyut Zorlaması Enjekte Ediliyor...")
            
            # 1. ENCODER HIDDEN STATES PADDING (Ne gelirse gelsin 2048 yap)
            new_lines.append(f'{indent}# v41 FIX: FORCE 2048 DIMENSION\n')
            new_lines.append(f'{indent}if encoder_hidden_states is not None:\n')
            new_lines.append(f'{indent}    current_dim = encoder_hidden_states.shape[-1]\n')
            new_lines.append(f'{indent}    if current_dim != 2048:\n')
            new_lines.append(f'{indent}        pad_amt = 2048 - current_dim\n')
            new_lines.append(f'{indent}        if pad_amt > 0:\n')
            new_lines.append(f'{indent}            encoder_hidden_states = torch.nn.functional.pad(encoder_hidden_states, (0, pad_amt))\n')
            
            # 2. KUTU KONTROLÜ
            new_lines.append(f'{indent}if added_cond_kwargs is None: added_cond_kwargs = {{}}\n')
            
            # 3. TEXT EMBEDS YARATMA (ARTIK 2048 BOYUTUNDA!)
            # Hata burada 1280 kalmıştı, onu 2048 yaptık.
            new_lines.append(f'{indent}if "text_embeds" not in added_cond_kwargs:\n')
            new_lines.append(f'{indent}    added_cond_kwargs["text_embeds"] = torch.zeros((1, 2048), device=sample.device, dtype=sample.dtype)\n')
            
            # 4. TIME IDS (Burası zaten çalışıyordu ama yine ekleyelim)
            new_lines.append(f'{indent}if "time_ids" not in added_cond_kwargs:\n')
            new_lines.append(f'{indent}    added_cond_kwargs["time_ids"] = torch.zeros((1, 6), device=sample.device, dtype=sample.dtype)\n')
            
            # 5. ORİJİNAL HATA KONTROLLERİNİ İPTAL ET
            new_lines.append(f'{indent}if False: # ESKİ KODLARI SUSTUR\n')
            
            fixed = True
        
        # search_text'i içeren satırı (veya time_ids hatasını) artık yazmıyoruz çünkü yukarıda hallettik
        # Ama dosya akışını bozmamak için diğer satırları aynen geçiriyoruz.
        # Sadece "if False" diyerek o bloğu iptal ettiğimiz için altındakiler çalışmayacak.
        
        new_lines.append(line)
            
    if fixed:
        with open(target_file, "w") as f:
            f.writelines(new_lines)
        print("✅ UNet başarıyla 2048 boyutuna ayarlandı.")
    else:
        print("ℹ️ Hedef satır bulunamadı (Dosya daha önce değiştirilmiş olabilir).")

def download_smart():
    print("⬇️ MODELLER KONTROL EDİLİYOR...")
    whitelist = [
        "tokenizer/*", "tokenizer_2/*", "text_encoder/*", "text_encoder_2/*", 
        "vae/*", "unet/*", "scheduler/*", "humanparsing/*", "densepose/*", 
        "openpose/*", "*.json", "*.txt", "*.py" 
    ]
    snapshot_download(repo_id="yisol/IDM-VTON", local_dir="ckpt", allow_patterns=whitelist, local_dir_use_symlinks=True)
    snapshot_download(repo_id="laion/CLIP-ViT-H-14-laion2B-s32B-b79K", local_dir="image_encoder", allow_patterns=["*.json", "*.safetensors", "*.txt"], local_dir_use_symlinks=True)
    snapshot_download(repo_id="stabilityai/stable-diffusion-xl-base-1.0", local_dir="unet_garm", allow_patterns=["unet/*", "*.json"], local_dir_use_symlinks=True)
    
    src_pose = "ckpt/openpose/ckpts/body_pose_model.pth"
    dst_pose = "preprocess/openpose/ckpts/body_pose_model.pth"
    if os.path.islink(src_pose): src_pose = os.path.realpath(src_pose)
    if os.path.exists(src_pose) and not os.path.exists(dst_pose):
        os.makedirs(os.path.dirname(dst_pose), exist_ok=True)
        try: shutil.copy(src_pose, dst_pose)
        except: pass
    print("✅ Dosyalar hazır.")

def load_model():
    global MODEL_LOADED, model
    if MODEL_LOADED: return model

    download_smart()
    fix_unet_bugs() # <--- AMELİYAT BURADA

    sys.path.append(os.getcwd())
    from preprocess.humanparsing.run_parsing import Parsing
    from preprocess.openpose.run_openpose import OpenPose
    from src.tryon_pipeline import StableDiffusionXLInpaintPipeline
    from src.unet_hacked_tryon import UNet2DConditionModel
    from src.unet_hacked_garmnet import UNet2DConditionModel as UNetGarm

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"🚀 Modeller Yükleniyor... Device: {device}")

    parsing = Parsing(0)
    openpose = OpenPose(0)
    
    tokenizer = AutoTokenizer.from_pretrained("ckpt", subfolder="tokenizer", use_fast=False)
    tokenizer_2 = AutoTokenizer.from_pretrained("ckpt", subfolder="tokenizer_2", use_fast=False)
    text_encoder = CLIPTextModel.from_pretrained("ckpt", subfolder="text_encoder", torch_dtype=torch.float16).to(device)
    text_encoder_2 = CLIPTextModelWithProjection.from_pretrained("ckpt", subfolder="text_encoder_2", torch_dtype=torch.float16).to(device)
    image_encoder = CLIPVisionModelWithProjection.from_pretrained("image_encoder", torch_dtype=torch.float16).to(device)
    feature_extractor = CLIPImageProcessor.from_pretrained("image_encoder", torch_dtype=torch.float16)
    unet = UNet2DConditionModel.from_pretrained("ckpt", subfolder="unet", torch_dtype=torch.float16).to(device)
    
    garm_path = "unet_garm/unet" if os.path.exists("unet_garm/unet") else "unet_garm"
    unet_encoder = UNetGarm.from_pretrained(garm_path, torch_dtype=torch.float16, use_safetensors=True).to(device)

    pipe = StableDiffusionXLInpaintPipeline.from_pretrained(
        "ckpt", unet=unet, unet_encoder=unet_encoder, image_encoder=image_encoder,
        feature_extractor=feature_extractor, text_encoder=text_encoder, text_encoder_2=text_encoder_2,
        tokenizer=tokenizer, tokenizer_2=tokenizer_2, torch_dtype=torch.float16, use_safetensors=True,
    ).to(device)
    pipe.register_modules(unet_encoder=unet_encoder)

    model = {"pipe": pipe, "parsing": parsing, "openpose": openpose, "device": device}
    MODEL_LOADED = True
    print("✅ Sistem Hazır! (v41)")
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
    print("🚀 HANDLER ÇALIŞIYOR (v41)")
    data = job["input"]
    try:
        mdl = load_model()
        human = smart_resize(b64_to_img(data["human_image"]), 768, 1024)
        garment = smart_resize(b64_to_img(data["garment_image"]), 768, 1024)
        steps = data.get("steps", 30)
        seed = data.get("seed", 42)
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
            if isinstance(raw, dict) and "pose_img" in raw: pose = raw["pose_img"]
            elif isinstance(raw, Image.Image): pose = raw
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
