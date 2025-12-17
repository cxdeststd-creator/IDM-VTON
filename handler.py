import os
import runpod
import torch
import base64
import io
import numpy as np
from PIL import Image
from huggingface_hub import hf_hub_download

MODEL_LOADED = False
model = {}

BASE_REPO = "yisol/IDM-VTON"

import os
import shutil
from huggingface_hub import hf_hub_download

import os
import shutil
from huggingface_hub import hf_hub_download

# -------------------------------------------------
# CKPT DOWNLOAD (RUNTIME) - FİNAL HARİTA
# -------------------------------------------------
def ensure_ckpts():
    print("⬇️ Model dosyaları Yisol reposundan çekiliyor...")
    
    REPO_ID = "yisol/IDM-VTON"
    
    # [HuggingFace Yolu]  ->  [Senin Kodunun İstediği Yol]
    download_map = {
        # Densepose (İsim değişikliği yapılıyor)
        "densepose/model_final_162be9.pkl": "ckpt/densepose/densepose_model.pkl",
        
        # Humanparsing (İki dosyayı da alalım garanti olsun)
        "humanparsing/parsing_atr.onnx": "ckpt/humanparsing/parsing_atr.onnx",
        "humanparsing/parsing_lip.onnx": "ckpt/humanparsing/parsing_lip.onnx",
        
        # Openpose (Senin dediğin klasör yolundan çekiyoruz)
        "openpose/ckpts/body_pose_model.pth": "ckpt/openpose/body_pose_model.pth"
    }

    for remote_path, local_path in download_map.items():
        # Dosya zaten var mı?
        if os.path.exists(local_path):
            continue

        print(f"⏳ İndiriliyor: {remote_path} -> {local_path}")
        
        # 1. Hedef klasörü oluştur
        os.makedirs(os.path.dirname(local_path), exist_ok=True)
        
        try:
            # 2. Dosyayı Hugging Face'ten indir
            downloaded_file_path = hf_hub_download(
                repo_id=REPO_ID,
                filename=remote_path
            )
            
            # 3. İndirilen dosyayı doğru isme ve yere kopyala
            shutil.copy(downloaded_file_path, local_path)
            print(f"✅ Tamamlandı: {local_path}")
            
        except Exception as e:
            print(f"❌ HATA: {remote_path} indirilemedi! Detay: {e}")
            # Bu dosya hayati, indiremezse devam etme
            raise e

# -------------------------------------------------
# MODEL LOAD
# -------------------------------------------------
def load_model():
    global MODEL_LOADED, model

    if MODEL_LOADED:
        return model

    ensure_ckpts()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    from preprocess.humanparsing.run_parsing import Parsing
    from preprocess.openpose.run_openpose import OpenPose
    from src.tryon_pipeline import StableDiffusionXLInpaintPipeline
    from src.unet_hacked_tryon import UNet2DConditionModel
    from src.unet_hacked_garmnet import UNet2DConditionModel as UNetGarm

    parsing = Parsing(0)
    openpose = OpenPose(0)

    unet = UNet2DConditionModel.from_pretrained(
        BASE_REPO,
        subfolder="unet",
        torch_dtype=torch.float16
    )

    unet_garm = UNetGarm.from_pretrained(
        BASE_REPO,
        subfolder="unet_garm",
        torch_dtype=torch.float16
    )

    pipe = StableDiffusionXLInpaintPipeline.from_pretrained(
        BASE_REPO,
        unet=unet,
        torch_dtype=torch.float16,
        use_safetensors=True
    ).to(device)

    pipe.unet_garm = unet_garm

    model = {
        "pipe": pipe,
        "parsing": parsing,
        "openpose": openpose,
        "device": device
    }

    MODEL_LOADED = True
    return model

# -------------------------------------------------
# HELPERS
# -------------------------------------------------
def b64_to_img(b64):
    if "," in b64:
        b64 = b64.split(",")[1]
    return Image.open(io.BytesIO(base64.b64decode(b64))).convert("RGB")

def img_to_b64(img):
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=95)
    return base64.b64encode(buf.getvalue()).decode()

# -------------------------------------------------
# HANDLER
# -------------------------------------------------
def handler(job):
    data = job["input"]

    human = b64_to_img(data["human_image"]).resize((768, 1024))
    garment = b64_to_img(data["garment_image"]).resize((768, 1024))

    steps = data.get("steps", 30)
    seed = data.get("seed", 42)

    mdl = load_model()

    with torch.no_grad():
        human_np = np.array(human)
        parsing = mdl["parsing"](human_np)
        pose = mdl["openpose"](human_np)

        result = mdl["pipe"](
            prompt="clothes",
            image=human,
            garment=garment,
            pose=pose,
            num_inference_steps=steps,
            guidance_scale=2.0,
            generator=torch.Generator(mdl["device"]).manual_seed(seed),
            height=1024,
            width=768
        ).images[0]

    return {"output": img_to_b64(result)}

runpod.serverless.start({"handler": handler})
