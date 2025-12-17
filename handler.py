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

# -------------------------------------------------
# CKPT DOWNLOAD (RUNTIME)
# -------------------------------------------------
def ensure_ckpts():
    # BURASI ÖNEMLİ:
    # Orijinal repoda dosya isimleri hatalı (model_final_... vs).
    # O yüzden isimlerin düzeltilmiş olduğu bu repoyu kullanıyoruz.
    AUX_REPO = "camenduru/IDM-VTON"
    
    ckpt_files = [
        "ckpt/densepose/densepose_model.pkl",
        "ckpt/humanparsing/parsing_atr.onnx",
        "ckpt/openpose/body_pose_model.pth"
    ]

    for file in ckpt_files:
        if not os.path.exists(file):
            print(f"⬇️ Downloading {file} from {AUX_REPO}...")
            try:
                hf_hub_download(
                    repo_id=AUX_REPO,
                    filename=file,
                    local_dir=".",
                    local_dir_use_symlinks=False
                )
            except Exception as e:
                print(f"⚠️ Hata: {file} indirilemedi! Detay: {e}")
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
