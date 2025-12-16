import runpod
import torch
import base64
import io
import numpy as np
from PIL import Image

MODEL = None

def load_model():
    global MODEL
    if MODEL is not None:
        return MODEL

    from preprocess.humanparsing.run_parsing import Parsing
    from preprocess.openpose.run_openpose import OpenPose
    from src.tryon_pipeline import StableDiffusionXLInpaintPipeline
    from src.unet_hacked_tryon import UNet2DConditionModel
    from src.unet_hacked_garmnet import UNet2DConditionModel as UNetGarm

    device = "cuda"

    parsing = Parsing(0)
    openpose = OpenPose(0)

    unet = UNet2DConditionModel.from_pretrained(
        "yisol/IDM-VTON", subfolder="unet", torch_dtype=torch.float16
    )
    unet_garm = UNetGarm.from_pretrained(
        "yisol/IDM-VTON", subfolder="unet_garm", torch_dtype=torch.float16
    )

    pipe = StableDiffusionXLInpaintPipeline.from_pretrained(
        "yisol/IDM-VTON",
        unet=unet,
        torch_dtype=torch.float16,
        variant="fp16",
        use_safetensors=True
    ).to(device)

    pipe.unet_garm = unet_garm

    MODEL = (pipe, parsing, openpose)
    return MODEL


def handler(event):
    pipe, parsing, openpose = load_model()

    human = Image.open(io.BytesIO(base64.b64decode(event["input"]["human"]))).convert("RGB")
    garment = Image.open(io.BytesIO(base64.b64decode(event["input"]["garment"]))).convert("RGB")

    human = human.resize((768, 1024))
    garment = garment.resize((768, 1024))

    pose = openpose(np.array(human))

    result = pipe(
        prompt=event["input"].get("prompt", "clothes"),
        image=human,
        garment=garment,
        pose=pose,
        num_inference_steps=30,
        guidance_scale=2.0,
        height=1024,
        width=768
    ).images[0]

    buf = io.BytesIO()
    result.save(buf, format="JPEG")
    return {"output": base64.b64encode(buf.getvalue()).decode()}


runpod.serverless.start({"handler": handler})
