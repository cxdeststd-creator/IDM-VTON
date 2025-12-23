import os
import io
import base64
import numpy as np
import torch
import shutil
import sys
import gc
import subprocess
from PIL import Image, ImageOps
import runpod
from torchvision import transforms 
from huggingface_hub import snapshot_download

# --- 1. ORTAM HAZIRLIĞI (SENİN KODUN İÇİN GEREKLİ) ---
# Senin kodun DensePose kullanıyor, bu yüzden Detectron2 şart.
try:
    import detectron2
except ImportError:
    print("⚙️ Detectron2 (DensePose) kuruluyor... Bu 2-3 dakika sürebilir.")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "git+https://github.com/facebookresearch/detectron2.git"])

print("⬇️ Modeller İndiriliyor...")
snapshot_download(repo_id="yisol/IDM-VTON", local_dir="ckpt", local_dir_use_symlinks=True)

# src klasörünü yerine koyuyoruz
if os.path.exists("ckpt/src") and not os.path.exists("src"):
    shutil.move("ckpt/src", "src")
# Config dosyalarını yerine koyuyoruz (DensePose için şart)
if os.path.exists("ckpt/configs") and not os.path.exists("configs"):
    shutil.move("ckpt/configs", "configs")

sys.path.append(os.getcwd())

# --- SENİN ATTIĞIN IMPORTLAR ---
from preprocess.humanparsing.run_parsing import Parsing
from preprocess.openpose.run_openpose import OpenPose
from src.tryon_pipeline import StableDiffusionXLInpaintPipeline as TryonPipeline
from src.unet_hacked_garmnet import UNet2DConditionModel as UNet2DConditionModel_ref
from src.unet_hacked_tryon import UNet2DConditionModel
from transformers import (
    CLIPImageProcessor,
    CLIPVisionModelWithProjection,
    CLIPTextModel,
    CLIPTextModelWithProjection,
    AutoTokenizer,
)
from diffusers import DDPMScheduler, AutoencoderKL
import apply_net # DensePose Scripti
from detectron2.data.detection_utils import convert_PIL_to_numpy, _apply_exif_orientation
from torchvision.transforms.functional import to_pil_image

MODEL_LOADED = False
model = {}

def load_model():
    global MODEL_LOADED, model
    if MODEL_LOADED: return model

    print("🚀 Modeller Yükleniyor...")
    base_path = 'ckpt'
    
    unet = UNet2DConditionModel.from_pretrained(base_path, subfolder="unet", torch_dtype=torch.float16)
    tokenizer_one = AutoTokenizer.from_pretrained(base_path, subfolder="tokenizer", use_fast=False)
    tokenizer_two = AutoTokenizer.from_pretrained(base_path, subfolder="tokenizer_2", use_fast=False)
    noise_scheduler = DDPMScheduler.from_pretrained(base_path, subfolder="scheduler")
    
    text_encoder_one = CLIPTextModel.from_pretrained(base_path, subfolder="text_encoder", torch_dtype=torch.float16)
    text_encoder_two = CLIPTextModelWithProjection.from_pretrained(base_path, subfolder="text_encoder_2", torch_dtype=torch.float16)
    image_encoder = CLIPVisionModelWithProjection.from_pretrained(base_path, subfolder="image_encoder", torch_dtype=torch.float16)
    vae = AutoencoderKL.from_pretrained(base_path, subfolder="vae", torch_dtype=torch.float16)
    
    # "unet_encoder" kısmını senin kodundaki gibi yüklüyoruz
    UNet_Encoder = UNet2DConditionModel_ref.from_pretrained(base_path, subfolder="unet_encoder", torch_dtype=torch.float16)
    
    parsing_model = Parsing(0)
    openpose_model = OpenPose(0) # Bu sadece el/yüz için, vücut için DensePose kullanacağız
    
    device = "cuda"
    UNet_Encoder.to(device)
    unet.to(device)
    image_encoder.to(device)
    text_encoder_one.to(device)
    text_encoder_two.to(device)
    vae.to(device)

    pipe = TryonPipeline.from_pretrained(
        base_path,
        unet=unet,
        vae=vae,
        feature_extractor=CLIPImageProcessor(),
        text_encoder=text_encoder_one,
        text_encoder_2=text_encoder_two,
        tokenizer=tokenizer_one,
        tokenizer_2=tokenizer_two,
        scheduler=noise_scheduler,
        image_encoder=image_encoder,
        torch_dtype=torch.float16,
    ).to(device)
    
    pipe.unet_encoder = UNet_Encoder
    
    model = {"pipe": pipe, "parsing": parsing_model, "openpose": openpose_model, "device": device}
    MODEL_LOADED = True
    print("✅ Sistem Hazır (Original Logic)!")
    return model

# --- YARDIMCI FONKSİYONLAR ---
def pil_to_binary_mask(pil_image, threshold=0):
    np_image = np.array(pil_image)
    grayscale_image = Image.fromarray(np_image).convert("L")
    binary_mask = np.array(grayscale_image) > threshold
    mask = np.zeros(binary_mask.shape, dtype=np.uint8)
    for i in range(binary_mask.shape[0]):
        for j in range(binary_mask.shape[1]):
            if binary_mask[i,j] == True :
                mask[i,j] = 1
    mask = (mask*255).astype(np.uint8)
    output_mask = Image.fromarray(mask)
    return output_mask

def b64_to_img(b64):
    if "," in b64: b64 = b64.split(",")[1]
    return ImageOps.exif_transpose(Image.open(io.BytesIO(base64.b64decode(b64))).convert("RGB"))

def img_to_b64(img):
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=95)
    return base64.b64encode(buf.getvalue()).decode()

# --- HANDLER (SENİN ATTIĞIN start_tryon FONKSİYONU) ---
def handler(job):
    print("🚀 HANDLER BAŞLADI (v60)")
    data = job["input"]
    mdl = load_model()
    
    # Girdileri hazırla
    garm_img = b64_to_img(data["garment_image"]).convert("RGB").resize((768,1024))
    human_img = b64_to_img(data["human_image"]).convert("RGB").resize((768,1024))
    garment_des = data.get("description", "clothes")
    denoise_steps = data.get("steps", 30)
    seed = data.get("seed", 42)
    
    pipe = mdl["pipe"]
    parsing_model = mdl["parsing"]
    openpose_model = mdl["openpose"]
    device = mdl["device"]
    
    tensor_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5]),
    ])

    with torch.no_grad():
        # 1. Parsing & Mask (Senin kodundaki gibi)
        # keypoints = openpose_model(human_img.resize((384,512))) # Gereksizse silebilirsin ama dursun
        model_parse, _ = parsing_model(human_img.resize((384,512)))
        
        # Basit maske mantığı (Parsing'den elbise bölgelerini al)
        parse_arr = np.array(model_parse)
        # 4: Üst, 6: Elbise, 7: Ceket
        mask_arr = (parse_arr == 4) | (parse_arr == 6) | (parse_arr == 7)
        mask = Image.fromarray((mask_arr * 255).astype(np.uint8)).resize((768, 1024))

        # Mask Gray (Senin kodundaki logic)
        mask_gray = (1-transforms.ToTensor()(mask)) * tensor_transform(human_img)
        mask_gray = to_pil_image((mask_gray+1.0)/2.0)

        # 2. DensePose (Senin kodundaki apply_net kısmı)
        human_img_arg = _apply_exif_orientation(human_img.resize((384,512)))
        human_img_arg = convert_PIL_to_numpy(human_img_arg, format="BGR")
        
        args = apply_net.create_argument_parser().parse_args((
            'show', './configs/densepose_rcnn_R_50_FPN_s1x.yaml', 
            './ckpt/densepose/model_final_162be9.pkl', 
            'dp_segm', '-v', '--opts', 'MODEL.DEVICE', 'cuda'
        ))
        pose_img = args.func(args, human_img_arg)
        pose_img = pose_img[:,:,::-1]
        pose_img = Image.fromarray(pose_img).resize((768,1024))

        # 3. Prompt Encoding (Senin kodundaki logic)
        prompt = "model is wearing " + garment_des
        negative_prompt = "monochrome, lowres, bad anatomy, worst quality, low quality"
        
        (prompt_embeds, negative_prompt_embeds, pooled_prompt_embeds, negative_pooled_prompt_embeds) = pipe.encode_prompt(
            prompt, num_images_per_prompt=1, do_classifier_free_guidance=True, negative_prompt=negative_prompt,
        )
        
        prompt_cloth = "a photo of " + garment_des
        (prompt_embeds_c, _, _, _) = pipe.encode_prompt(
            prompt_cloth, num_images_per_prompt=1, do_classifier_free_guidance=False, negative_prompt=negative_prompt,
        )

        # 4. Pipeline Run
        pose_img_tensor = tensor_transform(pose_img).unsqueeze(0).to(device, torch.float16)
        garm_tensor = tensor_transform(garm_img).unsqueeze(0).to(device, torch.float16)
        
        generator = torch.Generator(device).manual_seed(seed)
        
        images = pipe(
            prompt_embeds=prompt_embeds.to(device, torch.float16),
            negative_prompt_embeds=negative_prompt_embeds.to(device, torch.float16),
            pooled_prompt_embeds=pooled_prompt_embeds.to(device, torch.float16),
            negative_pooled_prompt_embeds=negative_pooled_prompt_embeds.to(device, torch.float16),
            num_inference_steps=denoise_steps,
            generator=generator,
            strength=1.0,
            pose_img=pose_img_tensor,
            text_embeds_cloth=prompt_embeds_c.to(device, torch.float16),
            cloth=garm_tensor,
            mask_image=mask,
            image=human_img,
            height=1024,
            width=768,
            ip_adapter_image=garm_img.resize((768,1024)),
            guidance_scale=2.0,
        )[0]

    return {"output": img_to_b64(images[0])}

runpod.serverless.start({"handler": handler})
