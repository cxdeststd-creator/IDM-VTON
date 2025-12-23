import sys
import os
import io
import base64
import numpy as np
import torch
import runpod
from PIL import Image, ImageOps
from torchvision import transforms
from huggingface_hub import snapshot_download

# --- GEREKLİ IMPORTLAR (shinsanghooon imajına göre) ---
try:
    sys.path.append(os.getcwd()) # Ana dizini path'e ekle
    from src.tryon_pipeline import StableDiffusionXLInpaintPipeline as TryonPipeline
    from src.unet_hacked_garmnet import UNet2DConditionModel as UNet2DConditionModel_ref
    from src.unet_hacked_tryon import UNet2DConditionModel
    from preprocess.humanparsing.run_parsing import Parsing
    from preprocess.openpose.run_openpose import OpenPose
    from detectron2.data.detection_utils import convert_PIL_to_numpy, _apply_exif_orientation
    import apply_net
    from transformers import (
        CLIPImageProcessor, CLIPVisionModelWithProjection,
        CLIPTextModel, CLIPTextModelWithProjection, AutoTokenizer
    )
    from diffusers import DDPMScheduler, AutoencoderKL
except ImportError as e:
    print(f"⚠️ İçe aktarma uyarısı (Normal olabilir): {e}")

# --- GLOBAL MODEL ---
MODEL = None

def load_model():
    global MODEL
    if MODEL is not None:
        return MODEL

    print("🚀 Modeller Yükleniyor...")
    base_path = 'ckpt' # İmaj içindeki model yolu

    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # UNet ve Bileşenler
    unet = UNet2DConditionModel.from_pretrained(base_path, subfolder="unet", torch_dtype=torch.float16)
    tokenizer_one = AutoTokenizer.from_pretrained(base_path, subfolder="tokenizer", use_fast=False)
    tokenizer_two = AutoTokenizer.from_pretrained(base_path, subfolder="tokenizer_2", use_fast=False)
    noise_scheduler = DDPMScheduler.from_pretrained(base_path, subfolder="scheduler")
    text_encoder_one = CLIPTextModel.from_pretrained(base_path, subfolder="text_encoder", torch_dtype=torch.float16)
    text_encoder_two = CLIPTextModelWithProjection.from_pretrained(base_path, subfolder="text_encoder_2", torch_dtype=torch.float16)
    image_encoder = CLIPVisionModelWithProjection.from_pretrained(base_path, subfolder="image_encoder", torch_dtype=torch.float16)
    vae = AutoencoderKL.from_pretrained(base_path, subfolder="vae", torch_dtype=torch.float16)
    UNet_Encoder = UNet2DConditionModel_ref.from_pretrained(base_path, subfolder="unet_encoder", torch_dtype=torch.float16)

    # Yardımcı Modeller (Parsing ve Skeleton için)
    parsing_model = Parsing(0)
    # OpenPose yerine DensePose kullanacağız ama nesnesi dursun
    openpose_model = OpenPose(0) 

    # Encoder'ları GPU'ya al (Ana model CPU offload ile yönetilecek)
    UNet_Encoder.to(device)
    image_encoder.to(device)
    
    # Pipeline Oluştur
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
    )
    pipe.unet_encoder = UNet_Encoder

    # --- KRİTİK BELLEK AYARLARI (OOM FİX) ---
    # Bu iki satır olmazsa 24GB kartta PATLAR.
    print("🧠 Bellek optimizasyonu (CPU Offload) devreye alınıyor...")
    pipe.enable_model_cpu_offload()
    pipe.enable_vae_slicing()
    # -----------------------------------------

    MODEL = {
        "pipe": pipe,
        "parsing": parsing_model,
        "device": device
    }
    print("✅ Sistem Hazır!")
    return MODEL

def b64_to_img(b64_str):
    if "," in b64_str: b64_str = b64_str.split(",")[1]
    return Image.open(io.BytesIO(base64.b64decode(b64_str))).convert("RGB")

def img_to_b64(img):
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=95)
    return base64.b64encode(buf.getvalue()).decode("utf-8")

def handler(job):
    print("⚡ İş Başladı")
    job_input = job["input"]
    
    model_data = load_model()
    pipe = model_data["pipe"]
    parsing_model = model_data["parsing"]
    device = model_data["device"]

    # 1. Resimleri Al ve Resize Et (Bellek için şart)
    human_img = b64_to_img(job_input["human_image"]).resize((768, 1024))
    garm_img = b64_to_img(job_input["garment_image"]).resize((768, 1024))
    garment_des = job_input.get("description", "a cloth")
    steps = job_input.get("steps", 30)
    seed = job_input.get("seed", 42)

    # 2. Maske Oluştur (Parsing)
    model_parse, _ = parsing_model(human_img.resize((384, 512)))
    mask, _ = parsing_model.get_mask(human_img, model_parse)
    if mask is None: # Yedek maske yöntemi
         parse_arr = np.array(model_parse)
         mask_arr = (parse_arr == 4) | (parse_arr == 6) | (parse_arr == 7)
         mask = Image.fromarray((mask_arr * 255).astype(np.uint8)).resize((768, 1024))

    # 3. İskelet Oluştur (DensePose - Detectron2)
    # "İskeleti alan" kısım burasıdır.
    human_img_arg = _apply_exif_orientation(human_img.resize((384, 512)))
    human_img_arg = convert_PIL_to_numpy(human_img_arg, format="BGR")
    
    # Detectron2 argümanları
    args = apply_net.create_argument_parser().parse_args((
        'show', './configs/densepose_rcnn_R_50_FPN_s1x.yaml', 
        './ckpt/densepose/model_final_162be9.pkl', 
        'dp_segm', '-v', '--opts', 'MODEL.DEVICE', 'cuda'
    ))
    pose_img = args.func(args, human_img_arg)
    pose_img = pose_img[:,:,::-1] # BGR -> RGB
    pose_img = Image.fromarray(pose_img).resize((768, 1024))

    # 4. Inference (Giydirme)
    with torch.no_grad(), torch.cuda.amp.autocast():
        (prompt_embeds, neg_prompt_embeds, pooled_embeds, neg_pooled_embeds) = pipe.encode_prompt(
            "model is wearing " + garment_des, 
            num_images_per_prompt=1, do_classifier_free_guidance=True, 
            negative_prompt="monochrome, lowres, bad anatomy, worst quality, low quality"
        )
        (prompt_embeds_c, _, _, _) = pipe.encode_prompt(
            "a photo of " + garment_des, 
            num_images_per_prompt=1, do_classifier_free_guidance=False, 
            negative_prompt="monochrome, lowres, bad anatomy, worst quality, low quality"
        )

        tensor_transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize([0.5], [0.5])])

        images = pipe(
            prompt_embeds=prompt_embeds.to(device, torch.float16),
            negative_prompt_embeds=neg_prompt_embeds.to(device, torch.float16),
            pooled_prompt_embeds=pooled_embeds.to(device, torch.float16),
            negative_pooled_prompt_embeds=neg_pooled_embeds.to(device, torch.float16),
            num_inference_steps=steps,
            generator=torch.Generator(device).manual_seed(seed),
            strength=1.0,
            pose_img=tensor_transform(pose_img).unsqueeze(0).to(device, torch.float16),
            text_embeds_cloth=prompt_embeds_c.to(device, torch.float16),
            cloth=tensor_transform(garm_img).unsqueeze(0).to(device, torch.float16),
            mask_image=mask,
            image=human_img,
            height=1024, width=768,
            ip_adapter_image=garm_img.resize((768, 1024)),
            guidance_scale=2.0,
        )[0]

    return {"image": img_to_b64(images[0])}

runpod.serverless.start({"handler": handler})
