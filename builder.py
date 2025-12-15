# builder.py
import torch
from diffusers import StableDiffusionXLInpaintPipeline
from diffusers import UNet2DConditionModel

def download_model():
    # Model ID
    model_id = "yisol/IDM-VTON"
    
    print("⏳ Model indiriliyor, bu işlem build sırasında 1 kere yapılır...")
    
    # UNet'leri indir
    UNet2DConditionModel.from_pretrained(
        model_id,
        subfolder="unet",
        torch_dtype=torch.float16
    )
    
    UNet2DConditionModel.from_pretrained(
        model_id,
        subfolder="unet_garm",
        torch_dtype=torch.float16
    )
    
    # Pipeline'ın geri kalanını indir (VAE, Scheduler, vb.)
    # Not: TryonPipeline class'ı özel olduğu için standart SDXL indiriyoruz
    # Bu sayede cache dolar.
    print("⏳ VAE ve Encoderlar indiriliyor...")
    StableDiffusionXLInpaintPipeline.from_pretrained(
        model_id,
        torch_dtype=torch.float16
    )
    
    print("✅ Tüm modeller indirildi ve Cache'lendi!")

if __name__ == "__main__":
    download_model()
