import runpod
import torch
# IDM-VTON importlarını buraya yap (Proje yapına göre değişir)
# from inference import tryon_pipeline 

# 1. MODELİ GLOBAL OLARAK YÜKLE (Cold Start)
print("🚀 Model Yükleniyor...")
# pipeline = tryon_pipeline.load() # Burası senin koduna göre değişir
print("✅ Model Hazır!")

def handler(job):
    job_input = job["input"]
    
    # Gelen veriyi al
    insan_img = job_input.get("human_image")
    kiyafet_img = job_input.get("garment_image")
    
    # İşlemi yap
    # sonuc = pipeline(insan_img, kiyafet_img)
    
    # Sonucu dön (Base64 string veya S3 linki)
    return {"output": "buraya_sonuc_resmi_gelecek"}

# RunPod'u dinlemeye başla
runpod.serverless.start({"handler": handler})
