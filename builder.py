import os
import requests

def download_file(url, path):
    if os.path.exists(path):
        print(f"✅ Dosya zaten var: {path}")
        return
    
    os.makedirs(os.path.dirname(path), exist_ok=True)
    print(f"⬇️ İndiriliyor: {path}...")
    
    try:
        response = requests.get(url, stream=True)
        if response.status_code == 200:
            with open(path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=1024*1024):
                    if chunk: f.write(chunk)
            print(f"🎉 İndirme tamamlandı: {path}")
        else:
            print(f"❌ HATA: Link bozuk ({response.status_code}): {url}")
    except Exception as e:
        print(f"❌ HATA: {e}")

# 1. HumanParsing Modelleri
download_file("https://huggingface.co/spaces/yisol/IDM-VTON/resolve/main/ckpt/humanparsing/parsing_atr.onnx", "ckpt/humanparsing/parsing_atr.onnx")
download_file("https://huggingface.co/spaces/yisol/IDM-VTON/resolve/main/ckpt/humanparsing/parsing_lip.onnx", "ckpt/humanparsing/parsing_lip.onnx")

# 2. OpenPose Modelleri (ControlNet Orijinal)
download_file("https://huggingface.co/lllyasviel/ControlNet/resolve/main/annotator/ckpts/body_pose_model.pth", "ckpt/openpose/body_pose_model.pth")
download_file("https://huggingface.co/lllyasviel/ControlNet/resolve/main/annotator/ckpts/hand_pose_model.pth", "ckpt/openpose/hand_pose_model.pth")

# 3. DensePose Modeli (Facebook Orijinal Sunucusu - ASLA BOZULMAZ)
download_file("https://dl.fbaipublicfiles.com/densepose/densepose_rcnn_R_50_FPN_s1x/165712039/model_final_162be9.pkl", "ckpt/densepose/model_final_162be9.pkl")
