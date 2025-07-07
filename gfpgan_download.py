import os
import requests

# Crear carpetas necesarias
os.makedirs("gfpgan/experiments/pretrained_models", exist_ok=True)
os.makedirs("gfpgan/weights", exist_ok=True)

def download(url, dest):
    if not os.path.exists(dest):
        print(f"📥 Descargando {os.path.basename(dest)}...")
        r = requests.get(url, stream=True)
        r.raise_for_status()
        with open(dest, 'wb') as f:
            for chunk in r.iter_content(chunk_size=8192):
                f.write(chunk)
        print(f"✅ Guardado en {dest}")
    else:
        print(f"✔️ Ya existe: {dest}")

# GFPGAN modelo principal
download(
    "https://github.com/TencentARC/GFPGAN/releases/download/v1.3.4/GFPGANv1.4.pth",
    "gfpgan/experiments/pretrained_models/GFPGANv1.4.pth"
)

# RealESRGAN (upscaler)
download(
    "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5/RealESRGAN_x4plus.pth",
    "gfpgan/experiments/pretrained_models/RealESRGAN_x4plus.pth"
)

# FaceXLib - modelo de detección de rostro
download(
    "https://github.com/TencentARC/GFPGAN/releases/download/v1.3.4/detection_Resnet50_Final.pth",
    "gfpgan/weights/detection_Resnet50_Final.pth"
)

# FaceXLib - modelo de segmentación facial
download(
    "https://github.com/TencentARC/GFPGAN/releases/download/v1.3.4/parsing_parsenet.pth",
    "gfpgan/weights/parsing_parsenet.pth"
)
