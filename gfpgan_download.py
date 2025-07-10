import os
import requests

def download(url, output_path):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    print(f"📥 Descargando {os.path.basename(output_path)}...")
    r = requests.get(url, stream=True)
    r.raise_for_status()
    with open(output_path, "wb") as f:
        for chunk in r.iter_content(chunk_size=8192):
            f.write(chunk)
    print(f"✅ Guardado en {output_path}")

# GFPGAN v1.4
download(
    "https://huggingface.co/gmk123/GFPGAN/resolve/main/GFPGANv1.4.pth",
    "gfpgan/experiments/pretrained_models/GFPGANv1.4.pth"
)

# RealESRGAN x4plus
download(
    "https://huggingface.co/lllyasviel/Annotators/resolve/main/RealESRGAN_x4plus.pth",
    "gfpgan/experiments/pretrained_models/RealESRGAN_x4plus.pth"
)

# FaceXLib modelos (weights/)
download(
    "https://huggingface.co/camenduru/facexlib/resolve/main/detection_Resnet50_Final.pth",
    "gfpgan/weights/detection_Resnet50_Final.pth"
)

download(
    "https://huggingface.co/camenduru/facexlib/resolve/main/parsing_parsenet.pth",
    "gfpgan/weights/parsing_parsenet.pth"
)

# RealESRGAN_x4plus_anime_6B
download(
    "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.2.4/RealESRGAN_x4plus_anime_6B.pth",
    "gfpgan/experiments/pretrained_models/RealESRGAN_x4plus_anime_6B.pth"
)


