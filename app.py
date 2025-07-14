import os
import cv2
import time
from datetime import datetime
from flask import Flask, render_template, request, redirect, send_from_directory
from PIL import Image
from werkzeug.utils import secure_filename
from basicsr.archs.rrdbnet_arch import RRDBNet
from gfpgan import GFPGANer
from realesrgan import RealESRGANer

app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = "inputs"
app.config["OUTPUT_FOLDER"] = "static/outputs"

os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)
os.makedirs(app.config["OUTPUT_FOLDER"], exist_ok=True)

def cleanup_old_files(folder, max_age=3600):
    now = time.time()
    for f in os.listdir(folder):
        path = os.path.join(folder, f)
        if os.path.isfile(path) and now - os.path.getmtime(path) > max_age:
            os.remove(path)

@app.route("/favicon.ico")
def favicon():
    return send_from_directory(
        os.path.join(app.root_path, "static"),
        "favicon.png",
        mimetype="image/png"
    )

@app.route("/")
def home_redirect():
    lang = request.accept_languages.best_match(["en", "es"])
    return redirect(f"/{lang or 'en'}")

@app.route("/<lang>", methods=["GET", "POST"])
def index(lang):
    if lang not in ["en", "es"]:
        return "Idioma no válido", 404

    # Limpieza de archivos antiguos
    cleanup_old_files(app.config["UPLOAD_FOLDER"])
    cleanup_old_files(app.config["OUTPUT_FOLDER"])

    output_image = None
    original_image = None

    if request.method == "POST":
        file = request.files.get("image")
        usar_fondo = request.form.get("mejorar_fondo") == "1"
        modo_dibujo = request.form.get("modo_dibujo") == "1"
        escala_str = request.form.get("escala", "2")

        try:
            upscale = int(escala_str)
        except ValueError:
            upscale = 2

        if file:
            safe_filename = secure_filename(file.filename)
            base_name = os.path.splitext(safe_filename)[0]
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            enhanced_filename = f"{base_name}_enhance_{timestamp}.png"
            original_filename = f"{base_name}_original_{timestamp}.png"

            input_path = os.path.join(app.config["UPLOAD_FOLDER"], safe_filename)
            output_path = os.path.join(app.config["OUTPUT_FOLDER"], enhanced_filename)
            original_copy_path = os.path.join(app.config["OUTPUT_FOLDER"], original_filename)

            output_image = f"outputs/{enhanced_filename}"
            original_image = f"outputs/{original_filename}"

            img = Image.open(file).convert("RGB")
            img.save(input_path)

            img_np = cv2.imread(input_path)
            if img_np is None:
                return "Error al leer la imagen. ¿Formato inválido?", 400

            max_dim = 500
            h, w = img_np.shape[:2]
            original_size = (w, h)

            if w > max_dim or h > max_dim:
                aspect_ratio = w / h
                if aspect_ratio > 1:
                    new_w = max_dim
                    new_h = int(max_dim / aspect_ratio)
                else:
                    new_h = max_dim
                    new_w = int(max_dim * aspect_ratio)
                img_np = cv2.resize(img_np, (new_w, new_h), interpolation=cv2.INTER_AREA)

            if modo_dibujo:
                model = RRDBNet(
                    num_in_ch=3, num_out_ch=3,
                    num_feat=64, num_block=6,
                    num_grow_ch=32, scale=4
                )
                anime_upsampler = RealESRGANer(
                    scale=upscale,
                    model_path='gfpgan/experiments/pretrained_models/RealESRGAN_x4plus_anime_6B.pth',
                    model=model,
                    tile=0,
                    tile_pad=10,
                    pre_pad=0,
                    half=False,
                    device='cpu'
                )
                output, _ = anime_upsampler.enhance(img_np, outscale=upscale)
                restored_rgb = cv2.cvtColor(output, cv2.COLOR_BGR2RGB)

            else:
                bg_upsampler = None
                if usar_fondo:
                    model = RRDBNet(
                        num_in_ch=3, num_out_ch=3,
                        num_feat=64, num_block=23,
                        num_grow_ch=32, scale=4
                    )
                    bg_upsampler = RealESRGANer(
                        scale=upscale,
                        model_path='gfpgan/experiments/pretrained_models/RealESRGAN_x4plus.pth',
                        model=model,
                        tile=0,
                        tile_pad=10,
                        pre_pad=0,
                        half=False,
                        device='cpu'
                    )

                restorer = GFPGANer(
                    model_path='gfpgan/experiments/pretrained_models/GFPGANv1.4.pth',
                    upscale=upscale,
                    arch='clean',
                    channel_multiplier=2,
                    bg_upsampler=bg_upsampler,
                    device='cpu'
                )
                _, _, restored_img = restorer.enhance(
                    img_np, has_aligned=False, only_center_face=False, paste_back=True
                )
                restored_rgb = cv2.cvtColor(restored_img, cv2.COLOR_BGR2RGB)

            result_pil = Image.fromarray(restored_rgb)

            if upscale == 1:
                final_size = original_size
            else:
                final_size = (original_size[0] * upscale, original_size[1] * upscale)

            result_pil = result_pil.resize(final_size, Image.LANCZOS)
            result_pil.save(output_path)

            resized_original = img.resize(final_size, Image.LANCZOS)
            resized_original.save(original_copy_path)

    return render_template(f"{lang}/index.html",
                           output_image=output_image,
                           original_image=original_image)

@app.route("/<lang>/about")
def acerca(lang):
    if lang not in ["en", "es"]:
        return "Idioma no válido", 404
    return render_template(f"{lang}/about.html")

@app.route("/<lang>/how-to-use")
def como_usar(lang):
    if lang not in ["en", "es"]:
        return "Idioma no válido", 404
    return render_template(f"{lang}/how-to-use.html")

@app.route("/<lang>/privacy")
def privacidad(lang):
    if lang not in ["en", "es"]:
        return "Idioma no válido", 404
    return render_template(f"{lang}/privacy.html")

@app.route("/<lang>/faq")
def faq(lang):
    if lang not in ["en", "es"]:
        return "Idioma no válido", 404
    return render_template(f"{lang}/faq.html")

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    app.run(host="0.0.0.0", port=port)
