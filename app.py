import os
import cv2
from datetime import datetime
from flask import Flask, render_template, request, redirect
from PIL import Image
from basicsr.archs.rrdbnet_arch import RRDBNet
from gfpgan import GFPGANer
from realesrgan import RealESRGANer

app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = "inputs"
app.config["OUTPUT_FOLDER"] = "static"

os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)
os.makedirs(app.config["OUTPUT_FOLDER"], exist_ok=True)

@app.route("/")
def home_redirect():
    return redirect("/en")

@app.route("/<lang>", methods=["GET", "POST"])
def index(lang):
    output_image = None
    original_image = None

    if request.method == "POST":
        file = request.files["image"]
        usar_fondo = request.form.get("mejorar_fondo") == "1"
        escala_str = request.form.get("escala", "2")

        try:
            upscale = int(escala_str)
        except ValueError:
            upscale = 2

        if file:
            # Nombre único
            base_name = os.path.splitext(file.filename)[0]
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            enhanced_filename = f"{base_name}_enhance_{timestamp}.png"
            original_filename = f"{base_name}_original_{timestamp}.png"

            input_path = os.path.join(app.config["UPLOAD_FOLDER"], file.filename)
            output_path = os.path.join(app.config["OUTPUT_FOLDER"], enhanced_filename)
            original_copy_path = os.path.join(app.config["OUTPUT_FOLDER"], original_filename)

            output_image = f"static/{enhanced_filename}"
            original_image = f"static/{original_filename}"

            # Guardar original
            img = Image.open(file).convert("RGB")
            img.save(input_path)

            img_np = cv2.imread(input_path)

            # Upsampler (fondo)
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

            # Restaurador principal
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

            # Guardar resultado
            restored_rgb = cv2.cvtColor(restored_img, cv2.COLOR_BGR2RGB)
            result_pil = Image.fromarray(restored_rgb)
            result_pil.save(output_path)

            # Redimensionar la original al mismo tamaño
            resized_original = img.resize(result_pil.size)
            resized_original.save(original_copy_path)

    return render_template(f"{lang}/index.html", output_image=output_image, original_image=original_image)

@app.route("/<lang>/acerca")
def acerca(lang):
    return render_template(f"{lang}/acerca.html")

@app.route("/<lang>/como-usar")
def como_usar(lang):
    return render_template(f"{lang}/como_usar.html")

@app.route("/<lang>/privacidad")
def privacidad(lang):
    return render_template(f"{lang}/privacidad.html")

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    app.run(host="0.0.0.0", port=port)
