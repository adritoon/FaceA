import os
import cv2
import json
from datetime import datetime
from flask import Flask, render_template, request, redirect, send_from_directory, jsonify, session
from PIL import Image
from basicsr.archs.rrdbnet_arch import RRDBNet
from gfpgan import GFPGANer
from realesrgan import RealESRGANer
from google.cloud import tasks_v2, storage
from werkzeug.utils import secure_filename

app = Flask(__name__)
app.secret_key = os.environ.get("SECRET_KEY", "supersecret")
app.config["UPLOAD_FOLDER"] = "inputs"
app.config["OUTPUT_FOLDER"] = "static"

os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)
os.makedirs(app.config["OUTPUT_FOLDER"], exist_ok=True)

GCP_PROJECT_ID = os.environ.get("GCP_PROJECT_ID")
TASK_HANDLER_URL = os.environ.get("TASK_HANDLER_URL")
QUEUE_ID = "image-processing"
LOCATION = "us-central1"
BUCKET_NAME = os.environ.get("GCS_BUCKET")

@app.route("/favicon.ico")
def favicon():
    return send_from_directory(
        os.path.join(app.root_path, "static"),
        "favicon.png",
        mimetype="image/png"
    )

@app.route("/")
def home_redirect():
    return redirect("/en")

@app.route("/<lang>", methods=["GET", "POST"])
def index(lang):
    if lang not in ["en", "es"]:
        return "Idioma no válido", 404

    output_image = None
    original_image = None
    is_processing = False

    if request.method == "POST":
        file = request.files.get("image")
        usar_fondo = request.form.get("mejorar_fondo") == "1"
        modo_dibujo = request.form.get("modo_dibujo") == "1"
        escala = int(request.form.get("escala", "2"))

        if file:
            base_name = os.path.splitext(secure_filename(file.filename))[0]
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            unique_id = f"{base_name}_{timestamp}"
            session["last_id"] = unique_id

            input_path = os.path.join(app.config["UPLOAD_FOLDER"], f"{unique_id}.png")
            file.save(input_path)

            # Subir archivo a Cloud Storage
            storage_client = storage.Client()
            bucket = storage_client.bucket(BUCKET_NAME)
            blob = bucket.blob(f"inputs/{unique_id}.png")
            blob.upload_from_filename(input_path)

            payload = {
                "gcs_input_path": f"inputs/{unique_id}.png",
                "use_background": usar_fondo,
                "anime_mode": modo_dibujo,
                "upscale": escala,
                "lang": lang,
                "output_name": unique_id
            }

            client = tasks_v2.CloudTasksClient()
            parent = client.queue_path(GCP_PROJECT_ID, LOCATION, QUEUE_ID)

            task = {
                "http_request": {
                    "http_method": tasks_v2.HttpMethod.POST,
                    "url": TASK_HANDLER_URL,
                    "headers": {"Content-Type": "application/json"},
                    "body": json.dumps(payload).encode()
                }
            }

            client.create_task(request={"parent": parent, "task": task})
            is_processing = True
            return render_template(f"{lang}/index.html", is_processing=True, output_image=None, original_image=None)

    # Verificación si ya fue procesada
    if "last_id" in session:
        unique_id = session["last_id"]
        enhanced_filename = f"{unique_id}_enhance.png"
        original_filename = f"{unique_id}_original.png"
        storage_client = storage.Client()
        bucket = storage_client.bucket(BUCKET_NAME)
        enhanced_blob = bucket.blob(f"outputs/{enhanced_filename}")
        original_blob = bucket.blob(f"outputs/{original_filename}")

        if enhanced_blob.exists() and original_blob.exists():
            output_image = enhanced_blob.public_url
            original_image = original_blob.public_url
            is_processing = False

    return render_template(f"{lang}/index.html", is_processing=is_processing, output_image=output_image, original_image=original_image)

@app.route("/task-handler", methods=["POST"])
def process_task():
    data = request.get_json()

    gcs_input_path = data["gcs_input_path"]
    usar_fondo = data["use_background"]
    modo_dibujo = data["anime_mode"]
    upscale = int(data["upscale"])
    output_name = data["output_name"]

    storage_client = storage.Client()
    bucket = storage_client.bucket(BUCKET_NAME)

    input_blob = bucket.blob(gcs_input_path)
    local_input = f"inputs/{output_name}.png"
    input_blob.download_to_filename(local_input)

    enhanced_filename = f"{output_name}_enhance.png"
    original_filename = f"{output_name}_original.png"
    output_path = f"static/{enhanced_filename}"
    original_copy_path = f"static/{original_filename}"

    img = Image.open(local_input).convert("RGB")
    img_np = cv2.imread(local_input)
    h, w = img_np.shape[:2]
    original_size = (w, h)

    max_dim = 500
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
        model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=6, num_grow_ch=32, scale=4)
        anime_upsampler = RealESRGANer(
            scale=upscale,
            model_path='gfpgan/experiments/pretrained_models/RealESRGAN_x4plus_anime_6B.pth',
            model=model,
            tile=0, tile_pad=10, pre_pad=0, half=False, device='cpu'
        )
        output, _ = anime_upsampler.enhance(img_np, outscale=upscale)
        restored_rgb = cv2.cvtColor(output, cv2.COLOR_BGR2RGB)
    else:
        bg_upsampler = None
        if usar_fondo:
            model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=4)
            bg_upsampler = RealESRGANer(
                scale=upscale,
                model_path='gfpgan/experiments/pretrained_models/RealESRGAN_x4plus.pth',
                model=model,
                tile=0, tile_pad=10, pre_pad=0, half=False, device='cpu'
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

    bucket.blob(f"outputs/{enhanced_filename}").upload_from_filename(output_path)
    bucket.blob(f"outputs/{original_filename}").upload_from_filename(original_copy_path)

    return jsonify({"status": "ok"}), 200

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
