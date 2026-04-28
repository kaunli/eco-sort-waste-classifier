from flask import Flask, render_template, request
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.vgg16 import preprocess_input as vgg_preprocess
from tensorflow.keras.applications.resnet50 import preprocess_input as resnet_preprocess
from tensorflow.keras.preprocessing import image
from PIL import Image
import os

# -----------------------
# Setup paths
# -----------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
STATIC_DIR = os.path.join(BASE_DIR, "static")
TEMPLATES_DIR = os.path.join(BASE_DIR, "templates")
os.makedirs(STATIC_DIR, exist_ok=True)
os.makedirs(TEMPLATES_DIR, exist_ok=True)

# -----------------------
# Initialize Flask app
# -----------------------
app = Flask(__name__, template_folder=TEMPLATES_DIR, static_folder=STATIC_DIR)

# -----------------------
# Load models
# -----------------------
vgg_model = load_model(os.path.join(BASE_DIR, "best_vgg16_ecosort.keras"))
resnet_model = load_model(os.path.join(BASE_DIR, "best_resnet50_ecosort.keras"))

# -----------------------
# Define categories
# -----------------------
categories = ['Hazardous', 'Non-Recyclable', 'Organic', 'Recyclable']

category_instructions = {
    'Hazardous': "⚠️ Must be handled through specialized disposal programs.",
    'Non-Recyclable': "🗑️ Dispose in regular trash.",
    'Organic': "🌱 Suitable for composting.",
    'Recyclable': "♻️ Place in the recycling bin."
}

# -----------------------
# Helper functions
# -----------------------
def prepare_image(file, target_size=(224, 224)):
    img = Image.open(file.stream).convert("RGB")
    img = img.resize(target_size)
    img_array = image.img_to_array(img)
    return img_array

def predict_image(model, img_array, model_type):
    x = np.expand_dims(img_array, axis=0)
    if model_type == "vgg":
        x = vgg_preprocess(x)
    else:
        x = resnet_preprocess(x)
    preds = model.predict(x)
    idx = np.argmax(preds)
    conf = float(preds[0][idx])
    return categories[idx], conf

# -----------------------
# Routes
# -----------------------
@app.route("/", methods=["GET", "POST"])
def index():
    result = None
    preview_path = None

    if request.method == "POST":
        if "file" not in request.files:
            return render_template("index.html", error="No file uploaded.")

        file = request.files["file"]
        if file.filename == "":
            return render_template("index.html", error="File not selected.")

        # Save uploaded image
        preview_path = os.path.join("static", "uploaded_image.jpg")
        file.save(os.path.join(BASE_DIR, preview_path))

        # Prepare image
        img_array = prepare_image(file)

        # Predict with both models
        vgg_pred, vgg_conf = predict_image(vgg_model, img_array, "vgg")
        res_pred, res_conf = predict_image(resnet_model, img_array, "resnet")

        # Final decision
        if vgg_pred == res_pred:
            final_pred = vgg_pred
            final_conf = (vgg_conf + res_conf) / 2
        else:
            if vgg_conf >= res_conf:
                final_pred = vgg_pred
                final_conf = vgg_conf
            else:
                final_pred = res_pred
                final_conf = res_conf

        result = {
            "vgg_pred": vgg_pred,
            "vgg_conf": round(vgg_conf, 3),
            "res_pred": res_pred,
            "res_conf": round(res_conf, 3),
            "final_pred": final_pred,
            "final_conf": round(final_conf, 3),
            "instruction": category_instructions[final_pred]
        }

    return render_template("index.html", result=result, image_path=preview_path)


if __name__ == "__main__":
    app.run(debug=True, host="127.0.0.1", port=5001)
