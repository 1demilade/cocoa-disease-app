from flask import Flask, request, jsonify
from flask_cors import CORS
import tensorflow as tf
import numpy as np
from PIL import Image
import os
import io
import requests

app = Flask(__name__)
CORS(app)

# ================================
# Download model from Google Drive if not present
# ================================
MODEL_PATH = "model/DenseNet121_Cocoa_diagnosis.keras"
GDRIVE_FILE_ID = "1UtlCqjhyIP5kFP-6SZxvBIAicQfk_51X"
GDRIVE_URL = f"https://drive.google.com/uc?export=download&id={GDRIVE_FILE_ID}"

def download_model():
    if not os.path.exists("model"):
        os.makedirs("model")
    if not os.path.exists(MODEL_PATH):
        print("📥 Downloading model from Google Drive...")
        response = requests.get(GDRIVE_URL, stream=True)
        with open(MODEL_PATH, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
        print("✅ Model downloaded.")

download_model()

# Load model
model = tf.keras.models.load_model("model/DenseNet121_Cocoa_diagnosis.keras")

# Define class names and image size
class_names = ['anthracnose', 'cssvd', 'healthy']
img_height, img_width = 224, 224

@app.route("/predict", methods=["POST"])
def predict():
    if "image" not in request.files:
        return jsonify({"error": "No image uploaded"}), 400

    image_file = request.files["image"]

    try:
        image = Image.open(image_file.stream).convert("RGB")
        image = image.resize((img_width, img_height))
        image_array = np.array(image) / 255.0
        image_array = np.expand_dims(image_array, axis=0)

        predictions = model.predict(image_array)[0]
        top_class_index = np.argmax(predictions)
        top_class_name = class_names[top_class_index]
        confidence = float(predictions[top_class_index]) * 100

        # Sample recommendation
        RECOMMENDATIONS = {
            "healthy": (
                "✅ Your cocoa plant appears healthy.\n\n"
                " Keep monitoring regularly and maintain good practices:\n"
                "- Prune diseased or overcrowded branches\n"
                "- Apply compost or fertilizer periodically\n"
                "- Weed around the base\n"
                "- Monitor for pests like mealybugs or black pod.\n\n"
                "📖 Learn more:<br>"
                '<a href="https://www.worldcocoafoundation.org/blog/best-practices-for-cocoa-farmers/" target="_blank">World Cocoa Foundation: Best Practices</a>'
            ),
            "cssvd": (
                "⚠️ Your cocoa plant may have Cocoa Swollen Shoot Virus Disease (CSSVD).\n\n"
                " Immediate Actions:\n"
                "- Isolate the affected tree to stop the spread\n"
                "- Contact your local agricultural extension officer or cocoa desk officer\n"
                "- Remove and destroy infected trees carefully under supervision\n"
                "- Control ants and mealybugs (they spread the virus)\n"
                "- Do not reuse tools without sanitizing.\n\n"
                " Spreads through mealybugs and grafting from infected trees.\n\n"
                "📖 Learn more:<br>"
                '<a href="https://www.plantwise.org/blog/agroforestry-mitigating-cocoa-swollen-shoot-virus-disease-in-ghana/" target="_blank">Plantwise: Agroforestry Mitigation</a><br>'
                '<a href="https://en.wikipedia.org/wiki/Cacao_swollen_shoot_virus" target="_blank">Wikipedia: Cacao Swollen Shoot Virus Overview</a>'
            ),
            "anthracnose": (
                "🚨 Your cocoa plant may have Anthracnose, a fungal disease caused by *Colletotrichum* species.\n\n"
                " What to do:\n"
                "- Remove and burn infected pods and leaves\n"
                "- Improve airflow by pruning densely packed trees\n"
                "- Apply copper-based fungicides if available\n"
                "- Disinfect tools after each use\n"
                "- Avoid overhead watering if possible.\n\n"
                " Fungi thrive in humid, wet conditions — improve drainage.\n\n"
                "📖 Learn more:<br>"
                '<a href="https://www.icco.org/about-cocoa/pests-diseases/" target="_blank">ICCO: Cocoa Pests & Diseases (includes Anthracnose)</a><br>'
                '<a href="https://www.cabi.org/cpc/datasheet/18236" target="_blank">CABI Datasheet: Cocoa Anthracnose</a>'
            )
        }

        return jsonify({
            "predicted_class": top_class_name.upper(),
            "confidence": f"{confidence:.2f}%",
            "recommendation": RECOMMENDATIONS[top_class_name]
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True)
