from pathlib import Path
import io
from flask import Flask, render_template, request, jsonify

app = Flask(__name__)
MODEL_FILES = [Path("model/cloud_classifier_model.keras"), Path("model/cloud_classifier_model.h5")]
CLASS_NAMES = ["Ac", "As", "Cc", "Cs", "Ci", "Cb", "Cu", "Ns", "Sc", "St", "Ct"]
CLASS_LABELS = {"Ac":"Altocumulus","As":"Altostratus","Cc":"Cirrocumulus","Cs":"Cirrostratus","Ci":"Cirrus","Cb":"Cumulonimbus","Cu":"Cumulus","Ns":"Nimbostratus","Sc":"Stratocumulus","St":"Stratus","Ct":"Contrail"}
_model = None
_error = None

def get_model():
    global _model, _error
    if _model is not None: return _model
    path = next((p for p in MODEL_FILES if p.exists()), None)
    if path is None:
        _error = "No trained model found. Add a real .keras or .h5 file to the model/ folder."
        return None
    try:
        from tensorflow.keras.models import load_model
        _model = load_model(path)
        _error = None
        return _model
    except Exception as exc:
        _error = f"Model loading failed: {exc}"
        return None

@app.get("/")
def home():
    return render_template("index.html")

@app.get("/api/status")
def status():
    model = get_model()
    return jsonify({"model_ready": model is not None, "message": "Model ready." if model is not None else _error})

@app.post("/api/predict")
def predict():
    uploaded = request.files.get("image")
    if not uploaded or not uploaded.filename:
        return jsonify({"error":"Please choose an image."}), 400
    model = get_model()
    if model is None:
        return jsonify({"error":"The website is running, but a trained model file is required for real predictions.", "details":_error}), 503
    try:
        import numpy as np
        from PIL import Image
        img = Image.open(io.BytesIO(uploaded.read())).convert("RGB").resize((150,150))
        x = np.expand_dims(np.asarray(img,dtype=np.float32)/255.0,axis=0)
        scores = np.asarray(model.predict(x,verbose=0))[0]
        if len(scores) != len(CLASS_NAMES):
            return jsonify({"error":f"Expected 11 outputs but model returned {len(scores)}. Check class mapping."}), 500
        indices = np.argsort(scores)[::-1][:3]
        top = [{"code":CLASS_NAMES[int(i)],"label":CLASS_LABELS[CLASS_NAMES[int(i)]],"confidence":round(float(scores[i])*100,2)} for i in indices]
        return jsonify({"prediction":top[0],"top_predictions":top})
    except Exception as exc:
        return jsonify({"error":f"Could not process image: {exc}"}), 400

if __name__ == "__main__":
    app.run(debug=True)
