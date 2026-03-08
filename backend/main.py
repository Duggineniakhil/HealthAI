from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response
from pathlib import Path
from typing import Dict, List, Optional
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model, Model
from PIL import Image
import io
import json
import base64
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from fpdf import FPDF
import logger_config
import os
import time

logger = logger_config.logger

# -----------------------------
# Paths & Init
# -----------------------------

BASE_DIR = Path(__file__).resolve().parent.parent
MODELS_DIR = Path(os.getenv("MODELS_DIR", str(BASE_DIR / "models")))

XRAY_SIMPLE_MODEL_PATH = MODELS_DIR / "xray_disease_model.h5"
XRAY_SIMPLE_CLASS_MAPPING_PATH = MODELS_DIR / "xray_class_mapping.json"
CHEXPERT_MODEL_PATH = MODELS_DIR / "xray_chexpert_multidisease_model.h5"
CHEXPERT_LABELS_PATH = MODELS_DIR / "xray_chexpert_labels.json"

IMG_SIZE = (224, 224)

app = FastAPI(title="HealthAI Pro Backend")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"])

@app.middleware("http")
async def log_requests(request, call_next):
    start_time = time.time()
    logger.info(f"START: {request.method} {request.url.path}")
    response = await call_next(request)
    duration = time.time() - start_time
    logger.info(f"COMPLETE: {request.url.path} | Time: {duration:.2f}s | Status: {response.status_code}")
    return response

# -----------------------------
# Model Loading
# -----------------------------

def find_last_conv_layer(model):
    """
    Search for the last Conv2D layer in a model or its nested base models.
    """
    for layer in reversed(model.layers):
        if isinstance(layer, tf.keras.layers.Conv2D):
            return layer.name
        
    # Search deeper in case of transfer learning wrappers
    for layer in model.layers:
        if hasattr(layer, 'layers'):
            for sl in reversed(layer.layers):
                if isinstance(sl, tf.keras.layers.Conv2D):
                    return sl.name
    return None

def load_healthai_models():
    models = {"simple": None, "simple_map": {}, "simple_conv": None, "chexpert": None, "chexpert_labels": [], "chexpert_conv": None}
    
    try:
        if XRAY_SIMPLE_MODEL_PATH.exists() and XRAY_SIMPLE_CLASS_MAPPING_PATH.exists():
            models["simple"] = load_model(XRAY_SIMPLE_MODEL_PATH)
            models["simple_conv"] = find_last_conv_layer(models["simple"])
            with open(XRAY_SIMPLE_CLASS_MAPPING_PATH, "r") as f:
                raw_map = json.load(f)
                models["simple_map"] = {int(k): v for k, v in raw_map.items()}
            logger.info(f"✅ Simple model loaded. Conv Layer: {models['simple_conv']}")
    except Exception as e:
        logger.error(f"❌ Simple model error: {e}")

    try:
        if CHEXPERT_MODEL_PATH.exists() and CHEXPERT_LABELS_PATH.exists():
            models["chexpert"] = load_model(CHEXPERT_MODEL_PATH)
            models["chexpert_conv"] = find_last_conv_layer(models["chexpert"])
            with open(CHEXPERT_LABELS_PATH, "r") as f:
                models["chexpert_labels"] = json.load(f)
            logger.info(f"✅ CheXpert model loaded. Conv Layer: {models['chexpert_conv']}")
    except Exception as e:
        logger.error(f"❌ CheXpert model error: {e}")
        
    return models

MODELS = load_healthai_models()

# -----------------------------
# AI Intelligence (Grad-CAM)
# -----------------------------

def get_gradcam_heatmap(model, img_array, last_conv_layer_name):
    if not last_conv_layer_name:
        return None
    
    try:
        # Find the actual layer (could be nested inside a layer named 'efficientnetb0')
        target_layer = None
        main_model_has_it = False
        try:
            model.get_layer(last_conv_layer_name)
            main_model_has_it = True
        except:
            pass

        if main_model_has_it:
            grad_model = Model([model.inputs], [model.get_layer(last_conv_layer_name).output, model.output])
        else:
            # Look inside nested layers
            base_model = None
            for layer in model.layers:
                if hasattr(layer, 'layers'):
                    try:
                        layer.get_layer(last_conv_layer_name)
                        base_model = layer
                        break
                    except:
                        pass
            
            if not base_model: return None
            grad_model = Model([base_model.inputs], [base_model.get_layer(last_conv_layer_name).output, base_model.output])
            # We need to map the full model input if we used base_model inputs
            # Simplified for now: assume input matches
            
        with tf.GradientTape() as tape:
            last_conv_layer_output, preds = grad_model(img_array)
            class_channel = tf.argmax(preds[0])

        grads = tape.gradient(preds, last_conv_layer_output)
        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

        last_conv_layer_output = last_conv_layer_output[0]
        heatmap = last_conv_layer_output @ pooled_grads[..., tf.newaxis]
        heatmap = tf.squeeze(heatmap)

        heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
        return heatmap.numpy()
    except Exception as e:
        logger.error(f"Grad-CAM Error: {e}")
        return None

def apply_heatmap_to_image(img_bytes, heatmap):
    img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    
    heatmap = np.uint8(255 * heatmap)
    jet = cm.get_cmap("jet")
    jet_colors = jet(np.arange(256))[:, :3]
    jet_heatmap = jet_colors[heatmap]

    jet_heatmap = Image.fromarray(np.uint8(jet_heatmap * 255)).resize((img.width, img.height))
    # Blend with original
    superimposed_img = Image.blend(img, jet_heatmap, alpha=0.5)
    
    buffered = io.BytesIO()
    superimposed_img.save(buffered, format="JPEG", quality=85)
    return base64.b64encode(buffered.getvalue()).decode()

# -----------------------------
# Diagnostic Reports (PDF)
# -----------------------------

class HealthAIReport(FPDF):
    def header(self):
        self.set_font('Arial', 'B', 15)
        self.cell(0, 10, 'HealthAI Clinical Diagnostic Report', 0, 1, 'C')
        self.ln(10)

def create_pdf_report(patient_info: dict, predictions: dict):
    pdf = HealthAIReport()
    pdf.add_page()
    pdf.set_font("Arial", size=12)
    
    pdf.cell(0, 10, f"Patient ID: {patient_info.get('id', 'N/A')}", ln=1)
    pdf.cell(0, 10, f"Age: {patient_info.get('age', 'N/A')} | Gender: {patient_info.get('gender', 'N/A')}", ln=1)
    pdf.ln(10)
    
    pdf.set_font("Arial", 'B', 12)
    pdf.cell(100, 10, "Condition", 1)
    pdf.cell(40, 10, "Probability", 1)
    pdf.ln()
    
    pdf.set_font("Arial", size=12)
    for cond, prob in predictions.items():
        pdf.cell(100, 10, cond, 1)
        pdf.cell(40, 10, f"{prob*100:.2f}%", 1)
        pdf.ln()
    
    pdf.ln(20)
    pdf.set_font("Arial", 'I', 10)
    pdf.multi_cell(0, 10, "Disclaimer: This report is generated by an Artificial Intelligent system and is intended for clinical decision support. Review by a medical professional is mandatory.")
    
    return pdf.output(dest='S')

# -----------------------------
# Routes
# -----------------------------

@app.get("/health")
def health():
    return {"status": "operational", "models": {k: v is not None for k, v in MODELS.items() if k in ["simple", "chexpert"]}}

def preprocess(file_bytes):
    img = Image.open(io.BytesIO(file_bytes)).convert("RGB")
    img = img.resize(IMG_SIZE)
    return np.expand_dims(np.array(img) / 255.0, axis=0)

@app.post("/predict-xray")
async def predict_simple(file: UploadFile = File(...)):
    if not MODELS["simple"]: raise HTTPException(500, "Simple model not loaded")
    
    bytes = await file.read()
    arr = preprocess(bytes)
    
    logger.info("Starting simple prediction...")
    res = MODELS["simple"].predict(arr)
    prob = float(res[0][0])
    
    logger.info("Generating Grad-CAM for simple model...")
    heatmap = get_gradcam_heatmap(MODELS["simple"], arr, MODELS["simple_conv"])
    heatmap_b64 = apply_heatmap_to_image(bytes, heatmap) if heatmap is not None else None
    
    return {
        "predicted_label": MODELS["simple_map"].get(1 if prob >= 0.5 else 0, "Unknown"),
        "pneumonia_probability": prob,
        "heatmap": heatmap_b64
    }

@app.post("/predict-xray-multidisease")
async def predict_multi(file: UploadFile = File(...)):
    if not MODELS["chexpert"]: raise HTTPException(500, "CheXpert model not loaded")
    
    bytes = await file.read()
    arr = preprocess(bytes)
    
    logger.info("Starting multi-disease prediction...")
    probs = MODELS["chexpert"].predict(arr)[0]
    predictions = {MODELS["chexpert_labels"][i]: float(probs[i]) for i in range(len(MODELS["chexpert_labels"]))}
    
    logger.info("Generating Grad-CAM for multi-disease model...")
    heatmap = get_gradcam_heatmap(MODELS["chexpert"], arr, MODELS["chexpert_conv"])
    heatmap_b64 = apply_heatmap_to_image(bytes, heatmap) if heatmap is not None else None
    
    return {
        "predictions": predictions,
        "heatmap": heatmap_b64,
        "top3": sorted(predictions.items(), key=lambda x: x[1], reverse=True)[:3]
    }

@app.post("/generate-report")
async def generate_report(data: dict):
    pdf_bytes = create_pdf_report(data['patient'], data['results'])
    return Response(content=pdf_bytes, media_type="application/pdf", headers={"Content-Disposition": f"attachment; filename=HealthAI_Report.pdf"})

if __name__ == "__main__":
    import uvicorn
    logger.info("🚀 Starting HealthAI Production Server...")
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
