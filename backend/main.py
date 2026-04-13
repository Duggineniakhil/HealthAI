from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response
from pathlib import Path
from typing import Dict, List, Optional
import numpy as np
import torch
import torchvision.transforms as transforms
import torchxrayvision as xrv
from PIL import Image
import io
import json
import base64
import matplotlib
matplotlib.use("Agg")
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

# Legacy simple model paths (still used for binary pneumonia endpoint)
XRAY_SIMPLE_MODEL_PATH = MODELS_DIR / "xray_disease_model.h5"
XRAY_SIMPLE_CLASS_MAPPING_PATH = MODELS_DIR / "xray_class_mapping.json"

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

MODELS = {
    "xrv": None,           # torchxrayvision DenseNet121 (multi-disease)
    "xrv_labels": [],
    "simple": None,        # legacy TF binary model (optional)
    "simple_map": {},
}

def load_xrv_model():
    """Load the torchxrayvision DenseNet121 pretrained on CheXpert + NIH + PadChest + MIMIC."""
    try:
        logger.info("Loading torchxrayvision DenseNet121 (all datasets)...")
        model = xrv.models.DenseNet(weights="densenet121-res224-all")
        model.eval()
        # Labels from the model
        labels = model.pathologies  # list of condition strings
        logger.info(f"✅ torchxrayvision model loaded. Pathologies: {list(labels)}")
        return model, list(labels)
    except Exception as e:
        logger.error(f"❌ torchxrayvision model error: {e}")
        return None, []

def load_simple_model():
    """Load legacy TF binary model (optional fallback)."""
    try:
        from tensorflow.keras.models import load_model
        if XRAY_SIMPLE_MODEL_PATH.exists() and XRAY_SIMPLE_CLASS_MAPPING_PATH.exists():
            model = load_model(XRAY_SIMPLE_MODEL_PATH)
            with open(XRAY_SIMPLE_CLASS_MAPPING_PATH, "r") as f:
                raw_map = json.load(f)
            class_map = {int(k): v for k, v in raw_map.items()}
            logger.info(f"✅ Simple binary model loaded: {class_map}")
            return model, class_map
    except Exception as e:
        logger.error(f"⚠️ Simple model not loaded (non-critical): {e}")
    return None, {}

# Load models at startup
MODELS["xrv"], MODELS["xrv_labels"] = load_xrv_model()
MODELS["simple"], MODELS["simple_map"] = load_simple_model()

# -----------------------------
# Image Preprocessing
# -----------------------------

def preprocess_xrv(file_bytes: bytes) -> torch.Tensor:
    """Preprocess image for torchxrayvision: converts to single-channel [-1024, 1024] range."""
    img = Image.open(io.BytesIO(file_bytes)).convert("L")  # grayscale
    img_np = np.array(img).astype(np.float32)
    # Normalize to [-1024, 1024] as expected by torchxrayvision
    img_np = xrv.datasets.normalize(img_np, maxval=255, reshape=True)  # shape: (1, H, W)
    transform = transforms.Compose([
        xrv.datasets.XRayCenterCrop(),
        xrv.datasets.XRayResizer(224),
    ])
    img_tensor = transform(img_np)  # shape: (1, 224, 224)
    return torch.from_numpy(img_tensor).unsqueeze(0)  # shape: (1, 1, 224, 224)

def preprocess_simple(file_bytes: bytes) -> np.ndarray:
    """Preprocess image for legacy TF model."""
    img = Image.open(io.BytesIO(file_bytes)).convert("RGB")
    img = img.resize(IMG_SIZE)
    return np.expand_dims(np.array(img) / 255.0, axis=0)

# -----------------------------
# Grad-CAM for torchxrayvision
# -----------------------------

def get_gradcam_xrv(model, img_tensor: torch.Tensor, class_idx: int):
    """Compute Grad-CAM for torchxrayvision DenseNet."""
    try:
        # Hook to capture gradients and activations from the last dense block
        activations = {}
        gradients = {}

        def forward_hook(module, input, output):
            activations["value"] = output.detach()

        def backward_hook(module, grad_input, grad_output):
            gradients["value"] = grad_output[0].detach()

        # Register on the last conv layer inside the features block
        target_layer = model.features.denseblock4.denselayer16.conv2
        fwd_handle = target_layer.register_forward_hook(forward_hook)
        bwd_handle = target_layer.register_full_backward_hook(backward_hook)

        img_tensor.requires_grad_(True)
        output = model(img_tensor)  # shape: (1, num_pathologies)

        model.zero_grad()
        # Backprop on the target class
        score = output[0, class_idx]
        score.backward()

        fwd_handle.remove()
        bwd_handle.remove()

        acts = activations["value"][0]   # (C, H, W)
        grads = gradients["value"][0]    # (C, H, W)

        pooled_grads = grads.mean(dim=(1, 2))  # (C,)
        cam = (acts * pooled_grads[:, None, None]).sum(dim=0)  # (H, W)
        cam = torch.relu(cam)
        cam = cam - cam.min()
        if cam.max() > 0:
            cam = cam / cam.max()
        return cam.numpy()
    except Exception as e:
        logger.error(f"Grad-CAM error: {e}")
        return None

def apply_heatmap(file_bytes: bytes, heatmap: np.ndarray) -> str:
    """Overlay Grad-CAM heatmap on original image and return base64 JPEG."""
    img = Image.open(io.BytesIO(file_bytes)).convert("RGB")
    heatmap_uint = np.uint8(255 * heatmap)
    jet = cm.get_cmap("jet")
    jet_colors = jet(np.arange(256))[:, :3]
    jet_heatmap = jet_colors[heatmap_uint]
    jet_img = Image.fromarray(np.uint8(jet_heatmap * 255)).resize((img.width, img.height))
    blended = Image.blend(img, jet_img, alpha=0.45)
    buf = io.BytesIO()
    blended.save(buf, format="JPEG", quality=88)
    return base64.b64encode(buf.getvalue()).decode()

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
    for cond, prob in sorted(predictions.items(), key=lambda x: x[1], reverse=True):
        pdf.cell(100, 10, cond, 1)
        pdf.cell(40, 10, f"{prob*100:.2f}%", 1)
        pdf.ln()
    pdf.ln(20)
    pdf.set_font("Arial", 'I', 10)
    pdf.multi_cell(0, 10, "Disclaimer: This report is AI-generated and intended for clinical decision support only. Review by a licensed medical professional is mandatory.")
    raw_pdf = pdf.output(dest='S')
    if isinstance(raw_pdf, bytearray):
        return bytes(raw_pdf)
    elif isinstance(raw_pdf, str):
        return raw_pdf.encode('latin-1')
    return raw_pdf

# -----------------------------
# Routes
# -----------------------------

@app.get("/health")
def health():
    return {
        "status": "operational",
        "xrv_model": MODELS["xrv"] is not None,
        "xrv_labels": MODELS["xrv_labels"],
        "simple_model": MODELS["simple"] is not None,
    }

@app.post("/predict-xray")
async def predict_simple(file: UploadFile = File(...)):
    """Binary NORMAL/PNEUMONIA prediction. Uses torchxrayvision if simple model unavailable."""
    file_bytes = await file.read()

    # Try torchxrayvision first (more reliable)
    if MODELS["xrv"] is not None:
        img_tensor = preprocess_xrv(file_bytes)
        with torch.no_grad():
            preds = torch.sigmoid(MODELS["xrv"](img_tensor))[0].numpy()
        labels = MODELS["xrv_labels"]
        # Get pneumonia score
        pneumonia_idx = next((i for i, l in enumerate(labels) if "pneumonia" in l.lower()), None)
        pneumonia_prob = float(preds[pneumonia_idx]) if pneumonia_idx is not None else 0.5
        label = "PNEUMONIA" if pneumonia_prob >= 0.5 else "NORMAL"

        # Grad-CAM
        img_tensor2 = preprocess_xrv(file_bytes)
        top_idx = int(np.argmax(preds))
        heatmap = get_gradcam_xrv(MODELS["xrv"], img_tensor2, top_idx)
        heatmap_b64 = apply_heatmap(file_bytes, heatmap) if heatmap is not None else None

        return {
            "predicted_label": label,
            "pneumonia_probability": pneumonia_prob,
            "heatmap": heatmap_b64
        }

    # Fallback to legacy TF model
    if MODELS["simple"] is not None:
        arr = preprocess_simple(file_bytes)
        prob = float(MODELS["simple"].predict(arr)[0][0])
        return {
            "predicted_label": MODELS["simple_map"].get(1 if prob >= 0.5 else 0, "Unknown"),
            "pneumonia_probability": prob,
            "heatmap": None
        }

    raise HTTPException(500, "No model available")


@app.post("/predict-xray-multidisease")
async def predict_multi(file: UploadFile = File(...)):
    """Multi-disease prediction using torchxrayvision DenseNet121."""
    if MODELS["xrv"] is None:
        raise HTTPException(500, "torchxrayvision model not loaded")

    file_bytes = await file.read()
    logger.info("Starting torchxrayvision multi-disease prediction...")

    img_tensor = preprocess_xrv(file_bytes)

    with torch.no_grad():
        raw_out = MODELS["xrv"](img_tensor)          # raw logits
        probs = torch.sigmoid(raw_out)[0].numpy()    # sigmoid probabilities

    labels = MODELS["xrv_labels"]
    raw_predictions = {labels[i]: float(probs[i]) for i in range(len(labels))}

    # ------------------------------------------------------------------
    # Filter out None/empty labels (some xrv models have blank slots)
    # ------------------------------------------------------------------
    predictions = {k: v for k, v in raw_predictions.items() if k and k.strip()}

    # Sort by probability
    sorted_preds = sorted(predictions.items(), key=lambda x: x[1], reverse=True)
    top_label, top_prob = sorted_preds[0]

    # ------------------------------------------------------------------
    # Determine if the scan is likely normal:
    # Normal = no disease scores >= 0.15 AND top score < 0.2
    # ------------------------------------------------------------------
    is_normal = top_prob < 0.15

    logger.info(f"Top prediction: {top_label} = {top_prob:.3f} | is_normal={is_normal}")
    logger.info(f"Top 5: {sorted_preds[:5]}")

    # Grad-CAM for top predicted class
    logger.info("Generating Grad-CAM...")
    img_tensor2 = preprocess_xrv(file_bytes)
    top_idx = labels.index(top_label) if top_label in labels else 0
    heatmap = get_gradcam_xrv(MODELS["xrv"], img_tensor2, top_idx)
    heatmap_b64 = apply_heatmap(file_bytes, heatmap) if heatmap is not None else None

    return {
        "predictions": predictions,
        "raw_predictions": raw_predictions,
        "no_finding_prob": 0.0,  # xrv doesn't have a "No Finding" label; use is_normal instead
        "heatmap": heatmap_b64,
        "top3": sorted_preds[:3],
        "is_normal": is_normal,
        "top_label": top_label,
        "top_prob": float(top_prob),
    }


@app.post("/generate-report")
async def generate_report(data: dict):
    pdf_bytes = create_pdf_report(data['patient'], data['results'])
    return Response(
        content=pdf_bytes,
        media_type="application/pdf",
        headers={"Content-Disposition": "attachment; filename=HealthAI_Report.pdf"}
    )


if __name__ == "__main__":
    import uvicorn
    logger.info("🚀 Starting HealthAI Production Server...")
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
