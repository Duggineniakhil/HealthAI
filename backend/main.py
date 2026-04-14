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
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input as preprocess_mobilenet
from tensorflow.keras.applications.densenet import preprocess_input as preprocess_densenet

def preprocess_simple(img_array):
    """Simple 0-1 scaling as used in training for the binary model."""
    img_array = img_array / 255.0
    return img_array

def preprocess_local_multi(img_array):
    """Simple 0-1 scaling as used in training for the multi-disease model."""
    img_array = img_array / 255.0
    return img_array

logger = logger_config.logger

# -----------------------------
# Paths & Init
# -----------------------------

BASE_DIR = Path(__file__).resolve().parent.parent
MODELS_DIR = Path(os.getenv("MODELS_DIR", str(BASE_DIR / "models")))

# Local model paths
XRAY_MULTI_MODEL_PATH = MODELS_DIR / "xray_chexpert_multidisease_model.h5"
XRAY_MULTI_LABELS_PATH = MODELS_DIR / "xray_chexpert_labels.json"
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
    "xrv": None,           # torchxrayvision DenseNet121 (secondary fallback)
    "xrv_labels": [],
    "multi": None,         # local CheXpert DenseNet121 (.h5)
    "multi_labels": [],
    "simple": None,        # local Binary MobileNetV2 (.h5)
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
    """Load local TF binary model (primary)."""
    try:
        if XRAY_SIMPLE_MODEL_PATH.exists() and XRAY_SIMPLE_CLASS_MAPPING_PATH.exists():
            logger.info(f"Loading local binary model: {XRAY_SIMPLE_MODEL_PATH.name}...")
            model = load_model(str(XRAY_SIMPLE_MODEL_PATH))
            with open(XRAY_SIMPLE_CLASS_MAPPING_PATH, "r") as f:
                raw_map = json.load(f)
            class_map = {int(k): v for k, v in raw_map.items()}
            logger.info(f"✅ Simple binary model loaded: {class_map}")
            return model, class_map
    except Exception as e:
        logger.error(f"❌ Binary model error: {e}")
    return None, {}

def load_multi_model():
    """Load local multi-disease model (primary)."""
    try:
        if XRAY_MULTI_MODEL_PATH.exists() and XRAY_MULTI_LABELS_PATH.exists():
            logger.info(f"Loading local multi-disease model: {XRAY_MULTI_MODEL_PATH.name}...")
            model = load_model(str(XRAY_MULTI_MODEL_PATH))
            with open(XRAY_MULTI_LABELS_PATH, "r") as f:
                labels = json.load(f)
            logger.info(f"✅ Local Multi-model loaded. Pathologies: {labels}")
            return model, labels
    except Exception as e:
        logger.error(f"❌ Multi-model error: {e}")
    return None, []

# Load models at startup
MODELS["xrv"], MODELS["xrv_labels"] = load_xrv_model()
MODELS["simple"], MODELS["simple_map"] = load_simple_model()
MODELS["multi"], MODELS["multi_labels"] = load_multi_model()

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

def preprocess_simple_wrapper(file_bytes: bytes) -> np.ndarray:
    """Preprocess image for local MobileNetV2 binary model."""
    img = Image.open(io.BytesIO(file_bytes)).convert("RGB")
    img = img.resize(IMG_SIZE)
    img_array = np.array(img).astype(np.float32)
    return np.expand_dims(preprocess_simple(img_array), axis=0)

def preprocess_local_multi_wrapper(file_bytes: bytes) -> np.ndarray:
    """Preprocess image for local DenseNet121 multi-disease model."""
    img = Image.open(io.BytesIO(file_bytes)).convert("RGB")
    img = img.resize(IMG_SIZE)
    img_array = np.array(img).astype(np.float32)
    return np.expand_dims(preprocess_local_multi(img_array), axis=0)

# -----------------------------
# Grad-CAM for torchxrayvision
# -----------------------------

def get_gradcam_keras(model, img_array: np.ndarray, last_conv_layer_name: str, nested_model_name: Optional[str] = None):
    """Compute Grad-CAM for a Keras model (supports nested functional models)."""
    try:
        # If it's a nested model (like densenet121 inside a wrapper)
        if nested_model_name:
            base_model = model.get_layer(nested_model_name)
        else:
            base_model = model

        # Create a model that maps the input to the activations of the last conv layer as well as the output predictions
        grad_model = tf.keras.models.Model(
            [base_model.inputs], [base_model.get_layer(last_conv_layer_name).output, base_model.output]
        )

        # If the input was to the outer wrapper, we need to handle that. 
        # But usually we can just use the base_model if weights are shared.
        
        with tf.GradientTape() as tape:
            last_conv_layer_output, preds = grad_model(img_array)
            class_channel = preds[:, np.argmax(preds[0])]

        # This is the gradient of the top predicted class with regard to
        # the output feature map of the last conv layer
        grads = tape.gradient(class_channel, last_conv_layer_output)

        # This is a vector where each entry is the mean intensity of the gradient
        # over a specific feature map channel
        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

        # We multiply each channel in the feature map array
        # by "how important this channel is" with regard to the top predicted class
        # then sum all the channels to obtain the heatmap class activation
        last_conv_layer_output = last_conv_layer_output[0]
        heatmap = last_conv_layer_output @ pooled_grads[..., tf.newaxis]
        heatmap = tf.squeeze(heatmap)

        # For visualization, we will also normalize the heatmap between 0 & 1
        heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
        return heatmap.numpy()
    except Exception as e:
        logger.error(f"Keras Grad-CAM error: {e}")
        return None

def get_gradcam_xrv(model, img_tensor: torch.Tensor, class_idx: int):
    """Compute Grad-CAM for torchxrayvision DenseNet (Fallback)."""
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
    """Binary NORMAL/PNEUMONIA prediction using local MobileNetV2."""
    file_bytes = await file.read()

    # Try local binary model first
    if MODELS["simple"] is not None:
        arr = preprocess_simple_wrapper(file_bytes)
        # MobileNetV2 output is sigmoid [0, 1]
        prob = float(MODELS["simple"].predict(arr, verbose=0)[0][0])
        label = MODELS["simple_map"].get(1 if prob >= 0.5 else 0, "Unknown")
        
        # Grad-CAM for Keras MobileNetV2
        heatmap = get_gradcam_keras(MODELS["simple"], arr, "Conv_1", "mobilenetv2_1.00_224")
        heatmap_b64 = apply_heatmap(file_bytes, heatmap) if heatmap is not None else None

        logger.info(f"Binary Prediction: {label} (prob={prob:.4f})")
        return {
            "predicted_label": label,
            "pneumonia_probability": prob,
            "heatmap": heatmap_b64
        }

    # Fallback to torchxrayvision if local model missing
    if MODELS["xrv"] is not None:
        img_tensor = preprocess_xrv(file_bytes)
        with torch.no_grad():
            preds = torch.sigmoid(MODELS["xrv"](img_tensor))[0].numpy()
        labels = MODELS["xrv_labels"]
        pneumonia_idx = next((i for i, l in enumerate(labels) if "pneumonia" in l.lower()), None)
        pneumonia_prob = float(preds[pneumonia_idx]) if pneumonia_idx is not None else 0.5
        label = "PNEUMONIA" if pneumonia_prob >= 0.5 else "NORMAL"

        return {
            "predicted_label": label,
            "pneumonia_probability": pneumonia_prob,
            "heatmap": None
        }

    raise HTTPException(500, "No binary model available")


@app.post("/predict-xray-multidisease")
async def predict_multi(file: UploadFile = File(...)):
    """Multi-disease prediction using local CheXpert DenseNet121 model."""
    if MODELS["multi"] is None:
        # Fallback to torchxrayvision if local multi-model missing
        if MODELS["xrv"] is not None:
            return await predict_multi_xrv(file)
        raise HTTPException(500, "No multi-disease model loaded")

    file_bytes = await file.read()
    logger.info("Starting local multi-disease prediction...")

    arr = preprocess_local_multi_wrapper(file_bytes)
    preds = MODELS["multi"].predict(arr, verbose=0)[0]    # sigmoid probabilities
    labels = MODELS["multi_labels"]
    
    predictions = {labels[i]: float(preds[i]) for i in range(len(labels))}
    sorted_preds = sorted(predictions.items(), key=lambda x: x[1], reverse=True)
    top_label, top_prob = sorted_preds[0]

    # Robust "is_normal" logic: 
    # High probability of "No Finding" OR low probability for all disease classes
    no_finding_prob = predictions.get("No Finding", 0.0)
    
    # Condition is normal if top label is "No Finding" OR top disease probability is very low
    top_disease_label, top_disease_prob = next(((l, p) for l, p in sorted_preds if l != "No Finding"), ("None", 0.0))
    
    # 4. Calibrate "Normal" verdict
    # SIGNIFICANT CHANGE: Lowering threshold to 8% (0.08) for high recall
    # If the top disease is > 0.08 and No Finding isn't overwhelmingly high (> 0.7), it's pathological.
    is_normal = (no_finding_prob > 0.5 and top_disease_prob < 0.08)
    
    if top_disease_prob > 0.25:
         is_normal = False # Confident pathology

    logger.info(f"Local Top: {top_label}={top_prob:.3f} | Disease Top: {top_disease_label}={top_disease_prob:.3f} | is_normal={is_normal}")

    # Grad-CAM for local DenseNet121 model
    logger.info("Generating local Grad-CAM...")
    heatmap = get_gradcam_keras(MODELS["multi"], arr, "conv5_block16_2_conv", "densenet121")
    heatmap_b64 = apply_heatmap(file_bytes, heatmap) if heatmap is not None else None

    return {
        "predictions": predictions,
        "is_normal": is_normal,
        "no_finding_prob": float(no_finding_prob),
        "top_label": top_disease_label,
        "top_prob": float(top_disease_prob),
        "top3": sorted_preds[:3],
        "heatmap": heatmap_b64
    }

async def predict_multi_xrv(file: UploadFile):
    """Fallback predict function using torchxrayvision."""
    file_bytes = await file.read()
    img_tensor = preprocess_xrv(file_bytes)
    with torch.no_grad():
        raw_out = MODELS["xrv"](img_tensor)
        probs = torch.sigmoid(raw_out)[0].numpy()
    labels = MODELS["xrv_labels"]
    predictions = {labels[i]: float(probs[i]) for i in range(len(labels)) if labels[i] and labels[i].strip()}
    sorted_preds = sorted(predictions.items(), key=lambda x: x[1], reverse=True)
    top_label, top_prob = sorted_preds[0]
    return {
        "predictions": predictions,
        "is_normal": top_prob < 0.15,
        "no_finding_prob": 0.0,
        "top_label": top_label,
        "top_prob": float(top_prob),
        "top3": sorted_preds[:3],
        "heatmap": None
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
