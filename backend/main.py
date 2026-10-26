from __future__ import annotations

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response
from pathlib import Path
from typing import Optional
import numpy as np
from PIL import Image, UnidentifiedImageError
import io
import json
import base64
import matplotlib
matplotlib.use("Agg")
from fpdf import FPDF
import logger_config
import os
import time
import tensorflow as tf
from tensorflow.keras.models import load_model

# torchxrayvision is an optional fallback; the backend runs on TensorFlow alone
try:
    import torch
    import torchvision.transforms as transforms
    import torchxrayvision as xrv
except ImportError:
    torch = transforms = xrv = None

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

# How each model expects its input and where Grad-CAM reads features.
# "nested_model": legacy Sequential([backbone, GAP, Dropout, Dense]) models, where the
# backbone's own output is the Grad-CAM feature map.
SIMPLE_CFG = {"name": "MobileNetV2-binary", "img_size": 224, "preprocess": "scale_0_1",
              "nested_model": "mobilenetv2_1.00_224"}
LEGACY_MULTI_CFG = {"name": "CheXpert-DenseNet121-v1", "img_size": 224, "preprocess": "scale_0_1",
                    "nested_model": "densenet121", "thresholds": None}

# Legacy multi-disease decision rule (used only when the model has no tuned thresholds)
LEGACY_DISEASE_THRESHOLD = 0.08

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
    "multi": None,         # fine-tuned CheXpert model (.keras, notebook 03) or legacy DenseNet121 (.h5)
    "multi_labels": [],
    "multi_cfg": {},
    "simple": None,        # local Binary MobileNetV2 (.h5)
    "simple_map": {},
}

def load_xrv_model():
    """Load the torchxrayvision DenseNet121 pretrained on CheXpert + NIH + PadChest + MIMIC."""
    if xrv is None:
        logger.info("torchxrayvision not installed - fallback model disabled.")
        return None, []
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
            model = load_model(str(XRAY_SIMPLE_MODEL_PATH), compile=False)
            with open(XRAY_SIMPLE_CLASS_MAPPING_PATH, "r") as f:
                raw_map = json.load(f)
            class_map = {int(k): v for k, v in raw_map.items()}
            logger.info(f"✅ Simple binary model loaded: {class_map}")
            return model, class_map
    except Exception as e:
        logger.error(f"❌ Binary model error: {e}")
    return None, {}

def find_multi_meta() -> Optional[Path]:
    """Metadata of the fine-tuned model: $MULTI_MODEL_META, else the newest chexpert_*_meta.json."""
    name = os.getenv("MULTI_MODEL_META")
    if name:
        return MODELS_DIR / name
    metas = sorted(MODELS_DIR.glob("chexpert_*_meta.json"), key=lambda p: p.stat().st_mtime)
    return metas[-1] if metas else None

def load_multi_model():
    """Load the fine-tuned multi-disease model (notebook 03), falling back to the legacy .h5."""
    meta_path = find_multi_meta()
    if meta_path is not None:
        try:
            meta = json.loads(meta_path.read_text())
            logger.info(f"Loading fine-tuned multi-disease model: {meta['model_file']}...")
            model = load_model(str(MODELS_DIR / meta["model_file"]), compile=False)
            cfg = {"name": meta["name"], "img_size": meta["img_size"], "preprocess": meta["preprocess"],
                   "cam_layer": meta["cam_layer"], "nested_model": None, "thresholds": meta.get("thresholds")}
            logger.info(f"✅ Fine-tuned model loaded: {cfg['name']} | thresholds: {cfg['thresholds']}")
            return model, meta["labels"], cfg
        except Exception as e:
            logger.error(f"❌ Fine-tuned model error ({meta_path.name}): {e} - trying legacy model")
    try:
        if XRAY_MULTI_MODEL_PATH.exists() and XRAY_MULTI_LABELS_PATH.exists():
            logger.info(f"Loading local multi-disease model: {XRAY_MULTI_MODEL_PATH.name}...")
            model = load_model(str(XRAY_MULTI_MODEL_PATH), compile=False)
            with open(XRAY_MULTI_LABELS_PATH, "r") as f:
                labels = json.load(f)
            logger.info(f"✅ Local Multi-model loaded. Pathologies: {labels}")
            return model, labels, LEGACY_MULTI_CFG
    except Exception as e:
        logger.error(f"❌ Multi-model error: {e}")
    return None, [], {}

# Load models at startup
MODELS["xrv"], MODELS["xrv_labels"] = load_xrv_model()
MODELS["simple"], MODELS["simple_map"] = load_simple_model()
MODELS["multi"], MODELS["multi_labels"], MODELS["multi_cfg"] = load_multi_model()

# -----------------------------
# Image Preprocessing
# -----------------------------

def decode_upload(file_bytes: bytes) -> Image.Image:
    """Decode an uploaded image to RGB, rejecting files PIL cannot read."""
    try:
        img = Image.open(io.BytesIO(file_bytes))
        img.load()
        return img.convert("RGB")
    except (UnidentifiedImageError, OSError) as e:
        raise HTTPException(400, f"Could not read image: {e}")

def prepare_input(img: Image.Image, cfg: dict) -> np.ndarray:
    """Resize and scale an RGB image the way the model was trained."""
    size = (cfg["img_size"], cfg["img_size"])
    if cfg["preprocess"] == "rgb_0_255":
        # Same steps as the notebook 03 training pipeline: bilinear tf.image.resize on 0-255 pixels;
        # normalisation happens inside the model
        arr = tf.image.resize(np.asarray(img, dtype=np.float32), size).numpy()
    else:
        # Legacy models (notebooks 01/02): resize, then 0-1 scaling
        arr = np.asarray(img.resize(size), dtype=np.float32) / 255.0
    return np.expand_dims(arr, axis=0)

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

# -----------------------------
# Grad-CAM
# -----------------------------

GRAD_MODELS = {}

def get_grad_model(model, cfg: dict):
    """Callable mapping the input to (feature map, predictions); cached per model."""
    key = id(model)
    if key not in GRAD_MODELS:
        if cfg.get("nested_model"):
            # Legacy Sequential: run the backbone, keep its feature map, then apply the head layers
            base = model.get_layer(cfg["nested_model"])
            head = model.layers[model.layers.index(base) + 1:]

            def grad_model(x):
                features = base(x, training=False)
                out = features
                for layer in head:
                    out = layer(out, training=False)
                return features, out
        else:
            # Fine-tuned models have a flat graph, so the feature layer is reachable by name
            grad_model = tf.keras.Model(model.inputs, [model.get_layer(cfg["cam_layer"]).output, model.output])
        GRAD_MODELS[key] = grad_model
    return GRAD_MODELS[key]

def get_gradcam(model, img_array: np.ndarray, cfg: dict, class_idx: int) -> Optional[np.ndarray]:
    """Grad-CAM heatmap (0-1) for one output class of a Keras model."""
    try:
        grad_model = get_grad_model(model, cfg)
        with tf.GradientTape() as tape:
            features, preds = grad_model(tf.convert_to_tensor(img_array))
            score = preds[:, class_idx]
        grads = tape.gradient(score, features)
        # Weight each feature channel by its mean gradient, keep positive evidence only
        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
        heatmap = tf.nn.relu(tf.reduce_sum(features[0] * pooled_grads, axis=-1)).numpy()
        peak = heatmap.max()
        return heatmap / peak if peak > 0 else heatmap
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

def apply_heatmap(img: Image.Image, heatmap: np.ndarray) -> str:
    """Overlay Grad-CAM heatmap on original image and return base64 JPEG."""
    heatmap_uint = np.uint8(255 * heatmap)
    jet_colors = matplotlib.colormaps["jet"](np.arange(256))[:, :3]
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
# Inference routes are plain `def` so FastAPI runs them in a worker thread
# instead of blocking the event loop during model execution.

@app.get("/health")
def health():
    return {
        "status": "operational",
        "xrv_model": MODELS["xrv"] is not None,
        "xrv_labels": MODELS["xrv_labels"],
        "simple_model": MODELS["simple"] is not None,
        "multi_model": MODELS["multi_cfg"].get("name") if MODELS["multi"] is not None else None,
        "multi_thresholds": MODELS["multi_cfg"].get("thresholds"),
    }

@app.post("/predict-xray")
def predict_simple(file: UploadFile = File(...)):
    """Binary NORMAL/PNEUMONIA prediction using local MobileNetV2."""
    file_bytes = file.file.read()

    # Try local binary model first
    if MODELS["simple"] is not None:
        img = decode_upload(file_bytes)
        arr = prepare_input(img, SIMPLE_CFG)
        # MobileNetV2 output is sigmoid [0, 1]
        prob = float(MODELS["simple"].predict(arr, verbose=0)[0][0])
        label = MODELS["simple_map"].get(1 if prob >= 0.5 else 0, "Unknown")

        heatmap = get_gradcam(MODELS["simple"], arr, SIMPLE_CFG, class_idx=0)
        heatmap_b64 = apply_heatmap(img, heatmap) if heatmap is not None else None

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
def predict_multi(file: UploadFile = File(...)):
    """Multi-disease prediction using the fine-tuned (or legacy) CheXpert model."""
    file_bytes = file.file.read()
    if MODELS["multi"] is None:
        # Fallback to torchxrayvision if local multi-model missing
        if MODELS["xrv"] is not None:
            return predict_multi_xrv(file_bytes)
        raise HTTPException(500, "No multi-disease model loaded")

    cfg = MODELS["multi_cfg"]
    logger.info(f"Starting multi-disease prediction ({cfg['name']})...")

    img = decode_upload(file_bytes)
    arr = prepare_input(img, cfg)
    preds = MODELS["multi"].predict(arr, verbose=0)[0]    # sigmoid probabilities
    labels = MODELS["multi_labels"]

    predictions = {labels[i]: float(preds[i]) for i in range(len(labels))}
    sorted_preds = sorted(predictions.items(), key=lambda x: x[1], reverse=True)
    no_finding_prob = predictions.get("No Finding", 0.0)
    disease_preds = [(l, p) for l, p in sorted_preds if l != "No Finding"]
    top_disease_label, top_disease_prob = disease_preds[0] if disease_preds else ("None", 0.0)

    thresholds = cfg.get("thresholds")
    if thresholds:
        # Per-condition thresholds tuned on validation data (notebook 03)
        findings = [{"label": l, "prob": p, "threshold": thresholds[l]}
                    for l, p in disease_preds if l in thresholds and p >= thresholds[l]]
        is_normal = not findings
    else:
        # Legacy rule: pathological if any disease exceeds 8% (high recall),
        # unless "No Finding" is likely and every disease stays under that bar
        findings = [{"label": l, "prob": p, "threshold": LEGACY_DISEASE_THRESHOLD}
                    for l, p in disease_preds if p >= LEGACY_DISEASE_THRESHOLD]
        is_normal = no_finding_prob > 0.5 and top_disease_prob < LEGACY_DISEASE_THRESHOLD

    logger.info(f"Disease Top: {top_disease_label}={top_disease_prob:.3f} | findings={[f['label'] for f in findings]} | is_normal={is_normal}")

    # Grad-CAM explains the most likely disease, not "No Finding"
    cam_idx = labels.index(top_disease_label) if top_disease_label in labels else int(np.argmax(preds))
    heatmap = get_gradcam(MODELS["multi"], arr, cfg, cam_idx)
    heatmap_b64 = apply_heatmap(img, heatmap) if heatmap is not None else None

    return {
        "predictions": predictions,
        "is_normal": is_normal,
        "no_finding_prob": float(no_finding_prob),
        "top_label": top_disease_label,
        "top_prob": float(top_disease_prob),
        "top3": sorted_preds[:3],
        "findings": findings,
        "thresholds": thresholds,
        "model": cfg["name"],
        "heatmap": heatmap_b64
    }

def predict_multi_xrv(file_bytes: bytes):
    """Fallback predict function using torchxrayvision."""
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
        "findings": [{"label": l, "prob": p, "threshold": 0.15} for l, p in sorted_preds if p >= 0.15],
        "thresholds": None,
        "model": "torchxrayvision-densenet121-res224-all",
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
