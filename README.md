# HealthAI – Clinical Neural Diagnostic System

**HealthAI** is a full-stack medical analysis workstation that bridges advanced Deep Learning with a state-of-the-art clinical user interface. It is designed for high-sensitivity screening of chest X-ray pathologies, featuring a robust backend for neural inference and a high-performance frontend for clinical patient management.


## The Full-Stack Architecture

HealthAI is built as a modular, three-tier ecosystem:

### 1. Machine Learning Engine (The Core)
*   **Architectures**: 
    *   **DenseNet121**: Multi-disease classification (8 conditions) trained on the **CheXpert** dataset.
    *   **MobileNetV2**: High-speed binary screening (Normal vs Pneumonia).
*   **Neural Visualization**: Gradient-weighted Class Activation Mapping (**Grad-CAM**) built natively in TensorFlow for real-time heatmap saliency.
*   **Clinical Calibration**: Each condition gets its own decision threshold, tuned on validation data for **high recall** (90% sensitivity by default) so subtle findings are flagged during screening. The legacy model uses a single 8% threshold instead.
*   **Fallback Intelligence**: If no local model is available, the **torchxrayvision** library (optional) serves predictions instead.

### 2. FastAPI Backend (The Bridge)
*   **Asynchronous Processing**: Handles heavy ML workloads without blocking the UI.
*   **Neural Pipelines**: Custom preprocessing wrappers for different model architectures (Rescaling, Resize, Normalization).
*   **Grad-CAM Generator**: Dynamic heatmap rendering and Base64 image encoding for seamless UI updates.

### 3. Glassmorphic Frontend (The Experience)
*   **Streamlit Pro UI**: A customized Streamlit interface featuring advanced **Vanilla CSS** for a premium "Glassmorphic" aesthetic.
*   **Interactive Diagnostic Workstation**: 3-column layout focused on image analysis, probability distribution, and patient findings.
*   **Neural Overlays**: Dynamic rendering of Grad-CAM heatmaps over original X-rays.

---

## Data & Lifecycle Management

*   **Clinical Persistence**: A JSON-backed persistence engine used for patient registration, session tracking, and historical logging.
*   **Automated Reporting**: Integrated **FPDF** engine that generates professional clinical PDF reports for every analysis session.
*   **ID Persistence**: Automated unique clinical ID generation for every new patient registration.

---

## Technical Highlights

| Feature | Technology |
| :--- | :--- |
| **Deep Learning** | TensorFlow 2.x, Keras, torchxrayvision |
| **Backend API** | FastAPI, Uvicorn, Python 3.10+ |
| **Frontend UI** | Streamlit, Custom CSS (Glassmorphism), Plotly |
| **Reporting** | FPDF, Base64 Encoding |
| **Computer Vision** | OpenCV, PIL (Pillow), Matplotlib |

---

## Training a Better Multi-Disease Model

`notebooks/03_chexpert_finetune.ipynb` fine-tunes a full backbone on CheXpert in TensorFlow. You can choose DenseNet121, EfficientNetV2-S or ConvNeXt-Tiny.

Compared with notebook 02, it trains on all frontal images at 320 px and splits train/validation by patient. It reports per-condition AUROC on the official validation set and tunes a decision threshold for each condition.

It is written for a free Kaggle GPU (T4 x2); the setup steps are at the top of the notebook. To check the pipeline on CPU first, set `SMOKE_TEST = True`; this takes a few minutes.

To deploy, copy the two output files into `models/`:

```
models/chexpert_densenet121_320.keras
models/chexpert_densenet121_320_meta.json
```

On startup, the backend loads the newest `chexpert_*_meta.json`. To choose a specific one, set `MULTI_MODEL_META=<file name>`. When the metadata contains tuned thresholds, the backend uses them for the normal/abnormal decision. If no metadata file is found, it falls back to the legacy `.h5` model.

---

## Getting Started

Model files (`*.h5`, `*.keras`) are too large for git, so they are not in this repository. Put them in `models/` before starting the backend. To produce them, train with the notebooks or copy them from your own storage.

### 1. Launch AI Backend
```bash
cd backend
pip install -r requirements.txt
python main.py
```

The backend runs on TensorFlow alone. To enable the torchxrayvision fallback, also install `torch`, `torchvision` and `torchxrayvision`.

### 2. Launch Clinical Dashboard
```bash
cd dashboard
pip install -r requirements.txt
streamlit run app.py
```

---

## Author
**Duggineni Akhil**  
*Lead Developer | AI Researcher*  
[LinkedIn](https://linkedin.com/in/akhil-duggineni) | [GitHub](https://github.com/Duggineniakhil)

