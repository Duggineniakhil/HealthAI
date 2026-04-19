# HealthAI – Clinical Neural Diagnostic System

**HealthAI** is a full-stack medical analysis workstation that bridges advanced Deep Learning with a state-of-the-art clinical user interface. It is designed for high-sensitivity screening of chest X-ray pathologies, featuring a robust backend for neural inference and a high-performance frontend for clinical patient management.


## The Full-Stack Architecture

HealthAI is built as a modular, three-tier ecosystem:

### 1. Machine Learning Engine (The Core)
*   **Architectures**: 
    *   **DenseNet121**: Multi-disease classification (8 conditions) trained on the **CheXpert** dataset.
    *   **MobileNetV2**: High-speed binary screening (Normal vs Pneumonia).
*   **Neural Visualization**: Gradient-weighted Class Activation Mapping (**Grad-CAM**) built natively in TensorFlow for real-time heatmap saliency.
*   **Clinical Calibration**: Optimized for **High Recall** (8% threshold) to ensure subtle markers (like early-stage Pneumothorax) are captured during screening.
*   **Fallback Intelligence**: Integrated with the **torchxrayvision** library to provide secondary diagnostic validation.

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

## Getting Started

### 1. Launch AI Backend
```bash
cd backend
pip install -r requirements.txt
python main.py
```

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

