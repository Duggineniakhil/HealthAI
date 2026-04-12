
---

# 🫁 **HealthAI Pro – Clinical Neural Diagnostic System**

*A premium, state-of-the-art AI medical dashboard designed for automated chest X-ray pathology detection, clinical patient management, and automated report archiving.*

![HealthAI Banner](https://raw.githubusercontent.com/Duggineniakhil/HealthAI/main/healthai.png)

---

## 🏷️ Tech Stack

<div align="center">
  
![Python](https://img.shields.io/badge/Python-3.10-blue?logo=python&logoColor=white)
![TensorFlow](https://img.shields.io/badge/TensorFlow-Deep%20Learning-orange?logo=tensorflow&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-Backend-009688?logo=fastapi&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-Glassmorphism-FF4B4B?logo=streamlit&logoColor=white)
![Plotly](https://img.shields.io/badge/Plotly-Clinical%20Visuals-3F4F75?logo=plotly)
![FPDF](https://img.shields.io/badge/FPDF-Report%20Generation-red)

</div>

---

## 🚀 **System Overview**

**HealthAI Pro** is a comprehensive clinical workstation that bridges the gap between deep learning and medical practice. It features a high-end **Medical Glassmorphism UI** that provides doctors with real-time neural insights, Grad-CAM visualization, and persistent patient tracking.

### ✨ Master Features
* 🌌 **Premium Visual Architecture**: A 3-column clinical layout built with advanced CSS glassmorphism, dark mode aesthetics, and Inter typography.
* 🧬 **Neural Saliency (Grad-CAM)**: Real-time heatmap generation highlighting precisely where the AI "looks" to detect pathologies.
* 📂 **Clinical Persistence Engine**:
    * **Automated ID Generation**: Unique clinical identifiers assigned to every patient session.
    * **Patient Database**: Persistent JSON-backed registration system with full lifecycle management (View/Register/Delete).
    * **Report Archives**: Automated PDF generation and server-side archiving for future retrieval.
* 🩻 **Multi-Disease Analysis**: Detects Atelectasis, Cardiomegaly, Consolidation, Edema, Effusion, and more with high-confidence probability spline charts.
* ⏱️ **Live Clinical Context**: Dynamic scan details including real-time timestamps, diagnostic status tracking, and engine metrics.

---

## 📁 **Project Architecture**

```text
HealthAI/
├── backend/            # FastAPI Production Server
│   ├── main.py         # AI Inference & Grad-CAM Pipeline
│   └── requirements.txt
├── dashboard/          # Modernized Streamlit UI
│   ├── app.py          # Dashboard Engine (Glassmorphic)
│   ├── data/           # Persistent JSON Patient Database
│   ├── reports/        # Global Clinical PDF Archive
│   └── requirements.txt
├── models/             # Clinical Neural Weights
│   ├── xray_chexpert_labels.json
│   └── xray_class_mapping.json
├── notebooks/          # R&D / Model Training
├── screenshots/        # Visual documentation
├── healthai.png        # Brand assets
└── README.md
```

---

## 🛠️ **Clinical Deployment**

### 1️⃣ Repository Initialization
```bash
git clone https://github.com/Duggineniakhil/HealthAI.git
cd HealthAI
```

### 2️⃣ Start AI Inference Engine (Backend)
```bash
cd backend
python -m venv venv
.\venv\Scripts\activate  # Windows
pip install -r requirements.txt
python main.py
```
> [!NOTE]
> Ensure `.h5` model files are placed in the `models/` directory.

### 3️⃣ Launch Clinical Workstation (Frontend)
```bash
cd dashboard
pip install -r requirements.txt
streamlit run app.py
```

---

## 📊 **Clinical Workflow**

1. **Registration**: Start the session. The system auto-generates a unique `PX-ID`.
2. **Analysis**: Drag & drop X-ray images. The AI runs a multi-disease pipeline with Grad-CAM overlays.
3. **Review**: Use the integrated **"Review Scan"** and **"Validate"** buttons to track diagnosis status.
4. **Archive**: Generate a PDF report. The system automatically archives a copy in the **Reports Grid** for long-term storage.
5. **Database Management**: Access the **PATIENTS** tab to view historical registration logs and manage trial data.

---

## 📸 **Live Previews**

### Ultra-Modern Dashboard
| Clinical Layout | Results & Distribution |
| :---: | :---: |
| ![Dashboard](https://raw.githubusercontent.com/Duggineniakhil/HealthAI/main/screenshots/dashboard_ha.jpg) | ![Results](https://raw.githubusercontent.com/Duggineniakhil/HealthAI/main/screenshots/multi_1.jpg) |

---

## 👤 **Author & Development**

**Duggineni Akhil**  
*Lead Developer | Medical AI Researcher*

📧 [duggineniakhil15@gmail.com](mailto:duggineniakhil15@gmail.com)  
🔗 [LinkedIn](https://linkedin.com/in/akhil-duggineni) | [GitHub](https://github.com/Duggineniakhil)

---

## 📄 **License**

Released under the **MIT License**. © 2024 HealthAI Systems.
