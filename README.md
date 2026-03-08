
---

# 🫁 **HealthAI – AI-Powered Chest X-Ray Disease Detection System**

*A deep-learning powered platform for multi-disease medical imaging analysis, built with TensorFlow, FastAPI, Streamlit, and optimized for performance.*

![HealthAI Banner](https://raw.githubusercontent.com/Duggineniakhil/HealthAI/main/healthai.png)

---

## 🏷️ Tech Stack

<div align="center">
  
![Python](https://img.shields.io/badge/Python-3.10-blue?logo=python&logoColor=white)
![TensorFlow](https://img.shields.io/badge/TensorFlow-Deep%20Learning-orange?logo=tensorflow&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-Backend-009688?logo=fastapi&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-Dashboard-FF4B4B?logo=streamlit&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-Data%20Processing-150458?logo=pandas)
![NumPy](https://img.shields.io/badge/NumPy-Scientific%20Computing-013243?logo=numpy&logoColor=white)
![EfficientNet](https://img.shields.io/badge/EfficientNet-Transfer%20Learning-brightgreen)

</div>

---

## 🚀 **Overview**

**HealthAI** is a comprehensive medical imaging platform designed to assist healthcare professionals in diagnosing chest pathologies from X-ray images. Utilizing a state-of-the-art **EfficientNetB0** model trained on the **CheXpert dataset**, it provides real-time, multi-label classification for various lung conditions.

### ✨ Key Features
* 📸 **Multi-Label Classification**: Simultaneously detects pathologies like Atelectasis, Cardiomegaly, Consolidation, Edema, and Pleural Effusion.
* 🧠 **State-of-the-Art Model**: Leverages Transfer Learning with EfficientNetB0 for high accuracy and robust feature extraction.
* ⚙️ **Robust API**: Built with FastAPI, providing a scalable and high-performance backend for AI inference.
* 🧩 **Interactive Dashboard**: A user-friendly Streamlit interface for seamless image upload and result visualization.
* 🔒 **Modular Architecture**: Clean separation of concerns between the data pipeline, model serving, and frontend.

---

## 📁 **Project Structure**

```text
HealthAI/
├── backend/            # FastAPI source code
│   ├── main.py         # Primary API entry point
│   ├── requirements.txt # Backend dependencies
├── dashboard/          # Streamlit UI source code
│   ├── app.py          # Dashboard logic
│   ├── requirements.txt # UI dependencies
├── models/             # Model configuration & labels (Weights excluded from Git)
│   ├── xray_chexpert_labels.json
│   └── xray_class_mapping.json
├── notebooks/          # Training & Experimentation
│   ├── 01_xray_disease_detection.ipynb
│   └── 02_xray_chexpert_multidisease.ipynb
├── screenshots/        # Project visuals
├── .gitignore          # Optimized for ML projects
└── README.md
```

---

## 🛠️ **Installation & Local Setup**

### 1️⃣ Clone the Repository
```bash
git clone https://github.com/Duggineniakhil/HealthAI.git
cd HealthAI
```

### 2️⃣ Setup Backend
It is recommended to use a virtual environment.
```bash
cd backend
python -m venv venv
# Windows:
.\venv\Scripts\activate
# Linux/Mac:
source venv/bin/activate

pip install -r requirements.txt
```

> [!IMPORTANT]
> **Models are not included in the repository** due to size constraints. Please ensure you place your `.h5` model files in the `models/` directory before running.

### 3️⃣ Run the Backend
```bash
python main.py
```
The API will be available at `http://127.0.0.1:8000`. You can explore the documentation at `/docs`.

### 4️⃣ Setup & Run Dashboard
Open a new terminal:
```bash
cd dashboard
pip install -r requirements.txt
streamlit run app.py
```
The UI will be available at `http://localhost:8501`.

---

## 📡 **API Reference**

### `GET /health`
Check if the API service is alive.

### `POST /predict-xray`
Simple binary classification for Pneumonia vs Normal.
- **Payload**: `file` (Multipart/form-data)

### `POST /predict-xray-multidisease`
Detailed analysis for multiple pathologies.
- **Payload**: `file` (Multipart/form-data)

---

## 📸 **Screenshots**

### Dashboard Interface
![Dashboard Home](https://raw.githubusercontent.com/Duggineniakhil/HealthAI/main/screenshots/dashboard_ha.jpg)

### Analysis Results
| Pneumonia Prediction | Multi-Disease Prediction |
| :---: | :---: |
| ![Pneumonia](https://raw.githubusercontent.com/Duggineniakhil/HealthAI/main/screenshots/pneumonia_ha.jpg) | ![Multi](https://raw.githubusercontent.com/Duggineniakhil/HealthAI/main/screenshots/multi_1.jpg) |

---

## 📊 **Model Metrics**

* **Architecture**: EfficientNetB0 (ImageNet weights)
* **Optimization**: Adam (1e-4) with Binary Crossentropy
* **Accuracy**: ~88% Validation Accuracy
* **Validation Loss**: ~0.27
* **Inference Speed**: < 200ms on CPU

---

## 👤 **Author**

**Duggineni Akhil**  
*B.Tech Computer Science*

📧 [duggineniakhil15@gmail.com](mailto:duggineniakhil15@gmail.com)  
🔗 [LinkedIn](https://linkedin.com/in/akhil-duggineni) | [GitHub](https://github.com/Duggineniakhil)

---

## 📄 **License**

Distributed under the **MIT License**. See `LICENSE` for more information.
