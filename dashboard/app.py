import streamlit as st
import requests
import os
from io import BytesIO
from PIL import Image
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import base64

# -----------------------------
# Config & Styling
# -----------------------------

API_URL = os.getenv("API_URL", "http://127.0.0.1:8000")

st.set_page_config(
    page_title="HealthAI Pro | Clinical X-ray Analysis",
    page_icon="🩻",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Premium Medical Glassmorphism Styling
st.markdown("""
<style>
    :root {
        --cyan: #00f0ff;
        --cyan-glow: rgba(0, 240, 255, 0.4);
        --purple: #a855f7;
        --purple-glow: rgba(168, 85, 247, 0.4);
    }
    
    /* App Background */
    .stApp {
        background: radial-gradient(circle at 40% 10%, #15243b 0%, #070d18 60%);
        font-family: 'Inter', sans-serif;
    }
    
    /* Top padding fix */
    .block-container {
        padding-top: 2rem !important;
    }
    
    /* Hide default header */
    header { visibility: hidden; }
    
    /* Sidebar styling */
    [data-testid="stSidebar"] {
        background-color: rgba(16, 25, 41, 0.6) !important;
        backdrop-filter: blur(16px);
        border-right: 1px solid rgba(43, 62, 88, 0.3);
    }
    
    /* Metric Cards */
    [data-testid="metric-container"] {
        background: rgba(16, 25, 41, 0.7) !important;
        backdrop-filter: blur(12px) !important;
        padding: 20px !important;
        border-radius: 12px !important;
        border: 1px solid rgba(43, 62, 88, 0.5) !important;
        box-shadow: 0 4px 20px rgba(0,0,0,0.3) !important;
        transition: transform 0.3s ease, border-color 0.3s ease;
    }
    [data-testid="metric-container"]:hover {
        transform: translateY(-5px);
        border-color: var(--cyan) !important;
        box-shadow: 0 8px 25px var(--cyan-glow) !important;
    }
    [data-testid="stMetricLabel"] {
        color: #8ba0b8 !important;
        font-weight: 500 !important;
        letter-spacing: 0.5px;
    }
    [data-testid="stMetricValue"] {
        color: #fff !important;
        font-weight: 300 !important;
        font-size: 2.2rem !important;
    }
    
    /* Action Buttons */
    .stButton>button {
        width: 100%;
        background: linear-gradient(135deg, var(--cyan), #3b82f6);
        color: #000;
        border: none;
        border-radius: 8px;
        height: 3.2em;
        font-weight: 700;
        letter-spacing: 1px;
        box-shadow: 0 0 15px rgba(0,240,255,0.3);
        transition: all 0.3s ease;
        text-transform: uppercase;
    }
    .stButton>button:hover {
        box-shadow: 0 0 25px rgba(0,240,255,0.6);
        transform: scale(1.02);
        color: #000;
        background: linear-gradient(135deg, #33f3ff, #60a5fa);
    }
    
    /* Secondary Download Button */
    .stDownloadButton>button {
        background: transparent !important;
        color: #fff !important;
        border: 1px solid rgba(255,255,255,0.2) !important;
        backdrop-filter: blur(4px) !important;
    }
    .stDownloadButton>button:hover {
        background: rgba(255,255,255,0.05) !important;
        border-color: #fff !important;
    }

    /* Headings */
    h1 {
        font-weight: 800 !important;
        letter-spacing: -1px;
        background: linear-gradient(90deg, #fff, #8ba0b8);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0 !important;
    }
    h2, h3 { 
        color: #58a6ff !important; 
        font-weight: 600 !important; 
    }
    
    /* Images */
    div[data-testid="stImage"] img {
        border-radius: 12px;
        box-shadow: 0 10px 30px rgba(0,0,0,0.5);
        border: 1px solid rgba(255,255,255,0.05);
    }
    
    /* Info Box */
    .stAlert {
        border-radius: 8px !important;
        background-color: rgba(16, 25, 41, 0.7) !important;
        backdrop-filter: blur(8px) !important;
        border: 1px solid rgba(43, 62, 88, 0.5) !important;
        color: #e2e8f0 !important;
    }
</style>
""", unsafe_allow_html=True)

# -----------------------------
# Sidebar - Patient Information
# -----------------------------

with st.sidebar:
    st.image("https://raw.githubusercontent.com/Duggineniakhil/HealthAI/main/healthai.png", use_column_width=True)
    st.title("🏥 Patient Context")
    
    patient_id = st.text_input("Patient ID", "PX-9921")
    patient_age = st.number_input("Age", 18, 100, 45)
    patient_gender = st.selectbox("Gender", ["Male", "Female", "Other"])
    
    st.divider()
    
    st.markdown("### 🛠 Analysis Settings")
    mode = st.radio(
        "Analysis Mode",
        ["Comprehensive (CheXpert)", "Pneumonia Screening"]
    )
    
    if mode == "Comprehensive (CheXpert)":
        threshold = st.slider("Detection Threshold", 0.1, 0.9, 0.3)
    else:
        threshold = 0.5

    st.divider()
    st.info("💡 Grad-CAM Heatmap generation is enabled for pathological localization.")

# -----------------------------
# Main Dashboard
# -----------------------------

st.title("🩻 HealthAI Pro - Diagnostic Support System")
st.caption(f"Connected to Clinical API: {API_URL}")

col1, col2 = st.columns([1, 1], gap="large")

with col1:
    st.subheader("📸 Image Acquisition")
    uploaded_file = st.file_uploader(
        "Upload DICOM or standard Chest X-ray (JPG/PNG)",
        type=["jpg", "jpeg", "png"]
    )
    
    if uploaded_file:
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption="Current Acquisition", use_container_width=True)
        
        analyze_btn = st.button("🚀 Run AI Diagnostic Suite")
    else:
        st.info("Please upload a chest X-ray to begin the automated analysis pipeline.")

with col2:
    st.subheader("📊 Diagnostic Results")
    
    if uploaded_file and analyze_btn:
        with st.spinner("Processing through Neural Network..."):
            try:
                files = {"file": (uploaded_file.name, uploaded_file.getvalue(), uploaded_file.type)}
                
                # Determine endpoint
                endpoint = "/predict-xray" if mode == "Pneumonia Screening" else "/predict-xray-multidisease"
                
                resp = requests.post(f"{API_URL}{endpoint}", files=files)
                
                if resp.status_code == 200:
                    data = resp.json()
                    heatmap_b64 = data.get("heatmap")
                    
                    if mode == "Pneumonia Screening":
                        prob = data.get("pneumonia_probability", 0.0)
                        label = data.get("predicted_label", "Unknown")
                        final_results = {"Pneumonia": prob}
                        
                        m_col1, m_col2 = st.columns(2)
                        m_col1.metric("Pneumonia Prob.", f"{prob*100:.1f}%")
                        m_col2.metric("Screening Result", label)
                        
                        # Gauge Chart
                        fig = go.Figure(go.Indicator(
                            mode = "gauge+number",
                            value = prob * 100,
                            gauge = {
                                'axis': {'range': [None, 100]},
                                'bar': {'color': "#ff4b4b" if prob > 0.5 else "#58a6ff"},
                                'steps': [
                                    {'range': [0, 50], 'color': "rgba(0, 255, 0, 0.1)"},
                                    {'range': [50, 100], 'color': "rgba(255, 0, 0, 0.1)"}
                                ],
                            }
                        ))
                        fig.update_layout(
                            height=300, 
                            margin=dict(l=20, r=20, t=30, b=20), 
                            paper_bgcolor='rgba(0,0,0,0)', 
                            font={'color': "#8ba0b8", 'family': "Inter"}
                        )
                        st.plotly_chart(fig, use_container_width=True)

                    else:
                        predictions = data.get("predictions", {})
                        final_results = predictions
                        df = pd.DataFrame([{"Condition": k, "Probability": v} for k, v in predictions.items()])
                        df = df.sort_values("Probability", ascending=True)
                        
                        # Horizontal Bar Chart
                        fig = px.bar(
                            df, x='Probability', y='Condition', 
                            orientation='h', color='Probability',
                            color_continuous_scale='RdBu_r'
                        )
                        fig.update_layout(
                            height=400, 
                            paper_bgcolor='rgba(0,0,0,0)', 
                            plot_bgcolor='rgba(0,0,0,0)', 
                            font={'color': "#8ba0b8", 'family': "Inter"},
                            margin=dict(l=0, r=0, t=20, b=0),
                            coloraxis_showscale=False
                        )
                        st.plotly_chart(fig, use_container_width=True)
                        
                        flagged = df[df["Probability"] >= threshold]
                        if not flagged.empty:
                            st.error(f"⚠️ {len(flagged)} pathologies detected for Patient {patient_id}")
                        else:
                            st.success("✅ No pathologies detected above threshold.")

                    # Show Heatmap if available
                    if heatmap_b64:
                        st.divider()
                        st.subheader("🧠 Pathological Localization (Grad-CAM)")
                        st.image(base64.b64decode(heatmap_b64), caption=f"Heatmap visualization for Patient {patient_id}", use_container_width=True)
                        st.info("The highlighted areas indicate the regions contributing most to the AI prediction.")

                    # Export Section
                    st.divider()
                    st.subheader("📑 Report Management")
                    
                    report_data = {
                        "patient": {"id": patient_id, "age": patient_age, "gender": patient_gender},
                        "results": final_results
                    }
                    
                    report_resp = requests.post(f"{API_URL}/generate-report", json=report_data)
                    if report_resp.status_code == 200:
                        st.download_button(
                            label="📄 Download Diagnostic PDF Report",
                            data=report_resp.content,
                            file_name=f"HealthAI_Report_{patient_id}.pdf",
                            mime="application/pdf"
                        )
                    else:
                        st.warning("Could not generate PDF report at this time.")

                else:
                    st.error(f"Clinical API Error: {resp.text}")
                    
            except Exception as e:
                st.error("Diagnostic Pipeline Interrupted")
                st.exception(e)

    elif not uploaded_file:
        st.write("Waiting for data acquisition...")

st.divider()
st.caption("© 2024 HealthAI Diagnostic Systems | Developed by Duggineni Akhil | For Clinical Decision Support Only")
