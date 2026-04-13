import streamlit as st
import requests
import os
from io import BytesIO
from PIL import Image
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import base64
import json
import uuid
import time
from datetime import datetime

# -----------------------------
# Data Persistence Utilities
# -----------------------------

DATA_DIR = "dashboard/data"
REPORTS_DIR = "dashboard/reports"
PATIENTS_FILE = os.path.join(DATA_DIR, "patients.json")

os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(REPORTS_DIR, exist_ok=True)

if not os.path.exists(PATIENTS_FILE):
    with open(PATIENTS_FILE, "w") as f:
        json.dump([], f)

def load_patients():
    try:
        with open(PATIENTS_FILE, "r") as f:
            return json.load(f)
    except:
        return []

def save_patients(patients):
    with open(PATIENTS_FILE, "w") as f:
        json.dump(patients, f, indent=4)

def generate_unique_id():
    return f"PX-{uuid.uuid4().hex[:4].upper()}"

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
# State Management & Routing
# -----------------------------

if 'patient' not in st.session_state:
    st.markdown("""
    <div style='display:flex; justify-content:center; margin-top:50px; margin-bottom:20px;'>
        <div style='background:rgba(16,25,41,0.8); padding:40px; border-radius:16px; border:1px solid rgba(0, 240, 255, 0.3); box-shadow:0 10px 40px rgba(0, 240, 255, 0.1); width:100%; max-width:600px; text-align:center;'>
            <svg viewBox="0 0 24 24" width="48" height="48" fill="none" stroke="var(--cyan)" stroke-width="2"><path d="M22 12h-4l-3 9L9 3l-3 9H2"></path></svg>
            <h2 style='color:#fff; margin-top:15px; margin-bottom:5px;'>Patient Registration</h2>
            <p style='color:#8ba0b8; margin-bottom:10px;'>Establish clinical context before proceeding to pipeline.</p>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    _, col_form, _ = st.columns([1, 1.5, 1])
    with col_form:
        # Initialize an ID if not present in session state for this form instance
        if 'form_patient_id' not in st.session_state:
            st.session_state['form_patient_id'] = generate_unique_id()

        with st.form("patient_form"):
            st.markdown("<strong style='color:var(--cyan);'>Primary Demographics</strong>", unsafe_allow_html=True)
            p_name = st.text_input("Full Name", placeholder="e.g. John Doe")
            
            c1, c2 = st.columns(2)
            # Display the auto-generated ID as a disabled text input
            p_id = c1.text_input("Patient ID", value=st.session_state['form_patient_id'], disabled=True)
            p_age = c2.number_input("Age", min_value=1, max_value=120, value=25)
            
            st.markdown("<br><strong style='color:var(--cyan);'>Clinical Context</strong>", unsafe_allow_html=True)
            c3, c4 = st.columns(2)
            p_gender = c3.selectbox("Gender", ["Male", "Female", "Other"])
            p_status = c4.selectbox("Status", ["Routine", "Urgent", "Critical"])
            
            st.markdown("<br>", unsafe_allow_html=True)
            submitted = st.form_submit_button("INITIALIZE DIAGNOSTIC SESSION")
            
            if submitted:
                if not p_name.strip():
                    st.error("Please enter the patient's name.")
                else:
                    new_patient = {
                        "name": p_name,
                        "id": p_id,
                        "age": p_age,
                        "gender": p_gender,
                        "status": p_status,
                        "registered_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    }
                    # Persist to database
                    db = load_patients()
                    db.append(new_patient)
                    save_patients(db)
                    
                    # Set session state and clear form ID for next time
                    st.session_state['patient'] = new_patient
                    del st.session_state['form_patient_id']
                    st.rerun()

else:
    patient = st.session_state['patient']
    if 'scan_status' not in st.session_state:
        st.session_state['scan_status'] = 'Pending'
    
    # Render Interactive Navigation using Streamlit native components customized by our CSS
    st.markdown("""
    <div style='display:flex; justify-content:space-between; align-items:center; padding:15px 30px; background:rgba(16,25,41,0.6); backdrop-filter:blur(10px); border-bottom:1px solid rgba(43,62,88,0.3); border-radius:12px;'>
        <div style='display:flex; align-items:center; gap:10px;'>
            <svg viewBox="0 0 24 24" width="28" height="28" fill="none" stroke="var(--cyan)" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M22 12h-4l-3 9L9 3l-3 9H2"></path></svg>
            <h3 style='margin:0; color:#fff; font-weight:700; font-size:1.4rem;'>MedAI Analysis</h3>
        </div>
        <div style='display:flex; align-items:center; gap:12px; color:#fff; font-weight:500;'>
            <img src='https://i.pravatar.cc/100?img=11' width='36' height='36' style='border-radius:50%; border:2px solid rgba(255,255,255,0.1);'>
            <span>Dr. Alex Chen</span>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # The actual functional navigation
    nav_choice = st.radio("App Route", ["PATIENTS", "ANALYSIS", "REPORTS", "SETTINGS"], horizontal=True, label_visibility="collapsed")
    st.markdown("<br>", unsafe_allow_html=True)

    if nav_choice == "PATIENTS":
        st.subheader("🏥 Patient Database")
        db = load_patients()
        
        if not db:
            st.info("No patients registered yet. Please register a patient in the Analysis tab.")
        else:
            # Custom styled table header
            st.markdown("""
            <div style='display:grid; grid-template-columns: 1fr 2fr 1fr 1fr 1fr 0.5fr; background:rgba(255,255,255,0.05); padding:10px; border-radius:8px; font-weight:bold; color:var(--cyan); margin-bottom:10px;'>
                <div>ID</div><div>NAME</div><div>AGE</div><div>GENDER</div><div>STATUS</div><div>ACTION</div>
            </div>
            """, unsafe_allow_html=True)
            
            for i, p in enumerate(db):
                cols = st.columns([1, 2, 1, 1, 1, 0.5])
                cols[0].write(p["id"])
                cols[1].write(p["name"])
                cols[2].write(p["age"])
                cols[3].write(p["gender"])
                status_color = "#ff4b4b" if p["status"] == "Critical" else ("#f59e0b" if p["status"] == "Urgent" else "#10b981")
                cols[4].markdown(f"<span style='color:{status_color};'>{p['status']}</span>", unsafe_allow_html=True)
                
                if cols[5].button("🗑️", key=f"del_p_{p['id']}_{i}"):
                    db.pop(i)
                    save_patients(db)
                    st.rerun()
            
        if st.button("REGISTER NEW PATIENT"):
            if 'patient' in st.session_state:
                del st.session_state['patient']
            st.rerun()
        
    elif nav_choice == "REPORTS":
        st.subheader("📑 Report Archives")
        report_files = [f for f in os.listdir(REPORTS_DIR) if f.endswith(".pdf")]
        
        if not report_files:
            st.info("No diagnostic reports archived yet.")
        else:
            for i, f_name in enumerate(report_files):
                cols = st.columns([4, 1, 0.5])
                cols[0].write(f"📄 {f_name}")
                
                file_path = os.path.join(REPORTS_DIR, f_name)
                with open(file_path, "rb") as f:
                    cols[1].download_button("Download", f, file_name=f_name, key=f"dl_{i}")
                
                if cols[2].button("🗑️", key=f"del_r_{i}"):
                    os.remove(file_path)
                    st.rerun()
        
    elif nav_choice == "SETTINGS":
        st.subheader("⚙️ Analysis Settings")
        st.slider("Baseline AI Confidence Threshold", 0.0, 1.0, 0.5)
        st.checkbox("Enable Background Telemetry Logging", value=True)
        st.checkbox("Force Grad-CAM High Resolution", value=False)
        st.button("SAVE SYSTEM PARAMETERS")
        
    elif nav_choice == "ANALYSIS":

        # 3-Column Layout
        col_left, col_mid, col_right = st.columns([1, 1.8, 1], gap="large")

        with col_left:
            # Patient Context Dynamic injection
            status_color = "#ff4b4b" if patient["status"] == "Critical" else ("#f59e0b" if patient["status"] == "Urgent" else "#10b981")
            
            st.markdown(f"""
            <div style='background:rgba(16,25,41,0.7); padding:24px; border-radius:16px; border:1px solid rgba(43,62,88,0.5); margin-bottom:24px; box-shadow:0 10px 30px rgba(0,0,0,0.3);'>
                <div style='display:flex; align-items:center; gap:16px; margin-bottom:24px;'>
                    <img src='https://i.pravatar.cc/100?img=5' width='64' height='64' style='border-radius:12px;'>
                    <div>
                        <h3 style='margin:0; color:#fff; font-weight:700; font-size:1.2rem; text-transform:uppercase;'>{patient["name"]}</h3>
                        <span style='color:#8ba0b8; font-size:13px;'>ID: {patient["id"]}</span>
                    </div>
                </div>
                <div style='display:flex; justify-content:space-between; margin-bottom:24px;'>
                    <div>
                        <span style='color:#8ba0b8; font-size:13px;'>Age</span><br>
                        <h2 style='margin:0; color:var(--cyan); font-weight:400;'>{patient["age"]}</h2>
                    </div>
                    <div>
                        <span style='color:#8ba0b8; font-size:13px;'>Status</span><br>
                        <h2 style='margin:0; color:{status_color}; font-weight:400;'>{patient["status"]}</h2>
                    </div>
                </div>
                <div style='border-top:1px solid rgba(255,255,255,0.1); padding-top:20px; color:#8ba0b8; font-size:13px; line-height:1.6;'>
                    <strong style='color:#fff; font-size:11px; letter-spacing:1px;'>ANALYSIS SUMMARY</strong><br><br>
                    Chest X-ray designed for advanced medical image analysis. Clinical triage mapped to {patient["status"]} priority tier.
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            # Placeholder for Confidence Results
            confidence_placeholder = st.empty()
            confidence_placeholder.markdown("""
            <div style='background:rgba(16,25,41,0.7); padding:24px; border-radius:16px; border:1px solid rgba(43,62,88,0.5); box-shadow:0 10px 30px rgba(0,0,0,0.3);'>
                <strong style='color:#fff; font-size:12px; letter-spacing:1px; display:block; margin-bottom:20px;'>DISEASE CONFIDENCE</strong>
                <p style='color:#8ba0b8; font-size:13px;'>Awaiting X-Ray scan...</p>
            </div>
            """, unsafe_allow_html=True)
    
        with col_mid:
            # Image Acquisition
            st.markdown("<strong style='color:#fff; font-size:12px; letter-spacing:1px; display:block; margin-bottom:10px;'>CHEST X-RAY IMAGES</strong>", unsafe_allow_html=True)
            
            uploaded_file = st.file_uploader("Upload Image", type=["jpg", "jpeg", "png"], label_visibility="collapsed")
            image_placeholder = st.empty()
            chart_placeholder = st.empty()
    
        with col_right:
            # Findings and Details panel
            findings_placeholder = st.empty()
            findings_placeholder.markdown("""
            <div style='background:rgba(16,25,41,0.7); padding:24px; border-radius:16px; border:1px solid rgba(43,62,88,0.5); height:400px; margin-bottom:24px; box-shadow:0 10px 30px rgba(0,0,0,0.3);'>
                <strong style='color:#fff; font-size:12px; letter-spacing:1px; display:block; margin-bottom:20px;'>FINDINGS & NOTES</strong>
                <p style='color:#8ba0b8; font-size:14px;'>Upload an X-ray to generate neural insights.</p>
            </div>
            """, unsafe_allow_html=True)
    
            st.markdown(f"""
            <div style='background:rgba(16,25,41,0.7); padding:24px; border-radius:16px; border:1px solid rgba(43,62,88,0.5); margin-bottom:24px; box-shadow:0 10px 30px rgba(0,0,0,0.3);'>
                <strong style='color:#fff; font-size:12px; letter-spacing:1px; display:block; margin-bottom:15px;'>SCAN DETAILS</strong>
                <div style='display:flex; justify-content:space-between; margin-bottom:10px;'><span style='color:#8ba0b8; font-size:13px;'>Patient ID</span><span style='color:#fff; font-size:13px;'>{patient["id"]}</span></div>
                <div style='display:flex; justify-content:space-between; margin-bottom:10px;'><span style='color:#8ba0b8; font-size:13px;'>Scan Date</span><span style='color:#fff; font-size:13px;'>{datetime.now().strftime("%Y.%m.%d")}</span></div>
                <div style='display:flex; justify-content:space-between; margin-bottom:10px;'><span style='color:#8ba0b8; font-size:13px;'>Engine</span><span style='color:#fff; font-size:13px;'>CheXpert-ResNet-v4</span></div>
                <div style='display:flex; justify-content:space-between;'><span style='color:#8ba0b8; font-size:13px;'>Status</span><span style='color:{status_color}; font-size:13px;'>{st.session_state['scan_status']}</span></div>
            </div>
            """, unsafe_allow_html=True)
            
            c_b1, c_b2 = st.columns(2)
            with c_b1:
                if st.button("REVIEW SCAN", use_container_width=True):
                    st.session_state['scan_status'] = 'Under Review'
                    st.rerun()
            with c_b2:
                if st.button("VALIDATE", use_container_width=True):
                    st.session_state['scan_status'] = 'Validated ✅'
                    st.balloons()
                    st.rerun()
            if st.button("RESET OVERVIEW", use_container_width=True):
                st.session_state.clear()
                st.rerun()
    
        # -----------------------------
        # Trigger Logic
        # -----------------------------
        if uploaded_file:
            image = Image.open(uploaded_file).convert("RGB")
            
            with st.spinner("Processing through Neural Network..."):
                try:
                    files = {"file": (uploaded_file.name, uploaded_file.getvalue(), uploaded_file.type)}
                    resp = requests.post(f"{API_URL}/predict-xray-multidisease", files=files)
                    
                    if resp.status_code == 200:
                        data = resp.json()
                        heatmap_b64 = data.get("heatmap")
                        predictions = data.get("predictions", {})
                        
                        # Image/Heatmap
                        if heatmap_b64:
                            image_placeholder.image(base64.b64decode(heatmap_b64), width="stretch")
                        else:
                            image_placeholder.image(image, width="stretch")
                        
                        # Dynamic Confidence HTML Generation
                        is_normal = data.get("is_normal", False)
                        no_finding_prob = data.get("no_finding_prob", 0.0)

                        # Filter out 'No Finding' from disease bars
                        disease_preds = {k: v for k, v in predictions.items() if k.lower() != "no finding"}
                        sorted_preds = sorted(disease_preds.items(), key=lambda x: x[1], reverse=True)
                        highest_dis = sorted_preds[0] if sorted_preds else ("Unknown", 0)

                        # Determine overall scan verdict
                        top_score = highest_dis[1]
                        if is_normal or top_score < 0.10:
                            verdict_color = "#10b981"
                            verdict_icon = "✅"
                            verdict_text = "NO SIGNIFICANT PATHOLOGY DETECTED"
                        elif top_score >= 0.5:
                            verdict_color = "#ef4444"
                            verdict_icon = "🔴"
                            verdict_text = f"CRITICAL: {highest_dis[0].upper()} DETECTED"
                        elif top_score >= 0.25:
                            verdict_color = "#f59e0b"
                            verdict_icon = "🟡"
                            verdict_text = f"SUSPECTED: {highest_dis[0].upper()}"
                        else:
                            verdict_color = "#00f0ff"
                            verdict_icon = "🔵"
                            verdict_text = f"MILD INDICATOR: {highest_dis[0].upper()}"

                        conf_rows = ""
                        colors = ['#00f0ff', '#a855f7', '#ef4444', '#f59e0b', '#3b82f6', '#10b981', '#6366f1']
                        for idx, (disease, prob) in enumerate(sorted_preds[:6]):
                            percentage = prob * 100
                            color = colors[idx % len(colors)]
                            level = "HIGH" if prob > 0.5 else "MED" if prob > 0.2 else "LOW"
                            level_color = "#ef4444" if level == "HIGH" else ("#f59e0b" if level == "MED" else "#8ba0b8")
                            conf_rows += f"""
<div style="margin-bottom:14px;">
<div style="display:flex;justify-content:space-between;margin-bottom:6px;font-size:12px;font-weight:600;">
<span style="color:#e2e8f0;">{disease.upper()}</span>
<span style="color:{color};">{percentage:.1f}% <span style="color:{level_color};font-size:10px;font-weight:700;background:rgba(255,255,255,0.07);padding:1px 5px;border-radius:3px;">{level}</span></span>
</div>
<div style="width:100%;background:rgba(255,255,255,0.07);height:7px;border-radius:4px;overflow:hidden;">
<div style="width:{min(percentage, 100):.1f}%;background:linear-gradient(90deg,{color},{color}88);height:100%;border-radius:4px;box-shadow:0 0 8px {color}88;"></div>
</div>
</div>"""

                        conf_html = f"""
<div style="background:rgba(16,25,41,0.85);padding:20px;border-radius:16px;border:1px solid rgba(43,62,88,0.6);box-shadow:0 10px 30px rgba(0,0,0,0.35);">
<div style="background:{verdict_color}22;border:1px solid {verdict_color}55;border-radius:10px;padding:12px 16px;margin-bottom:18px;">
<span style="font-size:18px;">{verdict_icon}</span>
<span style="color:{verdict_color};font-size:13px;font-weight:800;letter-spacing:0.5px;margin-left:8px;">{verdict_text}</span>
</div>
<strong style="color:#8ba0b8;font-size:10px;letter-spacing:1.5px;display:block;margin-bottom:14px;">DISEASE PROBABILITY (18 CONDITIONS)</strong>
{conf_rows}
<div style="border-top:1px solid rgba(255,255,255,0.07);padding-top:10px;margin-top:4px;">
<span style="color:#8ba0b8;font-size:10px;">Model: torchxrayvision DenseNet121 (CheXpert+NIH+MIMIC) &nbsp;|&nbsp; Direct sigmoid probabilities</span>
</div>
</div>"""
                        confidence_placeholder.markdown(conf_html, unsafe_allow_html=True)
                        
                        
                        # Compute logic colors for scan status
                        status_text_color = "#10b981" if "Validated" in st.session_state['scan_status'] else ("#f59e0b" if "Review" in st.session_state['scan_status'] else "#fff")

                        # Build severity info
                        if is_normal or top_score < 0.15:
                            primary_finding_html = (
                                "<span style='color:#10b981;font-size:20px;font-weight:800;'>&#9989; NORMAL SCAN</span>"
                                f"<br><span style='color:#8ba0b8;font-size:13px;'>No significant pathology detected. "
                                f"Normal confidence: {no_finding_prob*100:.0f}%</span>"
                            )
                            ai_rec = "<span style='color:#10b981;'>No severe anomalies. Routine follow-up recommended.</span>"
                        else:
                            sev = "Critical" if top_score >= 0.5 else "Moderate" if top_score >= 0.25 else "Possible"
                            sev_color = "#ef4444" if sev == "Critical" else ("#f59e0b" if sev == "Moderate" else "#00f0ff")
                            primary_finding_html = (
                                f"<span style='color:{sev_color};font-size:20px;font-weight:800;'>{highest_dis[0].upper()}</span>"
                                f"<br><span style='color:#8ba0b8;font-size:13px;'>Severity: <strong style='color:{sev_color};'>{sev}</strong>"
                                f" &nbsp;|&nbsp; Score: <strong style='color:#fff;'>{top_score*100:.1f}%</strong></span>"
                            )
                            if top_score >= 0.5:
                                ai_rec = "<span style='color:#ef4444;'>&#9888; Critical anomaly. Immediate clinical review required.</span>"
                            else:
                                ai_rec = "<span style='color:#f59e0b;'>Possible finding detected. Further evaluation advised.</span>"

                        secondary_html = ""
                        for d, p in sorted_preds[1:4]:
                            if p >= 0.08:
                                secondary_html += (
                                    f"<span style='display:inline-block;background:rgba(255,255,255,0.07);"
                                    f"border-radius:5px;padding:3px 8px;margin:3px 3px 3px 0;"
                                    f"color:#8ba0b8;font-size:11px;'>{d} ({p*100:.0f}%)</span>"
                                )
                        if not secondary_html:
                            secondary_html = "<span style='color:#8ba0b8;font-size:13px;'>None above threshold</span>"

                        # Findings panel - built as concatenated string to avoid Streamlit code-block bug
                        findings_html = (
                            "<div style='background:rgba(16,25,41,0.85);padding:24px;border-radius:16px;"
                            "border:1px solid rgba(43,62,88,0.5);min-height:400px;margin-bottom:24px;"
                            "overflow-y:auto;box-shadow:0 10px 30px rgba(0,0,0,0.3);'>"
                            "<strong style='color:#fff;font-size:11px;letter-spacing:1.5px;"
                            "display:block;margin-bottom:18px;'>FINDINGS &amp; NOTES</strong>"
                            "<div style='margin-bottom:16px;border-bottom:1px solid rgba(255,255,255,0.07);padding-bottom:16px;'>"
                            "<span style='color:#8ba0b8;font-size:10px;font-weight:700;letter-spacing:1px;'>PRIMARY DIAGNOSIS</span>"
                            "<br><br>" + primary_finding_html + "</div>"
                            "<div style='margin-bottom:16px;border-bottom:1px solid rgba(255,255,255,0.07);padding-bottom:16px;'>"
                            "<span style='color:#8ba0b8;font-size:10px;font-weight:700;letter-spacing:1px;'>SECONDARY INDICATORS</span>"
                            "<br><br>" + secondary_html + "</div>"
                            f"<div style='margin-bottom:16px;border-bottom:1px solid rgba(255,255,255,0.07);padding-bottom:16px;'>"
                            f"<span style='color:#8ba0b8;font-size:10px;font-weight:700;letter-spacing:1px;'>DR. REVIEW STATUS</span>"
                            f"<br><br><span style='color:{status_text_color};font-size:14px;font-weight:700;'>"
                            f"{st.session_state['scan_status']}</span></div>"
                            "<div><span style='color:#8ba0b8;font-size:10px;font-weight:700;letter-spacing:1px;'>AI RECOMMENDATION</span>"
                            "<br><br>" + ai_rec + "</div></div>"
                        )
                        findings_placeholder.markdown(findings_html, unsafe_allow_html=True)

                        # Line Chart (Spline / Area) - uses only disease predictions, not 'No Finding'
                        df = pd.DataFrame([
                            {"Condition": k, "Probability": v * 100}
                            for k, v in sorted_preds[:8]
                        ])
                        fig = px.bar(
                            df, x='Condition', y='Probability',
                            color='Probability',
                            color_continuous_scale=['#1e3a5f', '#00f0ff'],
                            text='Probability'
                        )
                        fig.update_traces(texttemplate='%{text:.1f}%', textposition='outside')
                        fig.update_layout(
                            height=260,
                            paper_bgcolor='rgba(0,0,0,0)',
                            plot_bgcolor='rgba(0,0,0,0)',
                            font={'color': "#8ba0b8", 'family': "Inter"},
                            margin=dict(l=0, r=0, t=30, b=0),
                            yaxis=dict(gridcolor="rgba(255,255,255,0.05)", zerolinecolor="rgba(255,255,255,0.05)", range=[0, 110]),
                            xaxis=dict(gridcolor="rgba(255,255,255,0.0)", tickangle=-20),
                            coloraxis_showscale=False,
                            showlegend=False
                        )
                        with chart_placeholder:
                            st.markdown("<strong style='color:#fff;font-size:11px;letter-spacing:1px;display:block;margin:20px 0 5px 0;'>DISEASE PROBABILITY DISTRIBUTION (CALIBRATED)</strong>", unsafe_allow_html=True)
                            st.plotly_chart(fig, use_container_width=True)
    
                        # Export Section (Using Patient Variables securely)
                        report_data = {
                            "patient": {"id": patient["id"], "age": patient["age"], "gender": patient["gender"]},
                            "results": predictions
                        }
                        report_resp = requests.post(f"{API_URL}/generate-report", json=report_data)
                        if report_resp.status_code == 200:
                            # Save copy to local server reports directory
                            report_filename = f"HealthAI_Report_{patient['id']}_{int(time.time())}.pdf"
                            report_path = os.path.join(REPORTS_DIR, report_filename)
                            with open(report_path, "wb") as rf:
                                rf.write(report_resp.content)
                            
                            st.download_button(
                                label="📄 Download Extracted PDF Report",
                                data=report_resp.content,
                                file_name=report_filename,
                                mime="application/pdf"
                            )
                            st.success(f"Report archived as {report_filename}")
    
                    else:
                        image_placeholder.error(f"API Error: {resp.text}")
                        
                except Exception as e:
                    image_placeholder.error("Diagnostic Pipeline Interrupted")
                    st.exception(e)
        else:
            # Placeholder empty image
            image_placeholder.markdown("""
            <div style='width:100%; height:400px; background:rgba(0,0,0,0.3); border:2px dashed rgba(255,255,255,0.1); border-radius:12px; display:flex; align-items:center; justify-content:center; color:#8ba0b8;'>
                Awaiting input image...
            </div>
            """, unsafe_allow_html=True)
    
    st.markdown("<div style='text-align:center; padding:20px; font-size:12px; color:#8ba0b8; opacity:0.6;'>© 2024 HealthAI Diagnostic Systems | Developed by Duggineni Akhil</div>", unsafe_allow_html=True)

