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
        with st.form("patient_form"):
            st.markdown("<strong style='color:var(--cyan);'>Primary Demographics</strong>", unsafe_allow_html=True)
            p_name = st.text_input("Full Name", value="")
            
            c1, c2 = st.columns(2)
            p_id = c1.text_input("Patient ID", value="PX-")
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
                    st.session_state['patient'] = {
                        "name": p_name,
                        "id": p_id,
                        "age": p_age,
                        "gender": p_gender,
                        "status": p_status
                    }
                    st.rerun()

else:
    patient = st.session_state['patient']
    
    # Top Navigation Bar Mockup
    st.markdown("""
    <div style='display:flex; justify-content:space-between; align-items:center; padding:15px 30px; background:rgba(16,25,41,0.6); backdrop-filter:blur(10px); border-bottom:1px solid rgba(43,62,88,0.3); margin-bottom:30px; border-radius:12px;'>
        <div style='display:flex; align-items:center; gap:10px;'>
            <svg viewBox="0 0 24 24" width="28" height="28" fill="none" stroke="var(--cyan)" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M22 12h-4l-3 9L9 3l-3 9H2"></path></svg>
            <h3 style='margin:0; color:#fff; font-weight:700; font-size:1.4rem;'>MedAI Analysis</h3>
        </div>
        <div style='display:flex; gap:30px; font-weight:600; color:#8ba0b8; font-size:14px; letter-spacing:1px;'>
            <span style='cursor:pointer;'>PATIENTS</span>
            <span style='color:var(--cyan); border-bottom:2px solid var(--cyan); padding-bottom:5px;'>ANALYSIS</span>
            <span style='cursor:pointer;'>REPORTS</span>
            <span style='cursor:pointer;'>SETTINGS</span>
        </div>
        <div style='display:flex; align-items:center; gap:12px; color:#fff; font-weight:500;'>
            <img src='https://i.pravatar.cc/100?img=11' width='36' height='36' style='border-radius:50%; border:2px solid rgba(255,255,255,0.1);'>
            <span>Dr. Alex Chen</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

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

        st.markdown("""
        <div style='background:rgba(16,25,41,0.7); padding:24px; border-radius:16px; border:1px solid rgba(43,62,88,0.5); margin-bottom:24px; box-shadow:0 10px 30px rgba(0,0,0,0.3);'>
            <strong style='color:#fff; font-size:12px; letter-spacing:1px; display:block; margin-bottom:15px;'>SCAN DETAILS</strong>
            <div style='display:flex; justify-content:space-between; margin-bottom:10px;'><span style='color:#8ba0b8; font-size:13px;'>Current Date</span><span style='color:#fff; font-size:13px;'>Just Now</span></div>
            <div style='display:flex; justify-content:space-between; margin-bottom:10px;'><span style='color:#8ba0b8; font-size:13px;'>Engine</span><span style='color:#fff; font-size:13px;'>CheXpert HD</span></div>
            <div style='display:flex; justify-content:space-between;'><span style='color:#8ba0b8; font-size:13px;'>Environment</span><span style='color:#fff; font-size:13px;'>Standard</span></div>
        </div>
        """, unsafe_allow_html=True)
        
        action_btn = st.button("REVIEW SCAN")
        if st.button("RESET OVERVIEW"):
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
                    conf_html = "<div style='background:rgba(16,25,41,0.7); padding:24px; border-radius:16px; border:1px solid rgba(43,62,88,0.5); box-shadow:0 10px 30px rgba(0,0,0,0.3);'><strong style='color:#fff; font-size:12px; letter-spacing:1px; display:block; margin-bottom:20px;'>DISEASE CONFIDENCE</strong>"
                    colors = ['#00f0ff', '#a855f7', '#ef4444', '#f59e0b', '#3b82f6', '#10b981', '#6366f1']
                    
                    sorted_preds = sorted(predictions.items(), key=lambda x: x[1], reverse=True)
                    highest_dis = sorted_preds[0] if sorted_preds else ("Unknown", 0)
                    
                    for idx, (disease, prob) in enumerate(sorted_preds[:5]):
                        percentage = prob * 100
                        color = colors[idx % len(colors)]
                        level = "High" if prob > 0.7 else "Med" if prob > 0.4 else "Low"
                        
                        conf_html += f"""
                        <div style="margin-bottom:15px;">
                            <div style="display:flex; justify-content:space-between; margin-bottom:8px; font-size:13px; font-weight:600;">
                                <span style="color:#fff;">{disease.upper()}</span>
                                <span style="color:{color};">{percentage:.1f}% <span style='opacity:0.7; font-size:11px;'>[{level}]</span></span>
                            </div>
                            <div style="width:100%; background:rgba(255,255,255,0.05); height:8px; border-radius:4px;">
                                <div style="width:{percentage}%; background:{color}; height:100%; border-radius:4px; box-shadow:0 0 10px {color};"></div>
                            </div>
                        </div>
                        """
                    conf_html += "</div>"
                    confidence_placeholder.markdown(conf_html, unsafe_allow_html=True)
                    
                    # Findings Replacement
                    findings_placeholder.markdown(f"""
                    <div style='background:rgba(16,25,41,0.7); padding:24px; border-radius:16px; border:1px solid rgba(43,62,88,0.5); height:400px; margin-bottom:24px; overflow-y:auto; box-shadow:0 10px 30px rgba(0,0,0,0.3);'>
                        <strong style='color:#fff; font-size:12px; letter-spacing:1px; display:block; margin-bottom:20px;'>FINDINGS & NOTES</strong>
                        <div style="margin-bottom:20px;">
                            <span style="color:#8ba0b8; font-size:12px; font-weight:bold;">Automated Detection</span><br>
                            <span style="color:#fff; font-size:14px; line-height:1.5;">Highest diagnostic marker: <strong>{highest_dis[0]}</strong> with {(highest_dis[1]*100):.1f}% confidence. Pathological structures detected resembling standard clinical manifestation.</span>
                        </div>
                        <div style="margin-bottom:20px;">
                            <span style="color:#8ba0b8; font-size:12px; font-weight:bold;">Dr. Review</span><br>
                            <span style="color:#fff; font-size:14px; line-height:1.5;">Pending formal validation.</span>
                        </div>
                        <div>
                            <span style="color:#8ba0b8; font-size:12px; font-weight:bold;">AI Recommendation</span><br>
                            <span style="color:{'#ff4b4b' if highest_dis[1] > 0.5 else '#00f0ff'}; font-size:14px; line-height:1.5;">{'Critical clinical anomaly recognized, prioritize review.' if highest_dis[1] > 0.5 else 'No severe anomalies observed at strict threshold.'}</span>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

                    # Line Chart (Spline / Area)
                    df = pd.DataFrame([{"Condition": k, "Probability": v*100} for k, v in predictions.items()][:5])
                    fig = px.area(
                        df, x='Condition', y='Probability', 
                        color_discrete_sequence=['#a855f7'],
                        markers=True
                    )
                    fig.update_layout(
                        height=250, 
                        paper_bgcolor='rgba(0,0,0,0)', 
                        plot_bgcolor='rgba(0,0,0,0)', 
                        font={'color': "#8ba0b8", 'family': "Inter"},
                        margin=dict(l=0, r=0, t=20, b=0),
                        yaxis=dict(gridcolor="rgba(255,255,255,0.05)", zerolinecolor="rgba(255,255,255,0.05)"),
                        xaxis=dict(gridcolor="rgba(255,255,255,0.0)", tickangle=0)
                    )
                    with chart_placeholder:
                        st.markdown("<strong style='color:#fff; font-size:12px; letter-spacing:1px; display:block; margin:20px 0 5px 0;'>DISEASE PROBABILITY DISTRIBUTION</strong>", unsafe_allow_html=True)
                        st.plotly_chart(fig, use_container_width=True)

                    # Export Section (Using Patient Variables securely)
                    report_data = {
                        "patient": {"id": patient["id"], "age": patient["age"], "gender": patient["gender"]},
                        "results": predictions
                    }
                    report_resp = requests.post(f"{API_URL}/generate-report", json=report_data)
                    if report_resp.status_code == 200:
                        st.download_button(
                            label="📄 Download Extracted PDF Report",
                            data=report_resp.content,
                            file_name=f"HealthAI_Report_{patient['id']}.pdf",
                            mime="application/pdf"
                        )

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

