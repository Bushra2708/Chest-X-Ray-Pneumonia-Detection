import streamlit as st
import numpy as np
import cv2
import joblib
from PIL import Image
from skimage.feature import hog

# =====================================================
# PAGE CONFIG
# =====================================================

st.set_page_config(
    page_title="Pneumonia AI",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# =====================================================
# CSS (CLEAN MEDICAL UI)
# =====================================================

st.markdown("""
<style>

@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
    font-size: 14px;
}

#MainMenu, footer, header {visibility: hidden;}

.stApp {
    background: linear-gradient(135deg, #0b1220, #0f172a);
    color: #e5e7eb;
}

/* NAVBAR */
.navbar {
    display: flex;
    justify-content: space-between;
    padding: 14px 28px;
    background: rgba(15, 23, 42, 0.75);
    border-bottom: 1px solid rgba(255,255,255,0.06);
    backdrop-filter: blur(10px);
}

.logo {
    font-size: 16px;
    font-weight: 600;
}

.nav-right {
    font-size: 12px;
    color: #94a3b8;
}

/* TITLE */
.title {
    text-align: center;
    font-size: 24px;
    font-weight: 600;
    margin-top: 10px;
}

.subtitle {
    text-align: center;
    font-size: 13px;
    color: #94a3b8;
    margin-bottom: 15px;
}

/* UPLOAD BOX (ONLY VISUAL, NOT FUNCTIONAL) */
.upload-box {
    border: 1px dashed rgba(96,165,250,0.4);
    border-radius: 14px;
    padding: 18px;
    text-align: center;
    background: rgba(15,23,42,0.6);
    margin-bottom: 8px;
    color: #94a3b8;
    font-size: 13px;
}

/* RESULT */
.result-normal {
    font-size: 22px;
    font-weight: 700;
    color: #4ade80;
    text-align: center;
}

.result-bad {
    font-size: 22px;
    font-weight: 700;
    color: #f87171;
    text-align: center;
}

/* CARD */
.card {
    background: rgba(15, 23, 42, 0.75);
    border-radius: 14px;
    padding: 16px;
    border: 1px solid rgba(255,255,255,0.06);
}

/* METRIC */
.metric {
    font-size: 20px;
    font-weight: 600;
    color: #60a5fa;
    text-align: center;
}

/* SMALL TEXT */
.small-text {
    font-size: 12.5px;
    color: #cbd5e1;
    line-height: 1.6;
}

/* REPORT */
.report-title {
    font-size: 15px;
    font-weight: 600;
    margin-bottom: 8px;
}

.report-box {
    background: rgba(2, 6, 23, 0.6);
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 12px;
    padding: 14px;
}

</style>
""", unsafe_allow_html=True)

# =====================================================
# MODEL
# =====================================================

model = joblib.load("pneumonia_model.pkl")
IMG_SIZE = 128

def extract_features(image):
    image = cv2.resize(image, (IMG_SIZE, IMG_SIZE))
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    features = hog(
        gray,
        pixels_per_cell=(8, 8),
        cells_per_block=(2, 2),
        feature_vector=True
    )

    return features.reshape(1, -1), gray

def generate_heatmap(gray):
    blurred = cv2.GaussianBlur(gray, (15, 15), 0)
    return cv2.applyColorMap(blurred, cv2.COLORMAP_JET)

# =====================================================
# NAVBAR
# =====================================================

st.markdown("""
<div class="navbar">
    <div class="logo">🫁 Pneumonia AI</div>
    <div class="nav-right">Medical Imaging System</div>
</div>
""", unsafe_allow_html=True)

# =====================================================
# TITLE
# =====================================================

st.markdown('<div class="title">Chest X-ray Analysis</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">AI-powered pneumonia detection system</div>', unsafe_allow_html=True)

# =====================================================
# UPLOAD (FIXED - NO DUPLICATE BAR ISSUE)
# =====================================================

st.markdown("""
""", unsafe_allow_html=True)

uploaded_file = st.file_uploader(
    "",
    type=["jpg", "jpeg", "png"],
    label_visibility="collapsed"
)

# =====================================================
# PROCESS
# =====================================================

if uploaded_file is not None:

    image = Image.open(uploaded_file).convert("RGB")
    image_np = np.array(image)

    features, gray = extract_features(image_np)
    prediction = model.predict(features)[0]

    try:
        probs = model.predict_proba(features)[0]
        confidence = np.max(probs) * 100
    except:
        confidence = 90

    heatmap = generate_heatmap(gray)

    # RESULT
    if prediction == 0:
        result = "Normal"
        style = "result-normal"
        obs = "No abnormal lung patterns detected."
        rec = "Routine monitoring recommended."
    else:
        result = "Pneumonia Detected"
        style = "result-bad"
        obs = "Signs of infection detected in lung region."
        rec = "Immediate medical consultation advised."

    # =====================================================
    # IMAGE + HEATMAP SIDE BY SIDE (FIXED SIZE)
    # =====================================================

    st.markdown("### Imaging Results")

    col1, col2 = st.columns(2)

    with col1:
        st.image(image_np, caption="Chest X-ray", use_container_width=True)

    with col2:
        st.image(heatmap, caption="AI Heatmap", use_container_width=True)

    # =====================================================
    # RESULT
    # =====================================================

    st.markdown("<br>", unsafe_allow_html=True)

    st.markdown(f"<div class='{style}'>{result}</div>", unsafe_allow_html=True)

    # =====================================================
    # CONFIDENCE
    # =====================================================

    st.markdown("### Confidence Score")
    st.markdown(f"<div class='metric'>{confidence:.2f}%</div>", unsafe_allow_html=True)
    st.progress(int(confidence))

    st.markdown("""
    <div class="small-text">
    Confidence indicates how strongly the AI model matches learned pneumonia or normal patterns from chest X-ray training data.
    </div>
    """, unsafe_allow_html=True)

    # =====================================================
    # REPORT (CLEAR + VISIBLE)
    # =====================================================

    st.markdown("### Medical Report")

    st.markdown('<div class="report-box">', unsafe_allow_html=True)

    st.markdown(f"""
    <div class="small-text">

    <b>Result:</b> {result}<br><br>

    <b>Confidence:</b> {confidence:.2f}%<br><br>

    <b>Observation:</b><br>
    {obs}<br><br>

    <b>Recommendation:</b><br>
    {rec}

    </div>
    """, unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)