import streamlit as st
import numpy as np
from PIL import Image
import onnxruntime as ort

# =========================================================
# Page Configuration
# =========================================================
st.set_page_config(
    page_title="Dental Age Estimation",
    page_icon="🦷",
    layout="centered"
)

# =========================================================
# Custom CSS for Modern UI
# =========================================================
st.markdown("""
    <style>
        .block-container {
            padding-top: 2rem;
            max-width: 750px;
        }

        .card {
            background: #ffffff;
            padding: 20px 25px;
            border-radius: 14px;
            box-shadow: 0 4px 14px rgba(0,0,0,0.12);
            margin-bottom: 30px;
        }

        .prediction {
            font-size: 32px;
            font-weight: 700;
            color: #2E86C1;
            text-align: center;
            margin-top: 15px;
        }

        h1 {
            text-align: center;
            font-weight: 800 !important;
        }
    </style>
""", unsafe_allow_html=True)

# =========================================================
# ONNX Model Loading
# =========================================================
MODEL_PATH = "model.onnx"
IMG_SIZE = 224

@st.cache_resource
def load_onnx_model():
    session = ort.InferenceSession(
        MODEL_PATH,
        providers=["CPUExecutionProvider"]
    )
    return session

session = load_onnx_model()

# =========================================================
# Image Preprocessing
# =========================================================
def preprocess(img):
    img = img.resize((IMG_SIZE, IMG_SIZE))
    img = np.array(img).astype("float32") / 255.0
    img = np.expand_dims(img, axis=0)   # shape: (1,224,224,3)
    return img

# =========================================================
# Sidebar Instructions
# =========================================================
st.sidebar.title("Instructions")
st.sidebar.write("""
1. Upload a **radiograph** of the **left maxillary canine**.
2. Image will be processed by CNN (ONNX runtime).
3. The model outputs an age estimate.
""")

st.sidebar.info("Model format: ONNX (TensorFlow-free)")

# =========================================================
# Main Header
# =========================================================
st.markdown("<h1>DENTAL AGE ESTIMATION (CNN – ONNX Model)</h1>", unsafe_allow_html=True)
st.markdown("### Upload a radiograph to estimate age.")

# =========================================================
# File Upload Section
# =========================================================
st.markdown('<div class="card">', unsafe_allow_html=True)
uploaded_file = st.file_uploader("Upload radiograph (JPG/PNG)", type=["jpg", "jpeg", "png"])
st.markdown("</div>", unsafe_allow_html=True)

# =========================================================
# Prediction Logic
# =========================================================
if uploaded_file:
    # Display uploaded image
    st.markdown('<div class="card">', unsafe_allow_html=True)
    img = Image.open(uploaded_file).convert("RGB")
    st.image(img, caption="Uploaded Radiograph", use_column_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

    # Preprocess
    x = preprocess(img)

    # Model inference
    inputs = {session.get_inputs()[0].name: x}
    with st.spinner("Estimating age..."):
        pred = session.run(None, inputs)[0][0][0]  # first batch, first output

    # Prediction Result
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown("### Estimated Age")
    st.markdown(f'<div class="prediction">{pred:.2f} years</div>', unsafe_allow_html=True)
    st.success("Prediction completed!")
    st.markdown("</div>", unsafe_allow_html=True)

else:
    st.info("Please upload an image to begin.")
