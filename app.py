import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# -------------------------------------------------------
# PAGE CONFIG
# -------------------------------------------------------
st.set_page_config(
    page_title="Dental Age Estimation",
    page_icon="🦷",
    layout="centered"
)

# -------------------------------------------------------
# CUSTOM CSS
# -------------------------------------------------------
st.markdown("""
    <style>

    /* Center the main content */
    .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
        max-width: 720px;
    }

    /* Header styling */
    h1 {
        text-align: center;
        font-weight: 700 !important;
    }

    /* Card container */
    .card {
        background-color: #ffffff;
        padding: 25px;
        border-radius: 12px;
        box-shadow: 0px 3px 12px rgba(0,0,0,0.12);
        margin-bottom: 25px;
    }

    /* Prediction box */
    .prediction {
        font-size: 28px;
        font-weight: 700;
        text-align: center;
        color: #2E86C1;
        margin-top: 10px;
    }

    /* Upload Section */
    .upload-title {
        text-align: center;
        font-size: 20px;
        font-weight: 600;
    }

    </style>
""", unsafe_allow_html=True)

# -------------------------------------------------------
# LOAD MODEL
# -------------------------------------------------------
MODEL_PATH = "best_age_cnn.h5"
IMG_SIZE = 224

@st.cache_resource
def load_model():
    return tf.keras.models.load_model(MODEL_PATH)

model = load_model()

# -------------------------------------------------------
# FUNCTIONS
# -------------------------------------------------------
def preprocess(img):
    img = img.resize((IMG_SIZE, IMG_SIZE))
    img = np.array(img) / 255.0
    img = np.expand_dims(img, axis=0)
    return img.astype("float32")

# -------------------------------------------------------
# SIDEBAR
# -------------------------------------------------------
st.sidebar.title("Instructions")
st.sidebar.markdown("""
1. Upload a **radiograph** of the **left maxillary canine**  
2. The image will be processed by the CNN  
3. The system predicts **chronological age**  
""")

st.sidebar.info("Model Loaded: EfficientNet-based CNN")

# -------------------------------------------------------
# MAIN HEADER
# -------------------------------------------------------
st.markdown("<h1>DENTAL AGE ESTIMATION (CNN MODEL)</h1>", unsafe_allow_html=True)
st.markdown("### AI Prediction for Left Maxillary Canine Radiographs")

# -------------------------------------------------------
# UPLOAD CARD
# -------------------------------------------------------
st.markdown('<div class="card">', unsafe_allow_html=True)

st.markdown('<p class="upload-title">Upload Radiograph Image</p>', unsafe_allow_html=True)
uploaded_file = st.file_uploader(" ", type=["jpg", "jpeg", "png"])

st.markdown('</div>', unsafe_allow_html=True)

# -------------------------------------------------------
# PROCESS & PREDICT
# -------------------------------------------------------
if uploaded_file:

    # Display Image Card
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown("### Uploaded Radiograph")
    img = Image.open(uploaded_file).convert("RGB")
    st.image(img, use_column_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

    # Processing Animation
    with st.spinner("Processing image and estimating age..."):
        processed = preprocess(img)
        pred_age = model.predict(processed)[0][0]

    # Prediction Result Card
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown("### Estimated Age")
    st.markdown(f'<div class="prediction">{pred_age:.2f} years</div>', unsafe_allow_html=True)
    st.success("Age estimation completed successfully.")
    st.markdown('</div>', unsafe_allow_html=True)

else:
    st.info("Please upload an image to begin.")
