import base64
import streamlit as st
import numpy as np
import cv2
import os
import joblib
from PIL import Image


def local_css(file_name):
    with open(file_name) as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

def set_background(image_file):
    with open(image_file, "rb") as f:
        encoded = base64.b64encode(f.read()).decode()
    st.markdown(
        f"""
        <style>
        .stApp {{
            background-image: url("data:image/png;base64,{encoded}");
            background-size: cover;
            background-position: stretch;
            background-repeat: no-repeat;
            background-attachment: fixed;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )

local_css("style.css")
set_background("background.png")

# Load models
MODEL_PATH = "saved_models"
models = {}
if os.path.exists(MODEL_PATH):
    for f in os.listdir(MODEL_PATH):
        if f.endswith(".pkl"):
            models[f.replace(".pkl", "")] = joblib.load(os.path.join(MODEL_PATH, f))

st.title("🧠 Brain Stroke Detection System")
st.write("Upload an MRI scan to classify as **Normal** or **Stroke**.")

main_container = st.container()

with main_container:
    uploaded_file = st.file_uploader("Upload Brain MRI Image", type=["jpg", "jpeg", "png"], label_visibility="collapsed")

    if uploaded_file is not None:
        image = Image.open(uploaded_file)

        col1, col2 = st.columns(2)

        with col1:
            st.image(image, caption="Uploaded MRI", use_column_width=True, output_format='PNG')

        with col2:
            img = np.array(image)

            if len(img.shape) == 2:
                gray = cv2.resize(img, (128, 128))
            else:
                img = cv2.resize(img, (128, 128))
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            flat = gray.flatten().reshape(1, -1)

            if st.button("Analyze Brain Scan"):
                with st.container():
                    st.markdown('<div class="predictions-container">', unsafe_allow_html=True)
                    st.subheader("Model Predictions")

                    sorted_models = sorted(models.items())

                    for model_name, model in sorted_models:
                        pred = model.predict(flat)[0]
                        prob = model.predict_proba(flat)[0]
                        label = "🛑 Stroke" if pred == 1 else "✅ Normal"
                        confidence = prob[pred] * 100

                        st.markdown(f"""
                        <div class="prediction-item">
                            <b>{model_name}:</b> {label}
                            <div style="width: {confidence}%; background-color: {'#ff4b4b' if pred==1 else '#28a745'}; height: 5px; border-radius: 5px; margin-top: 2px;"></div>
                            <small>Confidence: {confidence:.2f}%</small>
                        </div>
                        """, unsafe_allow_html=True)
                    st.markdown('</div>', unsafe_allow_html=True)

