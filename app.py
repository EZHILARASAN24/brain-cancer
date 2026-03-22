import streamlit as st
import numpy as np
import cv2
import os
import joblib
from PIL import Image

# Load models
MODEL_PATH = "saved_models"
models = {}
if os.path.exists(MODEL_PATH):
    for f in os.listdir(MODEL_PATH):
        if f.endswith(".pkl"):
            models[f.replace(".pkl", "")] = joblib.load(os.path.join(MODEL_PATH, f))

st.title("🧠 Brain Stroke Detection System")
st.write("Upload an MRI scan to classify as **Normal** or **Stroke**.")

uploaded_file = st.file_uploader("Upload Brain MRI Image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded MRI", use_column_width=True)

    # Convert & preprocess
    # Preprocess
    img = np.array(image)

    # Ensure 3-channel BGR if needed
    if len(img.shape) == 2:  # already grayscale
        gray = cv2.resize(img, (128, 128))
    else:
        img = cv2.resize(img, (128, 128))
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    flat = gray.flatten().reshape(1, -1)

    st.subheader("Choose Model")
    model_name = st.selectbox("Select a trained model", list(models.keys()))

    if st.button("Predict"):
        model = models[model_name]
        pred = model.predict(flat)[0]
        prob = model.predict_proba(flat)[0]

        label = "🛑 Stroke Detected" if pred == 1 else "✅ Normal"
        st.success(f"Prediction: {label}")
        st.write(f"Confidence: {prob[pred]*100:.2f}%")

