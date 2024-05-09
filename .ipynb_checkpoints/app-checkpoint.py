import streamlit as st
import numpy as np
from fastai.vision.all import *
import pathlib

def load_model():
    path = pathlib.Path("my_best_model_fastai.pkl")
    model = load_learner(path)
    return model

def predict(model, image):
    img = PILImage.create(image)  # Convert uploaded image
    pred_class, pred_idx, probs = model.predict(img)
    return pred_class, probs[pred_idx]

# App Theme and Title
st.set_page_config(page_title="Image Classification App", layout="centered")
st.title("Image Classifier") 
st.subheader("Let's analyze your image!")

# File Uploader
uploaded_file = st.file_uploader("Upload an image", type=["jpg", "png", "jpeg", "webp"])

if uploaded_file is not None:
    model = load_model()
    image_bytes = uploaded_file.getvalue()

    # Image Display
    st.image(image_bytes, width=250, caption="Uploaded Image")

    # Prediction and Confidence
    with st.spinner("Analyzing image..."):  # Added a spinner for visual feedback
        pred_class, prob = predict(model, image_bytes)
    st.success("Done!")
    st.write(f"Predicted Class: {pred_class}")
    st.write(f"Confidence: {prob:.2f}") 
