import streamlit as st
import numpy as np
from fastai.vision.all import *
import pathlib
import random
from PIL import Image
import requests
from io import BytesIO
import base64


# Setting the page config
st.set_page_config(page_title="Deepfake Image Detector App", layout="wide")

# Styling
def local_css(file_name):
    with open(file_name) as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

local_css("styles.css")

# Functions
def load_model():
    path = pathlib.Path("my_best_model_fastai.pkl")
    model = load_learner(path)
    return model

def predict(model, image):
    img = PILImage.create(image)
    pred_class, pred_idx, probs = model.predict(img)
    return pred_class, probs[pred_idx]

# Navigation Menu
with st.sidebar:
    selected_page = st.selectbox("Select a page:", ["Welcome", "Image Classification", "Educational Game", "Did you know?"])

# Welcome Page
if selected_page == "Welcome":
    st.title("Hiya!! Welcome to the AI Image Detection App")
    st.subheader("Explore the power of AI in image classification and have fun with my educational game!")
    st.write("Navigate the pages from the sidebar to get started.")

    # Embed YouTube video
    st.write("Check out our introductory video below:")
    st.markdown('<iframe width="560" height="315" src="https://www.youtube.com/embed/5Viy3Cu3DLk" frameborder="0" allowfullscreen></iframe>', unsafe_allow_html=True)

# Classification Page
elif selected_page == "Image Classification":
    st.title("Image Classifier")
    st.subheader("Let's analyze your image!")

    uploaded_file = st.file_uploader("Upload an image", type=["jpg", "png", "jpeg", "webp"])
    if uploaded_file is not None:
        model = load_model()
        image_bytes = uploaded_file.getvalue()

        # Image Display
        st.image(image_bytes, width=300, caption="Uploaded Image")

        # Prediction and Confidence
        with st.spinner("Analyzing image..."):
            pred_class, prob = predict(model, image_bytes)
        st.success("Done!")
        st.write(f"**Predicted Class:** {pred_class}")
        st.write(f"**Confidence:** {prob:.2f}")

    # Add disclaimer
    st.write("*Disclaimer: The image classification results provided by this application may not be 100% accurate. The predictions are based on the model's training data and may not always reflect the true content of the uploaded image.*")

# Educational Gamepage
elif selected_page == "Educational Game":
    st.title("Educational Game")
    st.write("Can you pick AI works vs human works?")

    sample_images = {
        'img/29.jpeg': 'AI', 'img/30.jpeg': 'AI', 'img/101.jpeg': 'Real', 
        'img/102.jpeg': 'Real', 'img/115.jpeg': 'Real', 'img/135 (1).jpeg': 'Real', 
        'img/148.jpeg': 'AI', 'img/200.jpeg': 'Real', 
        'img/221.jpeg': 'AI', 'img/227.jpeg': 'AI'
    }

    # Initialize session state
    if 'user_answers' not in st.session_state:
        st.session_state.user_answers = {}

    # Function to update user answers
    def update_user_answer(image_path, user_choice):
        st.session_state.user_answers[image_path] = user_choice

    # Display images and collect user choices
    for img_path, label in sample_images.items():
        st.write(f"Image {img_path.split('/')[-1]}")
        image = Image.open(img_path)
        st.image(image, width=600, use_column_width=False)

        user_choice = st.radio("Is this AI work or Real human work?", options=['AI', 'Real'], key=img_path)
        update_user_answer(img_path, user_choice)

    # Check Answers Button
    if st.button("Check Answers"):
        score = sum(1 for img_path, user_choice in st.session_state.user_answers.items() if sample_images[img_path] == user_choice)
        st.write(f"Your score: {score}/{len(sample_images)}")

# New Knowledge Page
elif selected_page == "Did you know?":
    st.title("Now you know!!")
    st.subheader("A quick pill of fact for you!")

    # Random fact
    fun_facts = [
        "Did you know? The term 'Artificial Intelligence' was coined by John McCarthy in 1956.",
        "Fun Fact: The first AI program was developed in 1951 by Christopher Strachey.",
        "Interesting: AI can help doctors detect diseases earlier and more accurately.",
        "Fascinating: AlphaGo, an AI program developed by DeepMind, defeated world champion Go player Lee Sedol in 2016.",
        "Impressive: Deep Blue, a chess-playing computer developed by IBM, defeated world chess champion Garry Kasparov in 1997.",
        "Intriguing: Machine learning allows computers to learn and improve from experience without being explicitly programmed.",
        "Remarkable: Neural networks, inspired by the structure and function of the human brain, are a key component of deep learning.",
        "Did you know? Natural Language Processing (NLP) enables computers to understand and interpret human language.",
        "Fun Fact: Computer vision allows machines to analyze and interpret visual information from the real world.",
        "Interesting: Reinforcement learning is a type of machine learning where agents learn to make decisions through trial and error.",
        "Fascinating: AI applications in healthcare include medical image analysis, predictive analytics, and personalized treatment plans.",
        "Impressive: Autonomous vehicles use AI to perceive their environment and navigate without human intervention.",
        "Intriguing: AI is used in finance for fraud detection, algorithmic trading, and risk management.",
        "Remarkable: Chatbots leverage AI to simulate human conversation and provide automated responses.",
        "Did you know? Virtual assistants like Siri and Alexa use AI to perform tasks and answer user inquiries.",
        "Fun Fact: Explainable AI aims to make AI systems transparent and understandable to users.",
        "Interesting: AI ethics involves establishing principles to ensure responsible development and deployment of AI technologies.",
        "Fascinating: Generative Adversarial Networks (GANs) consist of two neural networks that compete to generate realistic data.",
        "Impressive: AI is used to create art through techniques like style transfer and generative algorithms.",
        "Intriguing: AI-powered educational assistants provide personalized learning experiences and adaptive tutoring.",
        "Remarkable: AI enhances marketing efforts through customer segmentation, predictive analytics, and personalized recommendations.",
        "Did you know? AI algorithms can exhibit biases based on the data they are trained on.",
        "Fun Fact: AI is used in gaming for NPC behavior, procedural content generation, and adaptive difficulty adjustment.",
        "Interesting: AI applications in agriculture include crop monitoring, yield prediction, and pest detection.",
        "Fascinating: AI helps optimize energy production and distribution, predict demand, and improve renewable energy efficiency.",
        "Impressive: AI is used for wildlife monitoring, habitat protection, and climate modeling in environmental conservation.",
        "Intriguing: Robotics relies on AI for perception, planning, and interaction with humans.",
    ]

    fact = random.choice(fun_facts)
    st.markdown(f"<p style='font-size: 24px;'>**{fact}**</p>", unsafe_allow_html=True)
    st.image("44.jpeg", caption="AI in action!", width=500, use_column_width=False)