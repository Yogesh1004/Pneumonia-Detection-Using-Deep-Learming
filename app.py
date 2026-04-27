import streamlit as st
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import os

# Page Configuration
st.set_page_config(
    page_title="Pneumonia Detection AI",
    page_icon="🫁",
    layout="centered"
)

# Custom CSS for Premium UI
st.markdown("""
<style>
    .main {
        background: linear-gradient(135deg, #0f0c29, #302b63, #24243e);
        color: white;
    }
    .stApp {
        background: transparent;
    }
    .css-1d391kg {
        background-color: rgba(255, 255, 255, 0.05);
        backdrop-filter: blur(10px);
        border-radius: 15px;
        padding: 2rem;
        border: 1px solid rgba(255, 255, 255, 0.1);
    }
    h1, h2, h3 {
        color: #00d2ff !important;
        text-shadow: 0 0 10px rgba(0, 210, 255, 0.5);
    }
    .stButton>button {
        background: linear-gradient(45deg, #00d2ff, #3a7bd5);
        color: white;
        border: none;
        border-radius: 25px;
        padding: 0.6rem 2rem;
        font-weight: bold;
        transition: 0.3s;
    }
    .stButton>button:hover {
        transform: scale(1.05);
        box-shadow: 0 0 20px rgba(0, 210, 255, 0.4);
    }
    .prediction-box {
        padding: 20px;
        border-radius: 10px;
        text-align: center;
        margin-top: 20px;
    }
    .pneumonia {
        background-color: rgba(255, 75, 75, 0.2);
        border: 1px solid #ff4b4b;
    }
    .normal {
        background-color: rgba(0, 204, 153, 0.2);
        border: 1px solid #00cc99;
    }
</style>
""", unsafe_allow_html=True)

# App Content
st.title("🫁 Pneumonia Detection AI")
st.markdown("### Upload a Chest X-Ray image for instant analysis")
st.write("This AI-powered tool uses Deep Learning to detect signs of pneumonia in chest radiographs.")

# Model Architecture (Must match training)
def get_model():
    model = models.mobilenet_v2(pretrained=False)
    num_ftrs = model.classifier[1].in_features
    model.classifier[1] = nn.Sequential(
        nn.Linear(num_ftrs, 512),
        nn.ReLU(),
        nn.Dropout(0.5),
        nn.Linear(512, 1),
        nn.Sigmoid()
    )
    return model

# Load Model
@st.cache_resource
def load_pneumonia_model():
    model_path = "pneumonia_model.pth"
    if os.path.exists(model_path):
        model = get_model()
        model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
        model.eval()
        return model
    return None

model = load_pneumonia_model()

uploaded_file = st.file_uploader("Choose an X-Ray image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert('RGB')
    st.image(image, caption='Uploaded X-Ray', use_container_width=True)
    
    if model is None:
        st.error("Model weights not found. Please train the model first.")
    else:
        with st.spinner('Analyzing image...'):
            # Preprocess
            preprocess = transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])
            
            img_tensor = preprocess(image).unsqueeze(0)
            
            # Predict
            with torch.no_grad():
                output = model(img_tensor)
                prediction = output.item()
            
            # Display Results
            st.divider()
            if prediction > 0.5:
                # Pneumonia
                conf = prediction * 100
                st.markdown(f"""
                <div class="prediction-box pneumonia">
                    <h2>Result: PNEUMONIA DETECTED</h2>
                    <p>Confidence Score: {conf:.2f}%</p>
                </div>
                """, unsafe_allow_html=True)
                st.warning("Possibility of Pneumonia detected. Please consult a radiologist for confirmation.")
            else:
                # Normal
                conf = (1 - prediction) * 100
                st.markdown(f"""
                <div class="prediction-box normal">
                    <h2>Result: NORMAL</h2>
                    <p>Confidence Score: {conf:.2f}%</p>
                </div>
                """, unsafe_allow_html=True)
                st.success("The X-Ray appears to be normal. No signs of pneumonia detected.")

st.markdown("---")
st.caption("Disclaimer: This tool is for educational/informational purposes and should not replace professional medical diagnosis.")
