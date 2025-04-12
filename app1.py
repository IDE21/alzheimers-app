import os
import requests
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import streamlit as st
from PIL import Image
from copy import deepcopy
from torchvision import models

# -------------------- Download model from Hugging Face --------------------
def download_from_huggingface(repo_id, filename, local_path, token):
    url = f"https://huggingface.co/{repo_id}/resolve/main/{filename}"
    headers = {"Authorization": f"Bearer {token}"}
    response = requests.get(url, headers=headers)
    if response.status_code == 200:
        with open(local_path, "wb") as f:
            f.write(response.content)
    else:
        st.error("❌ Failed to download model from Hugging Face.")
        st.stop()

# Download model if not already present
model_path = "alzheimer_model.pth"
if not os.path.exists(model_path):
    download_from_huggingface(
        repo_id="Naif88/alzheimers-model-pytorch",
        filename="alzheimer_model.pth",
        local_path=model_path,
        token=st.secrets["HF_TOKEN"]
    )

# Load the model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = models.resnet18(weights=None)
model.fc = nn.Linear(model.fc.in_features, 4)
model.load_state_dict(torch.load(model_path, map_location=device))
model = model.to(device)
model.eval()

# -------------------- Utility functions --------------------
def process_image(im):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
    return transform(im.convert("RGB")).unsqueeze(0).to(device)

def get_footer():
    return "<p style='text-align: center; color: grey;'><small>Alzheimer’s Disease Detection App - Powered by Streamlit</small></p>"

def highlight_prediction(options, idx):
    options = deepcopy(options)
    highlight = f"<span style='color:green'><b>{options[idx]}</b></span>"
    options[idx] = highlight
    return '<br>'.join(options)

# -------------------- Streamlit UI --------------------
st.markdown("<h1 style='text-align: center; color: white;'>"
            "<center>The Early Detector of Alzheimer’s Disease</center></h1>",
            unsafe_allow_html=True)

# Display logo/banner image
st.image("logo2.png", use_container_width=True)

# Informational section
st.markdown("""
### 🧠 Why Early Detection of Alzheimer’s Matters

Early detection of Alzheimer’s disease helps in:

1. **Diagnosing the disease in its early stages:**  
   It aids in identifying symptoms of Alzheimer’s before they become more apparent, allowing for early intervention.

2. **Improving disease management:**  
   With early diagnosis, doctors can develop an appropriate treatment plan that may delay the progression of symptoms.

3. **Providing support for families:**  
   It helps doctors and families understand the nature of the disease and prepare them to cope with future challenges.

4. **Reducing psychological impact:**  
   Early diagnosis can help patients better adapt to their condition and reduce the stress and anxiety caused by uncertainty.

5. **Advancing therapeutic development:**  
   Early detection contributes to evaluating the effectiveness of new treatments, which may be more successful in the early stages of the disease.

6. **Enhancing quality of life:**  
   Through early intervention, patients can maintain their independence for as long as possible, leading to a better quality of life.

> 🧬 *Early diagnosis of Alzheimer’s disease plays a crucial role in slowing disease progression and enhancing patient care.*
""", unsafe_allow_html=True)

st.markdown("<br><br>", unsafe_allow_html=True)

# Upload brain MRI image
uploaded_file = st.file_uploader("Upload a brain image to analyze for Alzheimer's...", type=["jpg", "png"])

st.markdown(get_footer(), unsafe_allow_html=True)

if uploaded_file is not None:
    options = ['Mild Demented', 'Moderate Demented', 'Non Demented', 'Very Mild Demented']

    # Preprocess uploaded image
    img_in = Image.open(uploaded_file)
    img_tensor = process_image(img_in)

    # Make prediction
    with torch.no_grad():
        output = model(img_tensor)
        prediction = torch.softmax(output, dim=1).cpu().numpy().ravel()

    if len(prediction) != len(options):
        st.error(f"❌ Model output mismatch: Expected {len(options)} classes, but got {len(prediction)} outputs.")
    else:
        idx = prediction.argmax()

        # Show results
        st.markdown("### 🧪 Alzheimer's Class Prediction")
        st.markdown(highlight_prediction(options, idx), unsafe_allow_html=True)
        st.image(img_in, caption='🧠 Uploaded Brain Image', use_container_width=True)

