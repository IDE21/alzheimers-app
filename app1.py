import os
import requests
import numpy as np
import streamlit as st
from PIL import Image
from copy import deepcopy
import tensorflow as tf

# -------------------- Fix for protobuf issue --------------------
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"

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
model_path = "model.h5"
if not os.path.exists(model_path):
    download_from_huggingface(
        repo_id="Naif88/alzheimers-model",
        filename="Alzheimer’s-Disease-Model.h5",
        local_path=model_path,
        token=st.secrets["HF_TOKEN"]
    )

# Load the model
model = tf.keras.models.load_model(model_path, compile=False)

# -------------------- Utility functions --------------------
def process_image(im, target_size=(208, 176)):
    im = im.resize(target_size, resample=Image.LANCZOS)
    im = im.convert("RGB")
    im = np.array(im).astype('float32') / 255.0
    im = np.expand_dims(im, axis=0)
    return im

def get_footer():
    return """<p style='text-align: center; color: grey;'>
              <small>Alzheimer’s Disease Detection App - Powered by Streamlit</small></p>"""

def highlight_prediction(options, idx):
    options = deepcopy(options)
    highlight = f'''<span style="color:green"><b>{options[idx]}</b></span>'''
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
    options = ['NonDemented', 'VeryMildDemented', 'ModerateDemented', 'MildDemented']

    # Preprocess uploaded image
    img_in = Image.open(uploaded_file)
    img_in_processed = process_image(img_in)

    # Make prediction
    prediction = model.predict(img_in_processed).ravel()

    if len(prediction) != len(options):
        st.error(f"❌ Model output mismatch: Expected {len(options)} classes, but got {len(prediction)} outputs.")
    else:
        idx = np.argmax(prediction)

        # Show results
        st.markdown("### 🧪 Alzheimer's Class Prediction")
        st.markdown(highlight_prediction(options, idx), unsafe_allow_html=True)

        st.image(img_in, caption='🧠 Uploaded Brain Image', use_container_width=True)
