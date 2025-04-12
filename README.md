# Alzheimer's Disease Detection with Deep Learning

This project leverages **Convolutional Neural Networks (CNNs)** and **ResNet50** to classify the stages of Alzheimer's disease from MRI images. The model is trained to recognize four stages: **Non-Demented**, **Very Mild Demented**, **Mild Demented**, and **Moderate Demented**.

### **Key Features**:
- **High Accuracy**: The model achieves a **99.18% test accuracy** in classifying different stages of Alzheimer's disease.
- **Real-Time Predictions**: A **Streamlit** app allows users to upload MRI images and receive instant predictions.
- **Model Visualization**: Displays the classification results with real-time feedback and visualizations.

### **Technologies Used**:
- **Deep Learning**: TensorFlow, Keras, ResNet50 for feature extraction and classification.
- **Web Interface**: Streamlit for building an interactive interface for image upload and result visualization.

### **Model Performance**:
- **Test Accuracy**: 99.18%
- **Test Loss**: 0.022

### **How It Works**:
1. **Data Preprocessing**: The MRI images are preprocessed and augmented for better model generalization.
2. **Training**: The model is built using **ResNet50** as the base, with fine-tuning for Alzheimer's disease classification.
3. **Prediction**: Once trained, the model predicts the Alzheimer's stage from the uploaded MRI scan and displays the result through the **Streamlit** app.

### **Acknowledgments**:
- A special thanks to **[@neef02](https://github.com/neef02)** for their contribution in creating the **Streamlit** interface, enabling real-time predictions and visualizations.

Clone this repository, install the dependencies, and run the Streamlit app to start classifying Alzheimer's stages from MRI images.
