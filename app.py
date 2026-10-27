"""
app.py
------
Streamlit Web Application for Vehicle Type Classification.
Classifies uploaded vehicle images into: Car, Bus, Truck, Motorcycle.
Uses a transfer learning model powered by MobileNetV2.

Usage:
    streamlit run app.py
"""

import json
from pathlib import Path
import numpy as np
from PIL import Image
import streamlit as st

# Configure Streamlit page layout and appearance
st.set_page_config(
    page_title="Vehicle Type Classifier",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Project paths
BASE_DIR = Path(__file__).resolve().parent
MODELS_DIR = BASE_DIR / "models"
MODEL_PATH = MODELS_DIR / "vehicle_classifier_mobilenetv2.keras"
CLASS_INDICES_PATH = MODELS_DIR / "class_indices.json"

# Keep Keras cache locally inside workspace
import os
os.environ["KERAS_HOME"] = str(BASE_DIR / ".keras")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

IMAGE_SIZE = (224, 224)
DEFAULT_CLASSES = {0: "Bus", 1: "Car", 2: "Motorcycle", 3: "Truck"}


@st.cache_resource
def load_trained_model():
    """
    Loads and caches the trained Keras MobileNetV2 model and class mapping.
    Using st.cache_resource ensures the model is loaded only once in memory.
    """
    if not MODEL_PATH.exists():
        return None, None

    try:
        import keras
        model = keras.models.load_model(str(MODEL_PATH))

        if CLASS_INDICES_PATH.exists():
            with open(CLASS_INDICES_PATH, "r") as f:
                mapping = json.load(f)
                # Ensure integer keys
                class_mapping = {int(k): v for k, v in mapping.items()}
        else:
            class_mapping = DEFAULT_CLASSES

        return model, class_mapping
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None, None


def preprocess_image(image: Image.Image) -> np.ndarray:
    """
    Preprocess image for MobileNetV2 inference:
    1. Convert image to RGB (handling PNG alpha channels or grayscale)
    2. Resize to 224x224
    3. Convert to float numpy array with pixel values in range [0, 255]
    4. Expand dimensions to shape (1, 224, 224, 3)

    Important: The trained model architecture in train.py already includes
    preprocess_input inside its graph (true_divide and subtract layers).
    Passing raw [0, 255] values ensures preprocessing is applied exactly once.
    """
    img = image.convert("RGB")
    img = img.resize(IMAGE_SIZE, Image.Resampling.BILINEAR)
    img_array = np.array(img, dtype=np.float32)
    img_batch = np.expand_dims(img_array, axis=0)
    return img_batch


def predict_vehicle(model, class_mapping, image: Image.Image):
    """
    Runs model inference on the preprocessed image.
    Returns:
    - predicted_class: str
    - confidence: float (0.0 to 1.0)
    - probabilities: dict of {class_name: confidence}
    """
    input_tensor = preprocess_image(image)
    raw_preds = model.predict(input_tensor, verbose=0)[0]

    # Map predictions to class names
    probabilities = {}
    for idx, prob in enumerate(raw_preds):
        class_name = class_mapping.get(idx, f"Class_{idx}")
        probabilities[class_name] = float(prob)

    top_idx = int(np.argmax(raw_preds))
    top_class = class_mapping.get(top_idx, f"Class_{top_idx}")
    top_confidence = float(raw_preds[top_idx])

    return top_class, top_confidence, probabilities


def main():
    # ------------------ Sidebar ------------------
    with st.sidebar:
        st.header("⚙️ Project Overview")
        st.markdown(
            """
            **Vehicle Type Classification**
            - **Target Classes**: Car, Bus, Truck, Motorcycle
            - **Base Architecture**: MobileNetV2 (ImageNet Pretrained)
            - **Technique**: Transfer Learning
            - **Framework**: TensorFlow & Keras
            """
        )

        st.markdown("---")
        st.subheader("📊 Model Status")
        if MODEL_PATH.exists():
            st.success(f"Model found:\n`{MODEL_PATH.name}`")
        else:
            st.warning("Model not found.\nRun `python train.py` first to train and save the model.")

    # ------------------ Main UI ------------------
    st.title("🚗 Vehicle Type Classification")
    st.caption("Upload a vehicle picture to predict its category (Car, Bus, Truck, or Motorcycle).")

    # If model is not trained yet, show a helpful banner
    model, class_mapping = load_trained_model()
    if model is None:
        st.info(
            """
            ### 📌 Welcome to your first ML project!
            The model weights have not been generated yet.

            **To train your model, open PowerShell in this project folder and run:**
            ```powershell
            # Train and evaluate the MobileNetV2 model
            python train.py --epochs 5
            ```
            Once training finishes, refresh this browser tab to test predictions!
            """
        )
        return

    # Image Input Section
    uploaded_file = st.file_uploader(
        "Choose a vehicle image (JPG, JPEG, PNG)...",
        type=["jpg", "jpeg", "png", "bmp"]
    )

    image_to_predict = None
    image_source_label = ""

    if uploaded_file is not None:
        try:
            image_to_predict = Image.open(uploaded_file)
            image_source_label = f"Uploaded File: {uploaded_file.name}"
        except Exception as e:
            st.error(f"Could not open uploaded image: {e}")

    # Display image and run prediction
    if image_to_predict is not None:
        col1, col2 = st.columns([1, 1], gap="large")

        with col1:
            st.subheader("🖼️ Input Image")
            st.image(image_to_predict, caption=image_source_label, use_container_width=True)

        with col2:
            st.subheader("🎯 Prediction Result")
            with st.spinner("Analyzing image with MobileNetV2..."):
                top_class, top_confidence, probabilities = predict_vehicle(
                    model, class_mapping, image_to_predict
                )

            # Class badge styling
            badge_icons = {
                "Car": "🚗",
                "Bus": "🚌",
                "Truck": "🚚",
                "Motorcycle": "🏍️"
            }
            icon = badge_icons.get(top_class, "🚘")

            st.metric(
                label="Predicted Vehicle Type",
                value=f"{icon} {top_class}",
                delta=f"{top_confidence * 100:.2f}% Confidence"
            )

            st.markdown("#### Class Probabilities Breakdown")
            for cls_name, prob in sorted(probabilities.items(), key=lambda x: x[1], reverse=True):
                col_name, col_bar, col_pct = st.columns([2, 5, 2])
                with col_name:
                    st.write(f"**{badge_icons.get(cls_name, '')} {cls_name}**")
                with col_bar:
                    st.progress(float(prob))
                with col_pct:
                    st.write(f"`{prob * 100:.2f}%`")

    else:
        st.info("👆 Please upload a vehicle image to see predictions.")


if __name__ == "__main__":
    main()
