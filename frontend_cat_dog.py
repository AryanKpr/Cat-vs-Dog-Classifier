
import streamlit as st # type: ignore
import tensorflow as tf # type: ignore
import numpy as np # type: ignore
from PIL import Image # type: ignore

@st.cache_resource
def load_model():
    return tf.keras.models.load_model(
        "/Users/ADMIN/Desktop/cat vs dog/cat_dog_model.keras",
        compile=False
    )

model = load_model()

IMG_SIZE = (128, 128)

st.title("Cat vs Dog Classifier")
st.write("Upload an image to predict whether it is a cat or a dog.")

uploaded_file = st.file_uploader(
    "Upload an image",
    type=["jpg", "jpeg", "png"]
)

def preprocess_image(image):
    image = image.resize(IMG_SIZE)
    image = np.array(image) / 255.0
    image = np.expand_dims(image, axis=0)
    return image

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    img = preprocess_image(image)

    if st.button("Predict Animal"):
        prediction = model.predict(img)[0][0]

        confidence = prediction if prediction > 0.5 else 1 - prediction
        label = "Dog" if prediction > 0.5 else "Cat"

        if label == "Dog":
            st.error(f"Prediction: {label}")
        else:
            st.success(f"Prediction: {label}")

