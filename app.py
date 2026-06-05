import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

model = tf.keras.models.load_model(
    "flower_model.h5"
)

classes = [
    "daisy",
    "dandelion",
    "roses",
    "sunflowers",
    "tulips"
]

st.title("Flower AI Classifier")

file = st.file_uploader(
    "Upload Flower Image"
)

if file:

    img = Image.open(file)

    st.image(img)

    img = img.resize((224,224))

    img = np.array(img)/255.0

    img = np.expand_dims(img,0)

    pred = model.predict(img)

    result = classes[np.argmax(pred)]

    st.success(
        f"Prediction: {result}"
    )