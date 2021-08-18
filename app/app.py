import streamlit as st
import numpy as np
from PIL import Image
import cv2

from tensorflow import keras

from src.process_data import process_image

def app() -> None:
    sidebar()

    st.title('Brain tumor detector 🧠🩺')

    file_uploaded = st.file_uploader('File uploader', type=['jpg'])

    if file_uploaded is not None:
        st.image(file_uploaded, caption='Uploaded Image.')
        # image = Image.open(file_uploaded)
        st.write('Prediction is', predict(file_uploaded))
        

def sidebar() -> None:
    st.sidebar.title('📔 Info')

def predict(path: str) -> str:
    file_bytes = np.asarray(bytearray(path.read()),\
    dtype=np.uint8)
    path = cv2.imdecode(file_bytes, 1)

    img = process_image(path)
    img = np.array(img)

    images_list = []
    images_list.append(np.array(img))
    x = np.asarray(images_list)
    model = keras.models.load_model('./model/18-08-2021-2/brain_tumor_detector')

    labels = ['no', 'yes']
    y_pred = model.predict([x])
    y_pred_max = y_pred.argmax(axis=-1)
    return labels[y_pred_max[0]]

if __name__=='__main__':
    app()