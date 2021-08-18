import streamlit as st
import numpy as np

import cv2
from tensorflow import keras

from src.process_data import process_image


@st.cache
def load_model():
    model = keras.models.load_model("./model/18-08-2021-2/brain_tumor_detector")
    return model


def app() -> None:
    st.title("Brain tumor detector 🧠🩺")
    sidebar()

    model = load_model()

    file_uploaded = st.file_uploader(
        "File uploader", type=["jpg", "jpeg"], accept_multiple_files=False
    )

    if file_uploaded is not None:
        st.image(file_uploaded, caption="Uploaded MRI image", width=250)
        prediction = model.predict(predict(file_uploaded))
        st.markdown(
            f"**This patient {print_prediction(prediction.argmax(axis=-1))} a brain tumor with {np.amax(prediction)*100:.2f}% confidence.**"
        )


def sidebar() -> None:
    st.sidebar.title("📔 Info")
    st.sidebar.write(
        "A brain tumor is a mass or growth of abnormal cells in your brain."
    )
    st.sidebar.write("This project aims to detect brain tumor based on MRI images.")
    st.sidebar.write("The algorithm uses VGG16 layers.")

    st.sidebar.title("📊 Data")
    st.sidebar.markdown(
        "Data was found on kaggle. You can access the dataset [here](https://www.kaggle.com/navoneel/brain-mri-images-for-brain-tumor-detection)."
    )
    st.sidebar.write("It is a set of brain MRI images with and without tumors.")


def predict(path: str) -> np.array:
    file_bytes = np.asarray(bytearray(path.read()), dtype=np.uint8)
    path = cv2.imdecode(file_bytes, 1)

    img = process_image(path)
    img = np.array(img)

    images_list = []
    images_list.append(np.array(img))
    x = np.asarray(images_list)

    return x


def print_prediction(x):
    message = ""
    if x == 0:
        message = "does not have"
    else:
        message = "has"
    return message


if __name__ == "__main__":
    app()
