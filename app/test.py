import cv2
import numpy as np
import streamlit as st
from PIL import Image

# Exemple 1
uploaded_file = st.file_uploader("Choose a image file", type="jpg")

if uploaded_file is not None:
    # Convert the file to an opencv image.
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    opencv_image = cv2.imdecode(file_bytes, 1)

    # Now do something with the image! For example, let's display it:
    st.image(opencv_image, channels="BGR")

# Exemple 2
uploaded_file = st.file_uploader("Upload Image")
image = Image.open(uploaded_file)
st.image(image, caption='Input', use_column_width=True)
img_array = np.array(image)
cv2.imwrite('out.jpg', cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR))