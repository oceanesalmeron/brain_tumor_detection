import numpy as np
from tensorflow import keras
import cv2

from src.process_data import build_dataset, process_images

path='./data/no/No11.jpg'
path='./data/yes/Y1.jpg'
img = cv2.imread(path)
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
img = cv2.resize(img, (224,224))
img = np.array(img)

images_list = []
images_list.append(np.array(img))
x = np.asarray(images_list)
model = keras.models.load_model('./model/brain_tumor_detector')

labels = ['no', 'yes']
y_pred = model.predict([x])
y_pred_max = y_pred.argmax(axis=-1)
print(y_pred)
print(labels[y_pred_max[0]])
#im = im.reshape((1,) + im.shape)
