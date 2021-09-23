import numpy as np
import pickle

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer

from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, AveragePooling2D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.applications import VGG16
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import History

import matplotlib.pyplot as plt

from src.process_data import build_dataset, process_images

def build_model() -> Model:
    """
    Build model. #TODO

    Returns
    -------
    Built model.
    """
    baseline_model = VGG16(weights="imagenet", include_top=False, input_tensor=Input(shape=(224, 224, 3)))

    headline_model = baseline_model.output
    headline_model = AveragePooling2D(pool_size=(4, 4))(headline_model)
    headline_model = Flatten(name="flatten")(headline_model)
    headline_model = Dense(64, activation="relu")(headline_model)
    headline_model = Dropout(0.5)(headline_model)
    headline_model = Dense(2, activation="softmax")(headline_model)

    for layer in baseline_model.layers:
	    layer.trainable = False

    return Model(inputs=baseline_model.input, outputs=headline_model)


def save_model(model: Model, history: History) -> None:
    """
    Save a given model and its history in file.

    Parameters
    ----------
    model
        Model you want to save.
    history
        Model history.
    """
    model.save('./model/19-08-2021-3/brain_tumor_detector.h5')

    with open('./model/19-08-2021-3/history', 'wb') as f:
        pickle.dump(history.history, f)


def plot_performance(history: History) -> None:
    """
    Plot your model performance for better visualization.

    Parameters
    ----------
    history
        History of the model you want to plot performance.
    """
    accuracy = history.history['accuracy']
    val_accuracy = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    epochs_range = range(1, len(history.epoch) + 1)

    plt.figure(figsize=(15,5))

    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, accuracy, label='Train Set')
    plt.plot(epochs_range, val_accuracy, label='Validation Set')
    plt.legend(loc="best")
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.title('Model Accuracy')

    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, loss, label='Train Set')
    plt.plot(epochs_range, val_loss, label='Validation Set')
    plt.legend(loc="best")
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Model Loss')

    plt.tight_layout()
    plt.show()
