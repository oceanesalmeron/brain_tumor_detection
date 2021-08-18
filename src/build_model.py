import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer

from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, AveragePooling2D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.applications import VGG16
from tensorflow.keras.preprocessing.image import ImageDataGenerator

from src.process_data import build_dataset, process_images

def build_model():
    """
    Build model. #TODO
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

def compile_model(model, learning_rate, epochs):
    opt = Adam(learning_rate=learning_rate, decay=learning_rate / epochs)
    return model.compile(loss="binary_crossentropy", optimizer=opt, metrics=["accuracy"])


if __name__ == '__main__':
    batch_size = 8
    learning_rate = 0.0001
    epochs = 25

    print('-> Building dataset.')
    X, y = build_dataset()
    X = process_images(X)
    # y, labels = y.factorize()
    lb = LabelBinarizer()
    y = lb.fit_transform(y)
    y = to_categorical(y)
    X = np.array(X)/255.0
    y = np.array(y)
    print('x:', len(X.shape), 'y:', len(y.shape))

    print('-> Spliting into train/test.')
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=1)

    trainAug = ImageDataGenerator(rotation_range=15, fill_mode="nearest")

    print('-> Building model.')
    test = build_model()
    print('-> Compiling model.')
    opt = Adam(learning_rate=learning_rate, decay=learning_rate / epochs)
    test.compile(loss="binary_crossentropy", optimizer=opt, metrics=["accuracy"])
    print('-> Fitting.')
    test.fit_generator(
        trainAug.flow(X_train, y_train, batch_size=batch_size),
        steps_per_epoch=len(X_train) // batch_size,
        validation_data=(X_val, y_val),
        validation_steps=len(X_val) // batch_size,
        epochs=epochs
    )

    test.save('./model/18-08-2021-2/brain_tumor_detector')
    # print(X_train)