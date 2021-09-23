import click
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator

from src.build_model import build_model, save_model, plot_performance
from src.process_data import build_dataset, process_images

@click.group(invoke_without_command=True)
@click.option(
    "--batch-size",
    required=True,
    type=int,
    help="Batch size.",
)
@click.option(
    "--learning-rate",
    required=True,
    type=float,
    help="Learning rate.",
)
@click.option(
    "--epochs",
    required=True,
    type=int,
    help="Number of epochs.",
)
def cli(
    batch_size: int,
    learning_rate: float,
    epochs: int,
):   
    print('-> Building dataset.')
    X, y = build_dataset()
    X = process_images(X)

    lb = LabelBinarizer()
    y = lb.fit_transform(y)
    y = to_categorical(y)

    X = np.array(X)/255.0
    y = np.array(y)

    print('-> Spliting into train/test.')
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=1)

    data_augmentation = ImageDataGenerator(
        rotation_range=15,
        width_shift_range=0.05,
        height_shift_range=0.05,
        rescale=1,
        shear_range=0.05,
        brightness_range=[0.1, 1.5],
        horizontal_flip=True,
        vertical_flip=True
    )

    print('-> Building model.')
    model = build_model()

    print('-> Compiling model.')
    opt = Adam(learning_rate=learning_rate, decay=learning_rate/epochs)
    model.compile(loss="binary_crossentropy", optimizer=opt, metrics=["accuracy"])

    print('-> Fitting.')
    history = model.fit_generator(
        data_augmentation.flow(X_train, y_train, batch_size=batch_size),
        steps_per_epoch=len(X_train) // batch_size,
        validation_data=(X_val, y_val),
        validation_steps=len(X_val) // batch_size,
        epochs=epochs
    )

    save_model(model, history)
    plot_performance(history)
    # test.save('./model/19-08-2021/brain_tumor_detector.h5')