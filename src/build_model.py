from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, AveragePooling2D, Flatten, Dense, Dropout
from tensorflow.keras.applications import VGG16

def build_model() -> None:
    """
    Build model. #TODO
    """
    baseline_model = VGG16(weights="imagenet", include_top=False,input_tensor=Input(shape=(224, 224, 3)))

    headline_model = baseline_model.output
    headline_model = AveragePooling2D(pool_size=(4, 4))(headline_model)
    headline_model = Flatten(name="flatten")(headline_model)
    headline_model = Dense(64, activation="relu")(headline_model)
    headline_model = Dropout(0.5)(headline_model)
    headline_model = Dense(2, activation="softmax")(headline_model)

    return Model(inputs=baseline_model.input, outputs=headline_model)


if __name__ == '__main__':
    test = build_model()
    print(type(test))
    print(test.summary())