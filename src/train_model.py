from typing import List

import hydra
import mlflow
import pandas as pd
from dotenv import find_dotenv, load_dotenv
from hydra.utils import to_absolute_path as abspath
from keras.applications.inception_v3 import InceptionV3
from keras.layers import Dense, Dropout, GlobalAveragePooling2D
from keras.models import Sequential
from keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator

from config_classes import CatBreedClassifierConfig

load_dotenv(find_dotenv())


def load_data(config: CatBreedClassifierConfig) -> List[pd.DataFrame]:
    X_train = pd.read_csv(abspath(config.processed.X_train.path))
    X_val = pd.read_csv(abspath(config.processed.X_val.path))

    X_train["filepath"] = X_train["filepath"].map(abspath)
    X_val["filepath"] = X_val["filepath"].map(abspath)

    return X_train, X_val


@hydra.main(
    config_path="../config",
    config_name="main",
    version_base=None,
)
def train(config: CatBreedClassifierConfig) -> None:
    X_train, X_val = load_data(config)

    img_datagen = ImageDataGenerator(
        rescale=1.0 / 255,
        rotation_range=30,
        width_shift_range=0.2,
        height_shift_range=0.2,
        horizontal_flip=True,
    )

    X_train = img_datagen.flow_from_dataframe(
        dataframe=X_train,
        x_col="filepath",
        y_col="label",
        target_size=(299, 299),
        shuffle=False,
        batch_size=30,
        seed=12,
    )

    X_val = img_datagen.flow_from_dataframe(
        dataframe=X_val,
        x_col="filepath",
        y_col="label",
        target_size=(299, 299),
        shuffle=False,
        batch_size=30,
        seed=12,
    )

    mlflow.tensorflow.autolog(registered_model_name="cat_breed_classifier")

    i_model = InceptionV3(
        weights="imagenet", include_top=False, input_shape=(299, 299, 3)
    )

    for layer in i_model.layers:
        layer.trainable = False

    model = Sequential()
    model.add(i_model)
    model.add(GlobalAveragePooling2D())
    model.add(Dense(32))
    model.add(Dropout(0.20))
    model.add(Dense(3, activation="softmax"))

    model.compile(
        optimizer=Adam(), loss="categorical_crossentropy", metrics=["accuracy"]
    )

    model.fit(
        X_train,
        validation_data=X_val,
        steps_per_epoch=len(X_train),
        validation_steps=len(X_val),
        epochs=1,
    )

    model.save(abspath(config.model.path))


if __name__ == "__main__":
    print("running")
    train()
