import os
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
from sklearn.model_selection import train_test_split

from config_classes import CatBreedClassifierConfig

load_dotenv(find_dotenv())


def get_filelist(config: CatBreedClassifierConfig) -> List[str]:
    file_list = []
    for dirname, _, filenames in os.walk(abspath(config.paths.raw_images)):
        for filename in filenames:
            file_list.append(os.path.join(dirname, filename))

    return file_list


def generate_pandas_df(
    file_list: List[str], config: CatBreedClassifierConfig
) -> pd.DataFrame:
    labels_needed = config.process.labels
    file_paths = []
    labels = []

    for image_file in file_list:
        label = image_file.split(os.sep)[-2]
        if label in labels_needed:
            file_paths.append(image_file)
            labels.append(label)

    return pd.DataFrame(
        list(zip(file_paths, labels)), columns=["filepath", "label"]
    )


def load_data(config: CatBreedClassifierConfig):
    file_list = get_filelist(config)
    df = generate_pandas_df(file_list, config)

    # train_ratio = 0.75
    validation_ratio = 0.10
    test_ratio = 0.25

    train, test = train_test_split(df, test_size=test_ratio)
    val, test = train_test_split(
        test, test_size=test_ratio / (test_ratio + validation_ratio)
    )

    img_datagen = ImageDataGenerator(
        rescale=1.0 / 255,
        rotation_range=30,
        width_shift_range=0.2,
        height_shift_range=0.2,
        horizontal_flip=True,
    )

    x_train = img_datagen.flow_from_dataframe(
        dataframe=train,
        x_col="filepath",
        y_col="label",
        target_size=(299, 299),
        shuffle=False,
        batch_size=30,
        seed=12,
    )
    x_val = img_datagen.flow_from_dataframe(
        dataframe=val,
        x_col="filepath",
        y_col="label",
        target_size=(299, 299),
        shuffle=False,
        batch_size=30,
        seed=12,
    )
    x_test = img_datagen.flow_from_dataframe(
        dataframe=test,
        x_col="filepath",
        y_col="label",
        target_size=(299, 299),
        shuffle=False,
        batch_size=30,
        seed=12,
    )

    return x_train, x_val, x_test


@hydra.main(
    config_path="../config",
    config_name="main",
    version_base=None,
)
def train(config: CatBreedClassifierConfig) -> None:
    x_train, x_val, x_test = load_data(config)
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
        x_train,
        validation_data=x_val,
        steps_per_epoch=len(x_train),
        validation_steps=len(x_val),
        epochs=1,
    )

    model.save(abspath(config.model.path))
    test_accuracy = model.evaluate(x_test)[1] * 100
    print("Test accuracy is : ", test_accuracy, "%")


if __name__ == "__main__":
    print("running")
    train()
