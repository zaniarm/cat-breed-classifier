from typing import Any, List

import hydra
import pandas as pd
import tensorflow as tf
from dotenv import find_dotenv, load_dotenv
from hydra.utils import to_absolute_path as abspath
from keras.preprocessing.image import ImageDataGenerator

from config_classes import CatBreedClassifierConfig

load_dotenv(find_dotenv())


def load_data(processed_X_test_path: str) -> List[pd.DataFrame]:
    X_test = pd.read_csv(processed_X_test_path)
    X_test["filepath"] = X_test["filepath"].map(abspath)

    return X_test


def load_model(model_filepath: str) -> Any | None:
    return tf.keras.saving.load_model(model_filepath)


@hydra.main(config_path="../config", config_name="main", version_base=None)
def evaluate_model(config: CatBreedClassifierConfig):
    X_test = load_data(abspath(config.processed.X_test.path))
    model = load_model(abspath(config.model.path))

    img_datagen = ImageDataGenerator(
        rescale=1.0 / 255,
        rotation_range=30,
        width_shift_range=0.2,
        height_shift_range=0.2,
        horizontal_flip=True,
    )

    X_test = img_datagen.flow_from_dataframe(
        dataframe=X_test,
        x_col="filepath",
        y_col="label",
        target_size=(299, 299),
        shuffle=False,
        batch_size=30,
        seed=12,
    )

    test_accuracy = model.evaluate(X_test)[1] * 100
    print(test_accuracy)
    # TODO: log to mlflow


if __name__ == "__main__":
    evaluate_model()
