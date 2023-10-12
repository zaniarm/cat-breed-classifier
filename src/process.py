from config_classes import CatBreedClassifierConfig
from typing import List
import hydra
from hydra.utils import to_absolute_path as abspath
import pandas as pd
import os
from sklearn.model_selection import train_test_split


def load_data(raw_path: str) -> List[str]:
    file_list = []
    for dirname, _, filenames in os.walk(abspath(raw_path)):
        for filename in filenames:
            file_list.append(os.path.join(dirname, filename))

    return file_list


def get_data_frame(file_list: List[str], labels: List[str]) -> pd.DataFrame:
    labels_needed = labels
    file_paths = []
    labels = []

    for image_file in file_list:
        # We don't want to save absolute filepath
        split = image_file.split(os.sep)
        label = split[-2]
        relative_path = "/".join(split[-4:])
        if label in labels_needed:
            file_paths.append(relative_path)
            labels.append(label)

    return pd.DataFrame(
        list(zip(file_paths, labels)), columns=["filepath", "label"]
    )


@hydra.main(config_path="../config", config_name="main", version_base=None)
def process_data(config: CatBreedClassifierConfig):
    file_list = load_data(config.paths.raw_images)
    df = get_data_frame(file_list, config.process.labels)

    test_ratio = config.process.split_configs.test_ratio
    validation_ratio = config.process.split_configs.validation_ratio

    X_train, X_test = train_test_split(df, test_size=test_ratio)
    X_val, X_test = train_test_split(
        X_test, test_size=test_ratio / (test_ratio + validation_ratio)
    )

    X_train.to_csv(abspath(config.processed.X_train.path))
    X_val.to_csv(abspath(config.processed.X_val.path))
    X_test.to_csv(abspath(config.processed.X_test.path))


if __name__ == "__main__":
    process_data()
