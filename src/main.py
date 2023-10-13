import hydra
import os
from config_classes import CatBreedClassifierConfig
from evaluate import evaluate_model
from process import process_data
from train_model import train
from dotenv import load_dotenv, find_dotenv

load_dotenv(find_dotenv())

os.environ["HYDRA_FULL_ERROR"] = "1"


@hydra.main(config_path="../config", config_name="main", version_base=None)
def main(config: CatBreedClassifierConfig) -> None:
    process_data(config)
    train(config)
    evaluate_model(config)


if __name__ == "__main__":
    main()
