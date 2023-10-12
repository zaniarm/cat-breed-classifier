import hydra

from config_classes import CatBreedClassifierConfig
from evaluate import evaluate_model
from process import process_data
from train_model import train


@hydra.main(config_path="../config", config_name="main", version_base=None)
def main(config: CatBreedClassifierConfig) -> None:
    process_data(config)
    train(config)
    evaluate_model(config)


if __name__ == "__main__":
    main()
