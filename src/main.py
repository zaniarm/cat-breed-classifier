import hydra

from config_classes import CatBreedClassifierConfig
from train import train


@hydra.main(config_path="../config", config_name="main", version_base=None)
def main(config: CatBreedClassifierConfig) -> None:
    train(config)


if __name__ == "__main__":
    # pylint: disable=no-value-for-parameter
    main()
