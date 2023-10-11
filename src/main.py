import hydra

from src.config import CatBreedClassifierConfig


@hydra.main(config_path="../config", config_name="main", version_base=None)
def main(config: CatBreedClassifierConfig) -> None:
    print(config.paths.data)


if __name__ == "__main__":
    main()
