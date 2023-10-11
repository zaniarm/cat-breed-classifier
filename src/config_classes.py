from dataclasses import dataclass


@dataclass
class Paths:
    data: str


@dataclass
class CatBreedClassifierConfig:
    paths: Paths
