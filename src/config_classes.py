from dataclasses import dataclass
from typing import List


@dataclass
class PathsConfig:
    raw_images: str


@dataclass
class ProceccedDataFormatConfig:
    name: str
    path: str


class ProcessConfig:
    labels: List[str]


@dataclass
class ProcessedConfig:
    dir: str
    X_train: ProceccedDataFormatConfig
    X_val: ProceccedDataFormatConfig
    X_test: ProceccedDataFormatConfig


@dataclass
class ModelMainConfig:
    dir: str
    name: str
    path: str


@dataclass
class CatBreedClassifierConfig:
    paths: PathsConfig
    processed: ProcessedConfig
    model: ModelMainConfig
    process: ProcessConfig
