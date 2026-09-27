from dataclasses import dataclass
from pathlib import Path


@dataclass
class ExperimentConfig:
    seed: int = 42

    dataset: str = "cifar10"
    num_classes: int = 10

    batch_size: int = 128
    num_workers: int = 2

    epochs: int = 30

    learning_rate: float = 0.1
    momentum: float = 0.9
    weight_decay: float = 5e-4

    data_dir: str = "./data"
    checkpoint_dir: str = "./checkpoints"
    result_dir: str = "./results"

    device: str = "cuda"

    image_size: int = 32

    def prepare_directories(self) -> None:
        Path(self.data_dir).mkdir(parents=True, exist_ok=True)
        Path(self.checkpoint_dir).mkdir(parents=True, exist_ok=True)
        Path(self.result_dir).mkdir(parents=True, exist_ok=True)