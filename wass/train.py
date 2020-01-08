import os
import yaml
import torch

from wass.audio.dataset import CompositionDataset, ComposerConfig
from wass.convtasnet.loss import SI_SNR_Criterion
from wass.convtasnet.model import Conv_TasNet
from torch.utils.data import DataLoader
from wass.utils import TrainingHistory
from torch.optim import Adam


class TrainingConfig:
    """Training Configuration
        
    Helper to load and save training configurations for reproduction and 
    comparison benchmarks.

    Attributes:
        epochs {int} -- number of epoch to train the model on
        lr {int} -- learning rate for the optimizer
        batch_size {int} -- batch size
        n_workers {int} -- number of worker for the dataloader
        n_train {int} -- number of training samples
        n_test {int} -- number of testing samples
        composer_conf_path {str} -- path to a composer configuration
    """

    def __init__(
        self: "TrainingConfig",
        epochs: int,
        lr: int,
        batch_size: int,
        n_workers: int,
        n_train: int,
        n_test: int,
        composer_conf_path: str,
    ) -> None:
        """Initialization
        
        Arguments:
            epochs {int} -- number of epoch to train the model on
            lr {int} -- learning rate for the optimizer
            batch_size {int} -- batch size
            n_workers {int} -- number of worker for the dataloader
            n_train {int} -- number of training samples
            n_test {int} -- number of testing samples
            composer_conf_path {str} -- path to a composer configuration
        """
        self.epochs = epochs
        self.lr = lr
        self.batch_size = batch_size
        self.n_workers = n_workers
        self.n_train = n_train
        self.n_test = n_test
        self.composer_conf_path = composer_conf_path

    def save(self: "TrainingConfig", path: str) -> None:
        """Save to YAML
        
        Arguments:
            path {str} -- path to save the yaml file
        """
        with open(path, "w") as f:
            yaml.dump(self.__dict__, f)

    @classmethod
    def load(cls: "TrainingConfig", path: str) -> "TrainingConfig":
        """Load from YAML
        
        Returns:
            TrainingConfig -- loaded training configuration
        """
        with open(path, "r") as f:
            conf = cls(**yaml.load(f))

        return conf


class Solver:
    def __init__(self: "Solver") -> None:
        pass
