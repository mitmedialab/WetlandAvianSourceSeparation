"""train.py

The file contains all methods and classes needed to train the Conv-TasNet model
and saving its progress. The Solver class performs all the operations and is
initialized with a YAML configuration file for modularity, reproduction and 
test benchmarks.

@TODO:
    - Implement learning rate decay following original paper proceedure
"""
import os
import yaml
import torch
import torch.nn as nn

from wass.audio.dataset import CompositionDataset, ComposerConfig
from wass.convtasnet.model import Conv_TasNet
from wass.convtasnet.loss import SI_SNR
from torch.utils.data import DataLoader
from wass.utils import TrainingHistory
from torch.optim import Adam
from tqdm import tqdm


class TrainingConfig:
    """Training Configuration
        
    Helper to load and save training configurations for reproduction and 
    comparison benchmarks.

    Attributes:
        epochs {int} -- number of epoch to train the model on
        lr {float} -- learning rate for the optimizer
        max_norm {float} -- clip gradient norm
        batch_size {int} -- batch size
        n_workers {int} -- number of worker for the dataloader
        n_train {int} -- number of training samples
        n_test {int} -- number of testing samples
        composer_conf_path {str} -- path to a composer configuration
        saving_path {str} -- path where to save the experiment
        exp_name {str} -- experiment name (will be folder name within path)
        saving_rate {int} -- rate to save progress
    """

    def __init__(
        self: "TrainingConfig",
        epochs: int,
        lr: float,
        max_norm: float,
        batch_size: int,
        n_workers: int,
        n_train: int,
        n_test: int,
        composer_conf_path: str,
        saving_path: str,
        exp_name: str,
        saving_rate: int,
    ) -> None:
        """Initialization
        
        Arguments:
            epochs {int} -- number of epoch to train the model on
            lr {float} -- learning rate for the optimizer
            max_norm {float} -- clip gradient norm
            batch_size {int} -- batch size
            n_workers {int} -- number of worker for the dataloader
            n_train {int} -- number of training samples
            n_test {int} -- number of testing samples
            composer_conf_path {str} -- path to a composer configuration
            saving_path {str} -- path where to save the experiment
            exp_name {str} -- experiment name (will be folder name within path)
            saving_rate {int} -- rate to save progress
        """
        self.epochs = epochs
        self.lr = lr
        self.max_norm = max_norm
        self.batch_size = batch_size
        self.n_workers = n_workers
        self.n_train = n_train
        self.n_test = n_test
        self.composer_conf_path = composer_conf_path
        self.saving_path = saving_path
        self.exp_name = exp_name
        self.saving_rate = saving_rate

    def save(self: "TrainingConfig", path: str) -> None:
        """Save to YAML
        
        Arguments:
            path {str} -- path to save the yaml file
        """
        with open(path, "w") as f:
            yaml.dump(self.__dict__, f)

    def __str__(self: "TrainingConfig") -> str:
        """Stringify
        
        Returns:
            str -- stringified attribute list
        """
        msg = f"{self.__class__.__name__}:\n\t"
        msg += f"\n\t".join(f"{k}: {v}" for k, v in self.__dict__.items())

        return msg

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
    """Solver

    The solver encapsulate all the code to train the Conv-TasNet model using
    a standard proceedure of training:
        - For both training and testing:
            - For each epoch:
                - For each batch:
                    - forward pass
                    - backward pass
                    - clip gradients
                    - optimize
    
    Attributes:
        train_config {TrainingConfig} -- training configuration
        cuda {bool} -- use cuda GPU accelration or not ? (default: {True})
        exp_dir {str} -- experiment reference folder (save & load)
        composer_config {ComposerConfig} -- configuration to generate datasets
        train_dataset {Dataset} -- training dataset
        test_dataset {Dataset} -- testing dataset
        train_loader {DataLoader} -- training data loader
        test_loader {DataLoader} -- testing data loader
        model_path {str} -- path to model checkpoint
        model {Conv_TasNet} -- model to be trained
        criterion {SI_SNR} -- criterion to define objective function
        optim {Adam} -- adam optimize to train the model
        history_path {str} -- path to history data (progress loss data)
        history {TrainingHistory} -- training history (store loss progress)
    """

    def __init__(
        self: "Solver", train_config: "TrainingConfig", cuda: bool = True
    ) -> None:
        """Initialize
        
        Arguments:
            train_config {TrainingConfig} -- training configuration
        
        Keyword Arguments:
            cuda {bool} -- use cuda GPU accelration or not ? (default: {True})
        """
        self.train_config = train_config
        self.cuda = cuda

        self._init_exp_dir()
        self._init_datasets()
        self._init_dataloaders()
        self._init_model()
        self._init_criterion()
        self._init_optim()
        self._init_history()

    def _init_exp_dir(self: "Solver") -> None:
        """Setup Experiment folders
        """
        self.exp_dir = os.path.join(
            self.train_config.saving_path, self.train_config.exp_name
        )
        if not os.path.isdir(self.exp_dir):
            os.makedirs(self.exp_dir, exist_ok=True)

    def _init_datasets(self: "Solver") -> None:
        """Initialize Datasets from Config
        """
        self.composer_config = ComposerConfig.load(
            self.train_config.composer_conf_path
        )

        train_folder = os.path.join(self.exp_dir, "dataset/train")
        test_folder = os.path.join(self.exp_dir, "dataset/test")

        self.train_dataset = (
            CompositionDataset.generate_from_config(
                train_folder,
                self.composer_config,
                size=self.train_config.n_train,
            )
            if not os.path.isdir(train_folder)
            else CompositionDataset(train_folder)
        )
        self.test_dataset = (
            CompositionDataset.generate_from_config(
                test_folder,
                self.composer_config,
                size=self.train_config.n_test,
            )
            if not os.path.isdir(test_folder)
            else CompositionDataset(test_folder)
        )

    def _init_dataloaders(self: "Solver") -> None:
        """Initialize Data Loaders
        """
        self.train_loader = DataLoader(
            self.train_dataset,
            shuffle=True,
            batch_size=self.train_config.batch_size,
            num_workers=self.train_config.n_workers,
            pin_memory=self.cuda,
        )
        self.test_loader = DataLoader(
            self.test_dataset,
            shuffle=False,
            batch_size=self.train_config.batch_size,
            num_workers=self.train_config.n_workers,
            pin_memory=self.cuda,
        )

    def _init_model(self: "Solver") -> None:
        """Initialize Model
        
        Initialize model from scratch or from checkpoint if available.
        """
        self.model_path = os.path.join(self.exp_dir, "model.pt")
        self.model = (
            Conv_TasNet.load(self.model_path)
            if os.path.isfile(self.model_path)
            else Conv_TasNet(
                sr=self.composer_config.sr,
                n_sources=self.composer_config.n_label,
            )
        )
        if self.cuda:
            self.model = self.model.cuda()

    def _init_criterion(self: "Solver") -> None:
        """Initialize Criterion
        """
        self.criterion = SI_SNR()
        if self.cuda:
            self.criterion = self.criterion.cuda()

    def _init_optim(self: "Solver") -> None:
        """Initialize Optimize
        
        Initialize optimizer and load its weights when checkpoint is available.
        """
        self.optim = Adam(
            self.model.parameters(), lr=float(self.train_config.lr)
        )
        if os.path.isfile(self.model_path):
            package = torch.load(self.model_path)
            self.optim.load_state_dict(package["state"]["optim_dict"])

    def _init_history(self: "Solver") -> None:
        """Initialize History
        
        Initialize history from checkpoint if one already exists.
        """
        self.history_path = os.path.join(self.exp_dir, "history.csv")
        self.history = (
            TrainingHistory.load(self.history_path)
            if os.path.isfile(self.history_path)
            else TrainingHistory(
                self.train_config.saving_path, self.train_config.exp_name
            )
        )

    def __call__(self: "Solver") -> None:
        """Train the Model

        Perform training starting from last epoch and save progress if needed.
        """
        start_epoch = len(self.history)
        epochs = self.train_config.epochs
        pbar = tqdm(
            range(start_epoch, epochs), desc="Epoch", position=0, leave=True
        )
        for epoch in pbar:
            tr_loss = self._train()
            cv_loss = self._test()

            tr_tendency = self._tendency(tr_loss, "training_loss")
            cv_tendency = self._tendency(cv_loss, "validation_loss")

            self.history += (tr_loss, cv_loss)
            pbar.set_postfix(
                tr_loss=f"{tr_loss} {tr_tendency}",
                cv_loss=f"{cv_loss} {cv_tendency}",
            )

            save = ((epoch + 1) % self.train_config.saving_rate) == 0
            if save:
                self._save()

        self._save()

    def _tendency(self: "Solver", loss: float, data: str) -> str:
        """Get Loss Tendency String
        
        Arguments:
            loss {float} -- current loss
            data {str} -- name for the loss in history to compare with
        
        Returns:
            str -- tendency "-" or "↘" or "↗"
        """
        if len(self.history) == 0:
            return "-"

        last = self.history.data[data][-1]
        tendency = "↘" if loss < last else "↗" if loss > last else "-"

        return tendency

    def _train(self: "Solver") -> float:
        """Perform Training on One Epoch
        
        Returns:
            float -- train loss (averaged over batches)
        """
        self.model.train()
        tr_loss = 0.0
        pbar = tqdm(
            self.train_loader, desc="Train Batch", position=0, leave=True
        )
        for b, batch in enumerate(pbar):
            mixture, source = batch
            if self.cuda:
                mixture = mixture.cuda()
                source = source.cuda()

            estimate = self.model(mixture)
            loss = self.criterion(estimate, source)

            self.optim.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(
                self.model.parameters(), self.train_config.max_norm
            )
            self.optim.step()

            tr_loss += loss.item()
            pbar.set_postfix(tr_loss=tr_loss / (b + 1))

        tr_loss /= len(self.train_loader)

        return tr_loss

    def _test(self: "Solver") -> float:
        """Perform Testing on One Epoch
        
        Returns:
            float -- test loss (averaged over batches)
        """
        self.model.eval()
        cv_loss = 0.0
        pbar = tqdm(
            self.test_loader, desc="Test Batch", position=0, leave=True
        )
        for b, batch in enumerate(pbar):
            mixture, source = batch
            if self.cuda:
                mixture = mixture.cuda()
                source = source.cuda()

            estimate = self.model(mixture)
            loss = self.criterion(estimate, source)

            cv_loss += loss.item()
            pbar.set_postfix(cv_loss=cv_loss / (b + 1))

        cv_loss /= len(self.test_loader)

        return cv_loss

    def _save(self: "Solver") -> None:
        """Save Progress

        Saves history of losses and model state (checkpoint).
        """
        self.history.save()
        torch.save(
            Conv_TasNet.serialize(self.model, self.optim), self.model_path
        )


if __name__ == "__main__":
    B, C, S = 2, 3, 12
    print(f"B: {B}, C: {C}, S: {S}")

    model = Conv_TasNet(n_sources=C)
    optim = Adam(model.parameters())
    print("model and optim initialized")

    source = torch.randint(4, (B, C, S)).float()
    mixture = torch.sum(source, dim=1, keepdim=True)
    print(f"source: {tuple(source.shape)}")
    print(f"mixture: {tuple(mixture.shape)}")

    estimate = model(mixture)
    print(f"estimate: {tuple(estimate.shape)}")

    loss = SI_SNR()(estimate, source)
    print(f"criterion: {loss.item()}")

    optim.zero_grad()
    loss.backward()
    optim.step()
    print("backward pass and optim step success")

