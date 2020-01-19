"""dataset.py

Th file contains all the necessary classes and methods to generate and give 
access to the dataset used for training.

@TODO:
    * Use Multiprocessing for dataset generation
    * May Conflict with Random Generators
"""
import os
import yaml
import torch
import torchaudio

from wass.audio.composer import Composer
from torch.utils.data import Dataset
from typing import Tuple, List
from tqdm import tqdm


class ComposerConfig:
    """Composer Configuration

    Helper to load and save Composer Configurations to generate final dataset.
    
    Attributes:
        label_directory {str} -- path to the sequencer audio directories
        ambient_directory {str} -- path to the ambients directories
        label_size {Tuple[int, int]} -- number of sound range in the 
            sequence
        ambient_size {Tuple[int, int]} -- number of sound range in the 
            ambient
        label_scale {Tuple[float, float]} -- volume factor scale range
        ambient_scale {Tuple[float, float]} -- volume factor scale range
        duration {float} -- sequence duration in seconds
        sr {int} -- sample rate
        snr {Tuple[float, float]} -- signal to noise ratio range for ADWGN
        noise {AdditiveWhiteGaussianNoise} -- additive white gaussian noise
        sequencers {List[Sequencer]} -- sequencers
        ambients {List[Sequencer]} -- ambient generators
        focus {List[str]} -- specialize on spcific labels (default: None)

    Properties:
        n_label {int} -- number of labels/sources
    """

    def __init__(
        self: "ComposerConfig",
        label_directory: str,
        ambient_directory: str,
        label_size: Tuple[int, int] = (0, 8),
        ambient_size: Tuple[int, int] = (0, 2),
        label_scale: Tuple[float, float] = (0.5, 1.5),
        ambient_scale: Tuple[float, float] = (0.4, 1.4),
        duration: float = 4,
        sr: int = 16000,
        snr: Tuple[float, float] = (0, 100),
        focus: List[str] = None,
    ) -> None:
        """Initialization
        
        Arguments:
            label_directory {str} -- path to the sequencer audio directories
            ambient_directory {str} -- path to the ambients directories
            
        Keyword Arguments:
            label_size {Tuple[int, int]} -- number of sound range in the 
                sequence (default: {[0, 8]})
            ambient_size {Tuple[int, int]} -- number of sound range in the 
                ambient (default: {[0, 2]})
            label_scale {Tuple[float, float]} -- volume factor scale range 
                (default: {[0.5, 1.5]})
            ambient_scale {Tuple[float, float]} -- volume factor scale range
                (default: {[0.4, 1.4]})
            duration {float} -- sequence duration in seconds (default: {4})
            sr {int} -- sample rate (default: {16000})
            snr {Tuple[float, float]} -- signal to noise ratio range for ADWGN 
                (default: {(0, 54)})
            focus {List[str]} -- specialize on spcific labels (default: None)
        """
        self.label_directory = label_directory
        self.ambient_directory = ambient_directory
        self.label_size = label_size
        self.ambient_size = ambient_size
        self.label_scale = label_scale
        self.ambient_scale = ambient_scale
        self.duration = duration
        self.sr = sr
        self.snr = snr
        self.focus = focus

    @property
    def n_label(self: "ComposerConfig") -> int:
        """Number of Labels/Sources
        
        Returns:
            int -- number of labels/sources
        """
        if self.focus is None:
            path_fn = lambda dir: os.path.join(self.label_directory, dir)
            filter_fn = lambda dir: os.path.isdir(path_fn(dir))

            dirs = os.listdir(self.label_directory)
            filtered = list(filter(filter_fn, dirs))
            n_label = len(filtered)
        else:
            n_label = len(self.focus)

        return n_label

    def save(self: "ComposerConfig", path: str) -> None:
        """Save to YAML
        
        Arguments:
            path {str} -- path to save the yaml file
        """
        with open(path, "w") as f:
            yaml.dump(self.__dict__, f)

    @classmethod
    def load(cls: "ComposerConfig", path: str) -> "ComposerConfig":
        """Load from YAML
        
        Returns:
            ComposerConfig -- loaded composer configuration
        """
        with open(path, "r") as f:
            conf = cls(**yaml.load(f))

        return conf


class CompositionDataset(Dataset):
    """Composition Dataset
    
    Attributes:
        path {str} -- path to the dataset folder
        folders {List[str]} -- composition folders from the dataset
        size {int} -- number of composition in the dataset
    """

    def __init__(self: "CompositionDataset", path: str) -> None:
        """Initialization
        
        Arguments:
            path {str} -- path to the dataset folder
        """
        super(CompositionDataset, self).__init__()
        self.path = path
        self.folders = sorted(
            [
                os.path.join(path, dir)
                for dir in os.listdir(path)
                if os.path.isdir(os.path.join(path, dir))
            ]
        )
        self.size = len(self.folders)

    def __len__(self: "CompositionDataset") -> int:
        """Length
        
        Returns:
            int -- number of composition in the dataset
        """
        return self.size

    def __getitem__(
        self: "CompositionDataset", idx: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get Item at Index
        
        Arguments:
            idx {int} -- index of the requested datum
        
        Returns:
            Tuple[torch.Tensor, torch.Tensor] -- loaded composition and 
                sequences
        """
        folder = self.folders[idx]
        composition_path = os.path.join(folder, "composition.wav")
        sequences_path = sorted(
            [
                os.path.join(folder, file)
                for file in os.listdir(folder)
                if "sequence" in file
            ]
        )

        composition, _ = torchaudio.load(composition_path)
        composition = composition.mean(dim=0, keepdim=True)
        _type = composition.type()

        sequences = torch.zeros((len(sequences_path), composition.size(-1)))
        sequences = sequences.type(_type)
        for s, sequence_path in enumerate(sequences_path):
            sequence, _ = torchaudio.load(sequence_path)
            sequence = sequence.mean(dim=0)
            sequences[s] = sequence.type(_type)

        return composition, sequences

    @classmethod
    def generate_from_default(
        cls: "CompositionDataset",
        dest: str,
        label_directory: str,
        ambient_directory: str,
        size: int,
    ) -> "CompositionDataset":
        """Generate Dataset from Default Composer Configuration
        
        Arguments:
            dest {str} -- destination path to the dataset folder
            label_directory {str} -- path to the sequencer audio directories
            ambient_directory {str} -- path to the ambients directories
            size {int} -- number of generated composition for the dataset

        Returns:
            CompositionDataset -- dataset after beign generated
        """
        conf = ComposerConfig(label_directory, ambient_directory)
        dataset = cls.generate_from_config(dest, conf, size)

        return dataset

    @classmethod
    def generate_from_config(
        cls: "CompositionDataset",
        dest: str,
        conf: "ComposerConfig",
        size: int,
    ) -> "CompositionDataset":
        """Generate Dataset from Given Composer Configuration
        
        Arguments:
            dest {str} -- destination path to the dataset folder
            config {str} --composer configuration
            size {int} -- number of generated composition for the dataset

        Returns:
            CompositionDataset -- dataset after beign generated
        """
        if not os.path.isdir(dest):
            os.makedirs(dest, exist_ok=True)

        conf_path = os.path.join(dest, "composer.yaml")
        conf.save(conf_path)

        composer = Composer(**conf.__dict__)
        pbar = tqdm(range(size), desc=f"Generating Dataset [{dest}]")
        for i in pbar:
            composition, sequences = next(composer)
            folder = os.path.join(dest, f"{(i+1):06d}")
            os.mkdir(folder)

            composition_path = os.path.join(folder, "composition.wav")
            torchaudio.save(composition_path, composition, conf.sr)

            for s, sequence in enumerate(sequences):
                sequence = sequence.unsqueeze(0)
                sequence_path = os.path.join(
                    folder, f"sequence_{(s+1):02d}.wav"
                )
                torchaudio.save(sequence_path, sequence, conf.sr)

        dataset = cls(dest)

        return dataset

