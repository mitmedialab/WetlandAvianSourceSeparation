"""composer.py

Th file contains all the necessary classes and methods to generate natural 
widelife audios based on reference samples for difference species and different
ambient sounds. The Composer is the one meant to be used to generate such 
audio.

@TODO: 
    * Add time stretching variation to samples
    * Add pitch shifting variation to samples
    * See https://librosa.github.io/librosa/_modules/librosa/
        effects.html#time_stretch
        effects.html#pitch_shift
"""
import os
import torch
import torchaudio
import numpy as np
import torch.nn as nn

from collections import OrderedDict
from typing import Tuple, List


class AdditiveWhiteGaussianNoise(nn.Module):
    """Additive White Gaussian Noise (ADWGN)
    
    Pytorch implementation of ADWGN adapted from the following link:
        - https://stackoverflow.com/a/53688043

    Attributes:
        snr {Tuple[float, float]} -- signal to noise ratio range
    """

    def __init__(
        self: "AdditiveWhiteGaussianNoise", snr: Tuple[float, float]
    ) -> None:
        """Initialization

        Arguments:
            snr {Tuple[float, float]} -- signal to noise ratio range
        """
        self.snr = snr

    def __call__(
        self: "AdditiveWhiteGaussianNoise", X: torch.Tensor
    ) -> torch.Tensor:
        """Call
        
        Arguments:
            X {torch.Tensor} -- input tensor to add noise on
        
        Returns:
            torch.Tensor -- noisy output tensor
        """
        regsnr = np.random.uniform(*self.snr)
        length = X.size(-1)

        signal_power = torch.sum(torch.pow(X, 2)).item() / length
        noise_power = signal_power / ((regsnr / 10) ** 10)

        noise = np.sqrt(noise_power) * (np.random.uniform(-1, 1, size=length))
        X += torch.from_numpy(noise.astype(np.float32)).unsqueeze(0)

        return X


class Sequencer:
    """Sequencer

    The sequencer generates an audio containing multiple samples of one species
    with radomly generated variations based on the samples provided.
    
    Attributes:
        name {str} -- sequencer label
        directory {str} -- path to the directory containing the audios
        size {Tuple[int, int]} -- number of sound range in the sequence 
        scale {Tuple[float, float]} -- volume factor scale range
        duration {float} -- sequence duration in seconds
        sr {int} -- sample rate
        files {int} -- audio files path
    """

    def __init__(
        self: "Sequencer",
        name: str,
        directory: str,
        size: Tuple[int, int],
        scale: Tuple[float, float],
        duration: float = 4,
        sr: int = 16000,
    ) -> None:
        """Initialization
        
        Arguments:
            name {str} -- sequencer label
            directory {str} -- path to the directory containing the audios
            size {Tuple[int, int]} -- number of sound range in the sequence 
            scale {Tuple[float, float]} -- volume factor scale range
        
        Keyword Arguments:
            duration {float} -- sequence duration in seconds (default: {4})
            sr {int} -- sample rate (default: {16000})
        
        Raises:
            IOError: raise exception if the directory does not exists
        """
        if not os.path.isdir(directory):
            raise IOError(f"The directory [{directory}] does not exists.")

        self.name = name
        self.directory = directory
        self.size = size
        self.scale = scale
        self.duration = duration
        self.sr = sr

        self.files = [
            os.path.join(directory, file)
            for file in sorted(os.listdir(directory))
            if file.endswith(".wav")
        ]

    def __iter__(self: "Sequencer") -> "Sequencer":
        """Iterator
        
        Returns:
            Sequencer -- iterator
        """
        return self

    def __next__(self: "Sequencer") -> torch.Tensor:
        """Next Item
        
        Returns:
            torch.Tensor -- next output tensor
        """
        duration = int(self.duration * self.sr)
        sequence = torch.zeros((1, duration)).float()

        for id in range(np.random.randint(*self.size)):
            file = np.random.choice(self.files)
            scale = np.random.uniform(*self.scale)
            position = np.random.randint(duration)

            sample, sr = torchaudio.load(file)
            sample = sample.mean(dim=0, keepdim=True)
            if sr != self.sr:
                sample = torchaudio.transforms.Resample(sr, self.sr)(sample)
            sample *= scale

            start = position
            end = min(position + sample.size(-1), duration - 1)
            sequence[:, start:end] += sample[:, : end - start]

        return sequence


class Ambient:
    """Ambient

    The ambient generates an audio containing ambient sound with randomly 
    generated variations using the samples provided.
    
    Attributes:
        name {str} -- sequencer label
        directory {str} -- path to the directory containing the audios
        size {Tuple[int, int]} -- number of sound range in the sequence 
        scale {Tuple[float, float]} -- volume factor scale range
        duration {float} -- sequence duration in seconds
        sr {int} -- sample rate
        files {int} -- audio files path
    """

    def __init__(
        self: "Ambient",
        name: str,
        directory: str,
        size: Tuple[int, int],
        scale: Tuple[float, float],
        duration: float = 4,
        sr: int = 16000,
    ) -> None:
        """Initialization
        
        Arguments:
            name {str} -- sequencer label
            directory {str} -- path to the directory containing the audios
            size {Tuple[int, int]} -- number of sound range in the sequence 
            scale {Tuple[float, float]} -- volume factor scale range
        
        Keyword Arguments:
            duration {float} -- sequence duration in seconds (default: {4})
            sr {int} -- sample rate (default: {16000})
        
        Raises:
            IOError: raise exception if the directory does not exists
        """
        if not os.path.isdir(directory):
            raise IOError(f"The directory [{directory}] does not exists.")

        self.name = name
        self.directory = directory
        self.size = size
        self.scale = scale
        self.duration = duration
        self.sr = sr

        self.files = [
            os.path.join(directory, file)
            for file in sorted(os.listdir(directory))
            if file.endswith(".wav")
        ]

    def __iter__(self: "Ambient") -> "Ambient":
        """Iterator
        
        Returns:
            Ambient -- iterator
        """
        return self

    def __next__(self: "Ambient") -> torch.Tensor:
        """Next Item
        
        Returns:
            torch.Tensor -- next output tensor
        """
        duration = int(self.duration * self.sr)
        sequence = torch.zeros((1, duration)).float()

        for id in range(np.random.randint(*self.size)):
            file = np.random.choice(self.files)
            scale = np.random.uniform(*self.scale)
            position = np.random.randint(duration)

            sample, sr = torchaudio.load(file)
            sample = sample.mean(dim=0, keepdim=True)
            if sr != self.sr:
                sample = torchaudio.transforms.Resample(sr, self.sr)(sample)
            sample *= scale
            length = sample.size(-1)

            start = np.random.randint(length - duration)
            end = min(start + duration, length)
            sequence += sample[:, start:end]

        return sequence


class Composer:
    """Composer

    The composer generates audio compositions using mulitple sequencers and
    ambients with control.
    
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
        focus {List[str]} -- specialize on spcific labels
    """

    def __init__(
        self: "Composer",
        label_directory: str,
        ambient_directory: str,
        label_size: Tuple[int, int],
        ambient_size: Tuple[int, int],
        label_scale: Tuple[float, float],
        ambient_scale: Tuple[float, float],
        duration: float = 4,
        sr: int = 16000,
        snr: Tuple[float, float] = (0, 100),
        focus: List[str] = None,
    ) -> None:
        """Initialization
        
        Arguments:
            label_directory {str} -- path to the sequencer audio directories
            ambient_directory {str} -- path to the ambients directories
            label_size {Tuple[int, int]} -- number of sound range in the 
                sequence
            ambient_size {Tuple[int, int]} -- number of sound range in the 
                ambient
            label_scale {Tuple[float, float]} -- volume factor scale range
            ambient_scale {Tuple[float, float]} -- volume factor scale range
            
        Keyword Arguments:
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

        self.noise = AdditiveWhiteGaussianNoise(snr)

        self.sequencers = OrderedDict(
            (
                label,
                Sequencer(
                    label,
                    os.path.join(label_directory, label),
                    label_size,
                    label_scale,
                    duration,
                    sr,
                ),
            )
            for label in sorted(os.listdir(label_directory))
            if os.path.isdir(os.path.join(label_directory, label))
        )

        self.ambients = OrderedDict(
            (
                label,
                Ambient(
                    label,
                    os.path.join(ambient_directory, label),
                    ambient_size,
                    ambient_scale,
                    duration,
                    sr,
                ),
            )
            for label in sorted(os.listdir(ambient_directory))
            if os.path.isdir(os.path.join(ambient_directory, label))
        )

    def _normalize(
        self: "Composer", X: torch.tensor, dim: int = 0
    ) -> torch.Tensor:
        """Normalize
        
        Arguments:
            X {torch.tensor} -- input tensor to be normalized
        
        Keyword Arguments:
            dim {int} -- axis to retrieve max (default: {0})
        
        Returns:
            torch.Tensor -- normalized tensor
        """
        X_max, _ = torch.max(torch.abs(X), dim=dim, keepdim=True)
        X /= X_max

        return X

    def __iter__(self: "Composer") -> "Composer":
        """Iterator
        
        Returns:
            Composer -- iterator
        """
        return self

    def __next__(self: "Composer") -> Tuple[torch.Tensor, torch.Tensor]:
        """Next Item
        
        Returns:
            Tuple[torch.Tensor, torch.Tensor] -- next composition and sequences
        """
        seq_keys = self.sequences.keys()
        sequences = torch.cat([next(self.sequencers[key]) for key in seq_keys])
        composition = sequences.mean(dim=0, keepdim=True)

        if self.focus is not None:
            labels = [label for label in self.focus if label in seq_keys]
            idxs = [seq_keys.index(label) for label in labels]
            sequences = sequences[idxs]

        amb_keys = self.ambients.keys()
        ambient = torch.cat([next(self.ambients[key]) for key in amb_keys])
        ambient = ambient.mean(dim=0, keepdim=True)

        composition += ambient
        composition = self.noise(composition)

        composition = self._normalize(composition, dim=1)
        sequences = self._normalize(composition, dim=1)

        return composition, sequences

