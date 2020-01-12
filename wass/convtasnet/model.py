"""model.py

Th file contains an implementation of the Conv-TasNet model described in paper:
    - https://arxiv.org/pdf/1809.07454.pdf

Code is heavily inspired by the original implementation from the paper's 
authors and others which can be found here:
    - https://github.com/naplab/Conv-TasNet
    - https://github.com/kaituoxu/Conv-TasNet
"""
import torch
import torch.nn as nn

from wass.convtasnet.modules import Encoder, Separator, Decoder
from typing import Tuple, Dict, Any
from torch.autograd import Variable
from torch.optim import Optimizer


class Conv_TasNet(nn.Module):
    """Conv-TasNet Model

    Conv-TasNet Model for Waveform Source Separation (SS):
        - https://arxiv.org/pdf/1809.07454.pdf
    
    Attributes:
        n_sources {int} -- number of sources to seperate
        encoder_dim {int} -- encoder dimension
        feature_dim {int} -- feature dimension
        sr {int} -- sample rate
        win_length {float} -- window lenght in seconds
        win_size {int} -- window size
        stride {int} -- stride size
        layers {int} -- number of layers for each block
        stack {int} -- number of stacked blocks
        kernel {int} -- kernel size
        causal {bool} -- is the model causal ? 
        encoder {nn.Module} -- encoder module
        separator {nn.Module} -- separator module
        receptive_field {int} -- receptive field of the separator model
        decoder {nn.Module} -- decode module
    """

    def __init__(
        self: "Conv_TasNet",
        encoder_dim: int = 512,
        feature_dim: int = 128,
        sr: int = 16000,
        win_length: int = 2,
        layers: int = 8,
        stack: int = 3,
        kernel: int = 3,
        n_sources: int = 2,
        causal: bool = False,
    ) -> None:
        """Initialization
        
        Keyword Arguments:
            encoder_dim {int} -- encoder dimension (default: {512})
            feature_dim {int} -- feature dimension (default: {128})
            sr {int} -- sample rate (default: {16000})
            win_length {int} -- window length in seconds (default: {2})
            layers {int} -- number of layers for ach block (default: {8})
            stack {int} -- numbr of stacked blocks (default: {3})
            kernel {int} -- kernel size (default: {3})
            n_sources {int} -- number of sources to separate (default: {2})
            causal {bool} -- is the model causal ? (default: {False})
        """
        super(Conv_TasNet, self).__init__()
        self.n_sources = n_sources
        self.encoder_dim = encoder_dim
        self.feature_dim = feature_dim
        self.sr = sr
        self.win_length = win_length
        self.win_size = int(sr * win_length / 1000)
        self.stride = self.win_size // 2
        self.layers = layers
        self.stack = stack
        self.kernel = kernel
        self.causal = causal

        self.encoder = Encoder(encoder_dim, self.win_size, self.stride)
        self.separator = Separator(
            encoder_dim,
            encoder_dim * n_sources,
            feature_dim,
            feature_dim * 4,
            layers,
            stack,
            causal=causal,
        )
        self.receptive_field = self.separator.receptive_field
        self.decoder = Decoder(encoder_dim, self.win_size, self.stride)

        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_normal_(p)

    def pad(self: "Conv_TasNet", X: torch.Tensor) -> Tuple[torch.Tensor, int]:
        """Padd Input Tensor
        
        Arguments:
            X {torch.Tensor} -- input tensor to be padded
        
        Raises:
            RuntimeError: raise exception when input is not of dimension 2 or 3
        
        Returns:
            Tuple[torch.Tensor, int] -- padded tensor and padded amount
        """
        if X.dim() not in [2, 3]:
            raise RuntimeError("Input must contain either 2 ou 3 dimensions.")

        if X.dim() == 2:
            X = torch.unsqueeze(X, dim=1)

        B, _, S = X.size()
        _type = X.type()
        rest = (
            self.win_size - (self.stride + S % self.win_size) % self.win_size
        )

        if rest > 0:
            pad = Variable(torch.zeros(B, 1, rest)).type(_type)
            X = torch.cat([X, pad], 2)

        pad = Variable(torch.zeros(B, 1, self.stride)).type(_type)
        X = torch.cat([pad, X, pad], 2)

        return X, rest

    def forward(self: "Conv_TasNet", X: torch.Tensor) -> torch.Tensor:
        """Forward Pass
        
        Arguments:
            X {torch.Tensor} -- audio input tensor
        
        Returns:
            torch.Tensor -- separated audio output tensor
        """
        X, rest = self.pad(X)
        B, _, _ = X.size()

        mixture = self.encoder(X)

        masks = torch.sigmoid(self.separator(mixture)).view(
            B, self.n_sources, self.encoder_dim, -1
        )
        mixture = torch.unsqueeze(mixture, dim=1) * masks

        ss = self.decoder(
            mixture.view(B * self.n_sources, self.encoder_dim, -1)
        )
        ss = ss[:, :, self.stride : -(rest + self.stride)].contiguous()
        ss = ss.view(B, self.n_sources, -1)

        return ss

    @staticmethod
    def serialize(
        model: "Conv_TasNet", optimizer: Optimizer
    ) -> Dict[str, Any]:
        """Serialize Model to be Saved
        
        Arguments:
            model {Conv_TasNet} -- model to be serialized
            optimizer {Optimizer} -- optimizer to be serialized
        
        Returns:
            Dict[str, Any] -- package containing the training infos
                config:
                    - encoder_dim
                    - feature_dim
                    - sr
                    - win_length
                    - layers
                    - stack
                    - kernel
                    - n_sources
                    - causal
                state:
                    - state_dict
                    - optim_dict
        """
        config = {
            "encoder_dim": model.encoder_dim,
            "feature_dim": model.feature_dim,
            "sr": model.sr,
            "win_length": model.win_length,
            "layers": model.layers,
            "stack": model.stack,
            "kernel": model.kernel,
            "n_sources": model.n_sources,
            "causal": model.causal,
        }

        state = {
            "state_dict": model.state_dict(),
            "optim_dict": optimizer.state_dict(),
        }

        package = {"config": config, "state": state}

        return package

    @classmethod
    def load(cls: "Conv_TasNet", path: str) -> "Conv_TasNet":
        """Load Model from File
        
        Arguments:
            path {str} -- path to the model's checkpoint

        Returns:
            Conv_Tas -- loaded model
        """
        package = torch.load(path, map_location=lambda stor, loc: stor)
        model = cls(**package["config"])
        model.load_state_dict(package["state"]["state_dict"])

        return model


if __name__ == "__main__":
    B, C, S = 2, 3, 12
    print(f"B: {B}, C: {C}, S: {S}")

    mixture = torch.randint(4, (B, 1, S)).float()
    print(f"mixture: {tuple(mixture.shape)}")

    estimate = Conv_TasNet(n_sources=C)(mixture)
    print(f"estimate: {tuple(estimate.shape)}")
