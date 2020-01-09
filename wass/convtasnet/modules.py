"""modules.py

Th file contains all the modules used to compose the final Conv-TasNEt model
described in paper:
    - https://arxiv.org/pdf/1809.07454.pdf

Code is heavily inspired by the original implementation from the paper's 
authors which can be found here:
    - https://github.com/naplab/Conv-TasNet
"""
import torch
import torch.nn as nn

from typing import Tuple


class CumulativeLayerNorm(nn.Module):
    """Cumulative Layer Normalization (cLN)

    Implementation of the cLN as described in Conv-TasNet paper:
        - https://arxiv.org/pdf/1809.07454.pdf

    Attributes:
        eps {float} -- epsilon value to avoid 0 division (default: {1e-8})
        gain {torch.Tensor} -- gain weights (gamma)
        bias {torch.Tensor} -- bias weights (beta)
    """

    def __init__(
        self: "CumulativeLayerNorm", dimension: int, eps: float = 1e-8
    ) -> None:
        """Initialization

        Arguments:
            dimension {int} -- layer size

        Keyword Arguments:
            eps {float} -- epsilon value to avoid 0 division (default: {1e-8})
        """
        super(CumulativeLayerNorm, self).__init__()
        self.eps = eps
        self.gain = nn.Parameter(torch.ones(1, dimension, 1))
        self.bias = nn.Parameter(torch.zeros(1, dimension, 1))

    def forward(self: "CumulativeLayerNorm", X: torch.Tensor) -> torch.Tensor:
        """Forward Pass

        .. math::
            cLN(f_{k}) = 
            \frac{f_{k} - E[f_{t \leq k}]}{\sqrt{Var[f_{t \leq k}] + \epsilon}}
            \odot \gamma + \beta
        
        Arguments:
            X {torch.Tensor} -- input tensor
        
        Returns:
            torch.Tensor -- cLN output tensor
        """
        mean = torch.mean(X, dim=1, keepdim=True)
        var = torch.var(X, dim=1, keepdim=True, unbiased=False)
        std = torch.sqrt(var + self.eps)

        X = self.gain * (X - mean) / std + self.bias

        return X


class DepthWiseConv1DBLock(nn.Module):
    """Depth-Wise 1 Dimensionnal Convolution Block
        
    Attributes:
        causal {bool} -- choose cLN instead of gLN (default: {False})
        padding {int} -- padding size
        conv1d {nn.Module} -- 1 dimensionnal convolution layer
        dconv1d {nn.Module} -- 1 dimensionnal depth-wise convolution layer
        res {nn.Module} -- 1 dimensionnal resiudal convolution layer
        prelu1 {nn.Module} -- activation function PReLU
        prelu2 {nn.Module} -- activation function PReLU
        regul1 {nn.Module} -- cLN or gLN layer
        regul2 {nn.Module} -- cLN or gLN layer
        skip {nn.Module} -- 1 dimensionnal skip connection convolution layer
    """

    def __init__(
        self: "DepthWiseConv1DBLock",
        in_chan: int,
        hidden_chan: int,
        kernel: int,
        padding: int,
        dilation: int = 1,
        causal: bool = False,
    ) -> None:
        """Initialization
        
        Arguments:
            in_chan {int} -- input channel size
            hidden_chan {int} -- hidden channel size
            kernel {int} -- kernel size
            padding {int} -- padding size
        
        Keyword Arguments:
            dilation {int} -- dilation factor for altrous conv (default: {1})
            causal {bool} -- choose cLN instead of gLN (default: {False})
        """
        super(DepthWiseConv1DBLock, self).__init__()
        self.causal = causal
        self.padding = (kernel - 1) * dilation if causal else padding

        self.conv1d = nn.Conv1d(in_chan, hidden_chan, 1)
        self.dconv1d = nn.Conv1d(
            hidden_chan,
            hidden_chan,
            kernel,
            dilation=dilation,
            groups=hidden_chan,
            padding=self.padding,
        )
        self.res = nn.Conv1d(hidden_chan, in_chan, 1)

        self.prelu1 = nn.PReLU()
        self.prelu2 = nn.PReLU()

        self.regul1 = (
            CumulativeLayerNorm(hidden_chan, eps=1e-08)
            if self.causal
            else nn.GroupNorm(1, hidden_chan, eps=1e-08)
        )
        self.regul2 = (
            CumulativeLayerNorm(hidden_chan, eps=1e-08)
            if self.causal
            else nn.GroupNorm(1, hidden_chan, eps=1e-08)
        )

        self.skip = nn.Conv1d(hidden_chan, in_chan, 1)

    def forward(
        self: "DepthWiseConv1DBlock", X: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forwad Pass
        
        Arguments:
            X {torch.Tensor} -- mixture input tensor
        
        Returns:
            Tuple[torch.Tensor, torch.Tensor] -- residual and mixture output 
                tensors from the depth-wise conv block
        """
        X = self.regul1(self.prelu1(self.conv1d(X)))
        X = (
            self.regul2(self.prelu2(self.dconv1d(X)[:, :, : -self.padding]))
            if self.causal
            else self.regul2(self.prelu2(self.dconv1d(X)))
        )

        residual = self.res(X)
        X = self.skip(X)

        return residual, X


class Encoder(nn.Module):
    """Encoder
    
    Attributes:
        conv1d {nn.Module} -- 1 dimensionnal convolution layer
    """

    def __init__(
        self: "Encoder", encoder_dim: int, win_size: int, stride: int
    ) -> None:
        """Initialization
        
        Arguments:
            encoder_dim {int} -- encoder dimension
            win_size {int} -- window size
            stride {int} -- stride size
        """
        super(Encoder, self).__init__()
        self.conv1d = nn.Conv1d(
            1, encoder_dim, win_size, bias=False, stride=stride
        )

    def forward(self: "Encoder", X: torch.Tensor) -> torch.Tensor:
        """Forward Pass
        
        Arguments:
            X {torch.Tensor} -- input tensor
        
        Returns:
            torch.Tensor -- output tensor
        """
        return self.conv1d(X)


class Separator(nn.Module):
    """Separator Module
    
    Attributes:
        layer_norm {nn.Module} -- normalization layer either cLN or gLN
        batch_norm {nn.Module} -- 1 dimensionnal convolution layer
        receptive_field {int} -- receptive field of the module
        dilated {bool} -- is there dilation in the module conv layers ?
        TCN {nn.ModuleList} -- TCN module list (depth-wise conv blocks)
        out {nn.Module} -- output module
    """

    def __init__(
        self: "Separator",
        in_dim: int,
        out_dim: int,
        bn_dim: int,
        hidden_dim: int,
        layers: int,
        stack: int,
        kernel: int = 3,
        causal: bool = False,
        dilated: bool = True,
    ) -> None:
        """Initialization
        
        Arguments:
            in_dim {int} -- input dimension
            out_dim {int} -- outpit dimension
            bn_dim {int} -- batch norm dimension
            hidden_dim {int} -- hidden dimension
            layers {int} -- number of layers per stack
            stack {int} -- number of stackes layers block
        
        Keyword Arguments:
            kernel {int} -- kernel size (default: {3})
            causal {bool} -- is the module causal ? (default: {False})
            dilated {bool} -- is there dilation ? (default: {True})
        
        Returns:
            [type] -- [description]
        """
        super(Separator, self).__init__()
        self.layer_norm = (
            CumulativeLayerNorm(in_dim, eps=1e-8)
            if causal
            else nn.GroupNorm(1, in_dim, eps=1e-8)
        )
        self.batch_norm = nn.Conv1d(in_dim, bn_dim, 1)

        self.receptive_field = 0
        self.dilated = dilated

        self.TCN = nn.ModuleList([])
        for s in range(stack):
            for layer_id in range(layers):
                dilation = 2 ** layer_id if self.dilated else 1
                self.TCN.append(
                    DepthWiseConv1DBLock(
                        bn_dim,
                        hidden_dim,
                        kernel,
                        dilation=dilation,
                        padding=dilation,
                        causal=causal,
                    )
                )
                self.receptive_field = self.receptive_field + (
                    kernel
                    if layer_id == 0 and s == 0
                    else (kernel - 1) * dilation
                )

        self.out = nn.Sequential(nn.PReLU(), nn.Conv1d(bn_dim, out_dim, 1))

    def forward(self: "Separator", X: torch.Tensor) -> torch.Tensor:
        """Forward Pass
        
        Arguments:
            X {torch.Tensor} -- encoder output as input tensor
        
        Returns:
            torch.Tensor -- output mixture tensor to be fed to the decoder
        """
        X = self.batch_norm(self.layer_norm(X))

        skip_connection = 0.0
        for module in self.TCN:
            residual, skip = module(X)
            X = X + residual
            skip_connection = skip_connection + skip
        X = self.out(skip_connection)

        return X


class Decoder(nn.Module):
    """Decoder
    
    Attributes:
        conv1d {nn.Module} -- 1 dimensionnal convolution layer
    """

    def __init__(
        self: "Decoder", decoder_dim: int, win_size: int, stride: int
    ) -> None:
        """Initialization
        
        Arguments:
            decoder_dim {int} -- decoder dimension
            win_size {int} -- window size
            stride {int} -- stride size
        """
        super(Decoder, self).__init__()
        self.conv1d = nn.ConvTranspose1d(
            decoder_dim, 1, win_size, bias=False, stride=stride
        )

    def forward(self: "Decoder", X: torch.Tensor) -> torch.Tensor:
        """Forward Pass
        
        Arguments:
            X {torch.Tensor} -- input tensor
        
        Returns:
            torch.Tensor -- output tensor
        """
        return self.conv1d(X)
