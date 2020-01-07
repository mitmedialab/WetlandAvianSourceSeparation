"""loss.py

Th file contains an implementation of the Conv-TasNet Scale Invariant Signal to
Noise Ratio loss described in the TasNet paper 
(https://arxiv.org/pdf/1809.07454.pdf). The chosen loss (SI SNR) originates 
from the following paper:
    - https://arxiv.org/pdf/1811.02508.pdf

Code is heavily inspired by the original implementation from the paper's 
authors:
    - https://github.com/naplab/Conv-TasNet
"""
import torch
import torch.nn as nn

from itertools import permutations


class SI_SNR_OneAudio(nn.Module):
    """Scale Invariant Signal to Noise Ratio for One Audio

    SI SNR originate from:
        - https://arxiv.org/pdf/1811.02508.pdf

    The implementation is a straight adaptation from the original Conv-TasNet
    github repository:
        - https://github.com/kaituoxu/Conv-TasNet
    
    Attributes:
        eps {float} -- epsilon value to avoid 0 division
    """

    def __init__(self: "SI_SNR_OneAudio", eps: float = 1e-8) -> None:
        """Initialization
        
        Keyword Arguments:
            eps {float} -- epsilon value to avoid 0 division (default: {1e-8})
        """
        super(SI_SNR_OneAudio, self).__init__()
        self.eps = eps

    def forward(
        self: "SI_SNR_OneAudio",
        Y_: torch.Tensor,
        Y: torch.Tensor,
        mask: torch.Tensor = None,
    ) -> torch.Tensor:
        """Forward Pass
        
        Arguments:
            Y_ {torch.Tensor} -- estimated input tensor
            Y {torch.Tensor} -- target input tensor
        
        Keyword Arguments:
            mask {torch.Tensor} -- mask tensor if any (default: {None})
        
        Returns:
            torch.Tensor -- si snr of one audio output tensor
        """
        if mask is not None:
            Y_ *= mask
            Y *= mask

        origin_power = torch.pow(Y, 2).sum(1, keepdim=True) + self.eps
        scale = torch.sum(Y * Y_, 1, keepdim=True) / origin_power

        scaled_reference = Y * scale
        scaled_restitution = Y_ - scaled_reference

        power_reference = torch.pow(scaled_reference, 2).sum(1)
        power_restitution = torch.pow(scaled_restitution, 2).sum(1)

        cost = 10 * (
            torch.log10(power_reference) - torch.log10(power_restitution)
        )

        return cost


class SI_SNR_MultiAudio(nn.Module):
    """Scale Invariant Signal to Noise Ratio for Multiple Audios

    SI SNR originate from:
        - https://arxiv.org/pdf/1811.02508.pdf

    The implementation is a straight adaptation from the original Conv-TasNet
    github repository:
        - https://github.com/kaituoxu/Conv-TasNet
    
    Attributes:
        eps {float} -- epsilon value to avoid 0 division
        si_snr {SI_SNR_OneAudio} -- SI SNR for one audio
    """

    def __init__(self: "SI_SNR_MultiAudio", eps: float = 1e-8) -> None:
        """Initialization
        
        Keyword Arguments:
            eps {float} -- epsilon value to avoid 0 division (default: {1e-8})
        """
        super(SI_SNR_MultiAudio, self).__init__()
        self.eps = eps
        self.si_snr = SI_SNR_OneAudio(eps)

    def forward(
        self: "SI_SNR_MultiAudio",
        Y_: torch.Tensor,
        Y: torch.Tensor,
        mask: torch.Tensor = None,
    ) -> torch.Tensor:
        """Forward Pass
        
        Arguments:
            Y_ {torch.Tensor} -- estimated input tensor
            Y {torch.Tensor} -- target input tensor
        
        Keyword Arguments:
            mask {torch.Tensor} -- mask tensor if any (default: {None})
        
        Returns:
            torch.Tensor -- si snr of multiple audios output tensor
        """
        B, C, _ = Y.size()
        _type = Y.type()

        Y -= torch.mean(Y, 2, keepdim=True).expand_as(Y)
        Y_ -= torch.mean(Y_, 2, keepdim=True).expand_as(Y_)

        perms = list(set(permutations(torch.arange(C))))
        n_perms = len(perms)

        pairwise_SI_SNR = torch.zeros((B, C, C)).type(_type)
        for i in range(C):
            for j in range(C):
                pairwise_SI_SNR[i, j] = self.si_snr(Y_[:, i], Y[:, j], mask)

        _SI_SNR_perms = []
        for perm in perms:
            si_snr = [pairwise_SI_SNR[:, p, perm].view(B, -1) for p in n_perms]
            si_snr = torch.sum(torch.cat(si_snr, 1), 1).view(B, 1)
            SI_SNR_perms.append(si_snr)
        SI_SNR_perms = torch.cat(SI_SNR_perms, 1)
        SI_SNR_max, _ = torch.max(SI_SNR_perms, dim=1)

        SI_SNR = SI_SNR_max / C

        return SI_SNR


class SI_SNR_Criterion(nn.Module):
    """Scale Invariant Signal to Noise Ratio Criterion

    SI SNR originate from:
        - https://arxiv.org/pdf/1811.02508.pdf

    The implementation is a straight adaptation from the original Conv-TasNet
    github repository:
        - https://github.com/kaituoxu/Conv-TasNet
    
    Attributes:
        eps {float} -- epsilon value to avoid 0 division
        si_snr {SI_SNR_OneAudio} -- SI SNR for multiple audios
    """

    def __init__(
        self: "SI_SNR_Criterion", eps: float = 1e-8
    ) -> "SI_SNR_Criterion":
        """Initialization
           
        Keyword Arguments:
            eps {float} -- epsilon value to avoid 0 division (default: {1e-8})
        """
        super(SI_SNR_Criterion, self).__init__()
        self.eps = eps
        self.si_snr = SI_SNR_MultiAudio(eps)

    def forward(
        self: "SI_SNR_Criterion", Y_: torch.Tensor, Y: torch.Tensor
    ) -> torch.Tensor:
        """Forward Pass
        
        Arguments:
            Y_ {torch.Tensor} -- estimated input tensor
            Y {torch.Tensor} -- target input tensor
        
        Keyword Arguments:
            mask {torch.Tensor} -- mask tensor if any (default: {None})
        
        Returns:
            torch.Tensor -- negative si snr output loss
        """
        return -self.si_snr(Y_, Y)
