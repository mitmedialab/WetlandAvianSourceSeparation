"""loss.py

Th file contains an implementation of the Conv-TasNet Scale Invariant Signal to
Noise Ratio loss with PIT Training described in the TasNet paper 
(https://arxiv.org/pdf/1809.07454.pdf). The chosen loss (SI SNR) originates 
from the following paper:
    - https://arxiv.org/pdf/1811.02508.pdf

Code is heavily inspired by Kaituoxu implementation of the paper:
    - https://github.com/kaituoxu/Conv-TasNet/blob/master/src/pit_criterion.py
"""
import torch
import torch.nn as nn

from itertools import permutations


class SI_SNR(nn.Module):
    """Scale Invariant Signal to Noise Ratio with support for PIT Training
    
    Adapted from:
        - https://github.com/kaituoxu/Conv-TasNet

    Attributes:
        eps {float} -- epsilon to avoid 0 division
        pit {bool} -- use pit training https://arxiv.org/abs/1607.00325
    """

    def __init__(self: "SI_SNR", eps: float = 1e-8, pit: bool = False) -> None:
        """Initialization

        Keyword Arguments:
            eps {float} -- epsilon to avoid 0 division (default: {1e-8})
            pit {bool} -- use pit training (default: {False})
        """
        super(SI_SNR, self).__init__()
        self.eps = eps
        self.pit = pit

    def forward(
        self: "SI_SNR", Y_: torch.Tensor, Y: torch.Tensor
    ) -> torch.Tensor:
        """Forward Pass

        zero-mean prior to calculation:
            s_target = (<ŝ,s>.s) / ||s||²
            e_noise = ŝ - s_target
            si_snr = 10 torch.log10(||s_target||² / ||e_noise||²)
        
        Arguments:
            Y_ {torch.Tensor} -- estimated source separation input tensor
            Y {torch.Tensor} -- target source separation input tensor
        
        Returns:
            torch.Tensor -- si snr output loss tensor
        """
        B, C, S = Y.size()

        zero_mean_target = Y - torch.mean(Y, dim=-1, keepdim=True)
        zero_mean_estimate = Y_ - torch.mean(Y_, dim=-1, keepdim=True)

        s_target = torch.unsqueeze(zero_mean_target, dim=1)
        s_estimate = torch.unsqueeze(zero_mean_estimate, dim=2)

        pair_wise_dot = torch.sum(s_estimate * s_target, dim=3, keepdim=True)
        s_target_energy = (
            torch.sum(s_target ** 2, dim=3, keepdim=True) + self.eps
        )
        pair_wise_proj = pair_wise_dot * s_target / s_target_energy

        e_noise = s_estimate - pair_wise_proj

        pair_wise_si_snr = torch.sum(pair_wise_proj ** 2, dim=3) / (
            torch.sum(e_noise ** 2, dim=3) + self.eps
        )
        pair_wise_si_snr = 10 * torch.log10(pair_wise_si_snr + self.eps)

        if self.pit:
            perms = Y.new_tensor(list(permutations(range(C)))).long()
            index = torch.unsqueeze(perms, 2)
            perms_one_hot = Y.new_zeros((*perms.size(), C)).scatter_(
                2, index, 1
            )
            snr_set = torch.einsum(
                "bij,pij->bp", [pair_wise_si_snr, perms_one_hot]
            )

            max_snr_idx = torch.argmax(snr_set, dim=1)
            max_snr, _ = torch.max(snr_set, dim=1)
            max_snr /= C

        si_snr = max_snr if self.pit else pair_wise_si_snr
        loss = 0 - torch.mean(si_snr)

        return loss


if __name__ == "__main__":
    B, C, S = 2, 3, 12
    print(f"B: {B}, C: {C}, S: {S}")

    source = torch.randint(4, (B, C, S)).float()
    estimate = torch.randint(4, (B, C, S)).float()
    print(f"source: {tuple(source.shape)}")
    print(f"estimate: {tuple(estimate.shape)}")

    loss = SI_SNR()(estimate, source).detach().cpu().item()
    print(f"loss: {loss}")

