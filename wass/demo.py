"""demo.py

The file contains all necessary methods to perform demonstrations of the 
trained Conv-TasNet model using this library.
"""
import os
import torch
import torchaudio
import matplotlib.pyplot as plt

from wass.convtasnet.model import Conv_TasNet
from tqdm import tqdm


def normalize(X: torch.tensor, dim: int = 0) -> torch.Tensor:
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


def inference_demo(model_path: str, mixture_path: str, dest_path: str) -> None:
    """Inference Demonstration

    Performs source separation given a trained model on a single audio mixture
    file. The result is n (number of sources) audio wav files containing signal
    produced only by the sources respectively.
    
    Arguments:
        model_path {str} -- path to the trained model
        mixture_path {str} -- path to the mixture audio file
        dest_path {str} -- destination path to save the produced signals
    """
    if not os.path.isdir(dest_path):
        os.makedirs(dest_path, exist_ok=True)

    with torch.no_grad():
        model = Conv_TasNet.load(model_path)
        model.eval()

        mixture, sr = torchaudio.load(mixture_path)
        resampler = torchaudio.transforms.Resample(sr, model.sr)
        mixture = resampler(mixture)
        mixture = mixture.mean(dim=0, keepdim=True)
        mixture = mixture.unsqueeze(0)

        sources = model(mixture)
        sources = sources.detach().squeeze(0)
        sources = normalize(sources, dim=1)

        pbar = tqdm(sources, desc="Saving Sources", position=0, leave=True)
        for s, source in enumerate(pbar):
            source_path = os.path.join(dest_path, f"source_{s+1:02d}.wav")
            torchaudio.save(source_path, source, model.sr)

        n = len(sources)
        fig = plt.figure(figsize=(4, n * 2))
        pbar = tqdm(sources, desc="Plot Sources", position=0, leave=True)
        for s, source in enumerate(pbar):
            ax = fig.add_subplot(n, 1, s + 1)
            ax.plot(source)
        fig.tight_layout()
        fig.canvas.draw()

        fig_path = os.path.join(dest_path, "visualization.png")
        fig.savefig(fig_path)
