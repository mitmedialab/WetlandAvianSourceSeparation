"""demo.py

The file contains all necessary methods to perform demonstrations of the 
trained Conv-TasNet model using this library.
"""
import os
import torch
import torchaudio

from wass.convtasnet.model import Conv_TasNet


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
        for s, source in enumerate(sources):
            source_path = os.path.join(dest_path, f"source_{s}.wav")
            torchaudio.save(source_path, source, model.sr)
