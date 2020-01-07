# Wetland Avian Source Separation (WASS)

The repository is currently under **development**.

## Install

*Install directives.*

## Usage

*Usage directives.*

### Dataset

The Dataset is a simple class inheriting from `torch.util.data.Dataset`. It provide basic access to its length and item through a pytonic API. The class also provides direct ways to generate the dataset, either from a configuration file *(see Configuration section)* or from default values *(see code)*.

```python
from wass.dataset import CompositionDataset

# Generate dataset from config
train_dataset = CompositionDataset.generate_from_config(
    "dataset/train",
    config=config,
    size=20000
)

# Get length
print(len(train_dataset))   # >>> 20000

# Get datum
#   composition: [B, 1, S]
#       composition audio 
#   sequences: [B, C, S]
#       individual sequences from the composition
composition, sequences = train_dataset[0]
```

#### Data

The Dataset is generated out of bird and abmient samples. Those sample are 
available in the data folder and contains 10 bird species:

```
data/
|   ambient/
|   |   airplane/
|   |   rain/
|   |   wind/
|   birds/
|   |   blue_jay/
|   |   cardinal/
|   |   crow/
|   |   eastern_wood_pewee/
|   |   flicker/
|   |   green_heron/
|   |   purple_finch/
|   |   veery/
|   |   willow_flycatcher/
|   |   yellow_vireo/
```

#### Configuration

The generated compositions can be controlled with various parameters through a confirguration file saved as a `.yaml` file.

```python
from wass.audio.dataset import ComposerConfig

# Create configuration
config = ComposerConfig(
    "data/birds",
    "data/ambient",
    label_size=(0, 8),         # Range of samples for birds
    ambient_size=(0, 2),       # Range of samples for ambient
    label_scale=(0.5, 1.5),    # Range of volume scale for birds
    ambient_scale=(0.4, 1.4),  # Range of volume scale for ambient
    duration=4,                # Duration in second
    sr=16000,                  # Sample rate
    snr=(0, 100),              # Signal to noise ratio for Noise
)

# Save configuration
config.save("path.yaml")

# Retrieve from file
config = ComposerConfig.load("path.yaml)
```

### Training

*Training directives.*

### Inference

*Inference directives.*

## References

### Papers
- [Conv-TasNet] - TasNet: Surpassing Ideal Time-Frequency Masking for Speech Separation
- [SI SDR] - SDR – Half-Baked or Well Done?

### Implementations
- [Authors] - Github repository
- [Naplab] - Github repository
- [Jhuiac] - Github repository

[Conv-TasNet]: https://arxiv.org/abs/1809.07454
[SI SDR]: https://arxiv.org/pdf/1811.02508.pdf
[Authors]: https://github.com/kaituoxu/Conv-TasNet
[Naplab]: https://github.com/naplab/Conv-TasNet
[Jhuiac]: https://github.com/jhuiac/conv-tas-net

## Authors

- [Felix]
- [Yliess]
- [Clement]

[Felix]: https://github.com/FelixMichaud
[Yliess]: https://github.com/yliess86
[Clement]: https://github.com/slash6475