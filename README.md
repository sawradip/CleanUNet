# cleanunet: A Python Package for Speech Denoising

## Introduction

`cleanunet` is a Python package that provides an easy-to-use interface for speech denoising, based on the official PyTorch implementation of CleanUNet. This package allows users to perform high-quality speech denoising with minimal setup, leveraging the power of CleanUNet directly from PyPI.

CleanUNet is a state-of-the-art model for causal speech denoising on raw waveforms, utilizing an encoder-decoder architecture with self-attention blocks. It was originally developed by Kong, Zhifeng; Ping, Wei; Dantrey, Ambrish; and Catanzaro, Bryan. Full details can be found in their paper: [Speech Denoising in the Waveform Domain with Self-Attention](https://arxiv.org/abs/2202.07790).

## Installation

To install `cleanunet`, simply use pip:

```bash
pip install cleanunet
```

## Usage

`cleanunet` can be used to denoise audio files easily. Here's a basic example:

```python
import torchaudio
from cleanunet import CleanUNet

# Load an audio file
aud, sr = torchaudio.load("path_to_your_audio_file.wav")

# Initialize CleanUNet
net = CleanUNet.from_pretrained(variant='full') # or 'high'

# Perform denoising
denoised_aud = net(aud)[0]

# Save the denoised audio
torchaudio.save('path_to_denoised_audio_file.wav', denoised_aud, sr)
```

To utilize CUDA for faster processing:

```python
aud, sr = torchaudio.load("path_to_your_audio_file.wav")
net = CleanUNet.from_pretrained(variant='full', device='cuda')
denoised_aud = net(aud.to('cuda'))[0]
```

## Credits and References

This package is a pip-installable version of the CleanUNet model, as described in the paper by Kong et al. The original implementation and more comprehensive details can be found in their [GitHub repository](https://github.com/nv-adlr/CleanUNet).

The structure and distributed training are adapted from [WaveGlow (PyTorch)](https://github.com/NVIDIA/waveglow), and other components are adapted from various sources as detailed in the original CleanUNet repository.

## Citation

If you use `cleanunet` or CleanUNet in your research, please cite the original paper:

```
@inproceedings{kong2022speech,
  title={Speech Denoising in the Waveform Domain with Self-Attention},
  author={Kong, Zhifeng and Ping, Wei and Dantrey, Ambrish and Catanzaro, Bryan},
  booktitle={ICASSP 2022-2022 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)},
  pages={7867--7871},
  year={2022},
  organization={IEEE}
}
```

## License

This package is distributed under the same license as the original CleanUNet implementation. Please refer to the [original repository](https://github.com/nv-adlr/CleanUNet) for license details.
