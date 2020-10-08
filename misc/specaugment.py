import argparse
import librosa
import numpy as np
import torch
from SpecAugment import spec_augment_pytorch
import os, sys

audio_path = '/Users/ethanzh/Code/automatic-speaker-recognition/audio/original/SherlockHolmes/ethan.wav'

# Step 0 : load audio file, extract mel spectrogram
audio, sampling_rate = librosa.load(audio_path)
mel_spectrogram = librosa.feature.melspectrogram(y=audio,
                                                    sr=sampling_rate,
                                                    n_mels=256,
                                                    hop_length=128,
                                                    fmax=8000)

# reshape spectrogram shape to [batch_size, time, frequency]
shape = mel_spectrogram.shape
#mel_spectrogram = np.reshape(mel_spectrogram, (-1, shape[0], shape[1]))
mel_spectrogram = torch.from_numpy(mel_spectrogram)

# Show Raw mel-spectrogram
spec_augment_pytorch.visualization_spectrogram(mel_spectrogram=mel_spectrogram, title="Raw Mel Spectrogram")

# Calculate SpecAugment pytorch
warped_masked_spectrogram = spec_augment_pytorch.spec_augment(mel_spectrogram=mel_spectrogram)

# Show time warped & masked spectrogram
spec_augment_pytorch.visualization_spectrogram(mel_spectrogram=warped_masked_spectrogram, title="pytorch Warped & Masked Mel Spectrogram")