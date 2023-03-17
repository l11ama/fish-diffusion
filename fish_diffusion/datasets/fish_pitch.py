import random

import librosa
import numpy as np
import torch
from audiomentations import (
    AddBackgroundNoise,
    AddGaussianSNR,
    AirAbsorption,
    ApplyImpulseResponse,
    Compose,
    Mp3Compression,
)
from torchaudio.transforms import MelSpectrogram
from fish_diffusion.utils.audio import dynamic_range_compression

from fish_diffusion.utils.tensor import repeat_expand

from .builder import DATASETS
from .naive import NaiveDataset


@DATASETS.register_module()
class FishPitchPredictorDataset(NaiveDataset):
    processing_pipeline = [
        dict(type="PickKeys", keys=["path", "audio", "pitches"]),
        dict(type="UnSqueeze", keys=[("audio", 0)]),  # (T) -> (1, T)
    ]

    collating_pipeline = [
        dict(type="ListToDict"),
        dict(type="PadStack", keys=[("audio", -1), ("mel", -1), ("pitches", -1)]),
    ]

    def __init__(
        self,
        path="dataset",
        sampling_rate=16000,
        n_fft=1024,
        hop_length=256,
        win_length=1024,
        mel_channels=128,
        f_min=0,
        f_max=8000,
        augmentation=True,
        segment_size=10,
    ):
        super().__init__(path)

        self.sampling_rate = sampling_rate
        self.segment_size = segment_size
        self.hop_length = hop_length

        if augmentation:
            self.augment = Compose(
                [
                    AddGaussianSNR(),
                    AirAbsorption(),
                    Mp3Compression(),
                ]
            )

        self.spec = MelSpectrogram(
            n_fft=n_fft,
            hop_length=hop_length,
            win_length=win_length,
            n_mels=mel_channels,
            f_min=f_min,
            f_max=f_max,
            center=True,
            power=1.0,
            pad_mode="reflect",
            norm="slaney",
            mel_scale="slaney",
        )

    def __getitem__(self, idx):
        x = super().__getitem__(idx)

        # process audio
        audio = x["audio"]
        pitches = x["pitches"]

        if hasattr(self, "augment"):
            audio = self.augment(samples=audio, sample_rate=self.sampling_rate)

            if random.random() > 0.5:
                pitch_shift = random.randint(-4, 4)
                audio = librosa.effects.pitch_shift(
                    audio, sr=self.sampling_rate, n_steps=pitch_shift
                )
                pitches *= 2 ** (pitch_shift / 12)
        
        estimated_mel_length = audio.shape[1] // self.hop_length + 1
        pitches = repeat_expand(pitches, estimated_mel_length)

        # Randomly crop the audio and mel
        if (
            self.segment_size is not None
            and self.segment_size > 0
            and audio.shape[1] - self.segment_size + 1 > 0
        ):
            start = np.random.randint(0, audio.shape[1] - self.segment_size + 1)
            audio = audio[:, start : start + self.segment_size]
            pitches = pitches[
                start // self.hop_length : (start + self.segment_size) // self.hop_length
            ]
        
        # Get mel spectrogram after augmentation
        mel = dynamic_range_compression(self.spec(torch.from_numpy(audio))).squeeze(0).numpy()
        pitches = repeat_expand(pitches, mel.shape[-1])

        x["mel"] = mel
        x["audio"] = audio
        x["pitches"] = pitches

        return x
