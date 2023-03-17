import math

import torch.nn.functional as F
from torch import nn
from torchaudio.transforms import MelSpectrogram


def res_block(
    in_channels, out_channels, kernel_size=3, stride=1, padding=1, dilation=1
):
    return nn.Sequential(
        nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            dilation=dilation,
            padding=padding,
        ),
        nn.SiLU(),
        nn.BatchNorm2d(out_channels),
        nn.Conv2d(
            out_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            dilation=dilation,
            padding=padding,
        ),
    )


class FishPitchPredictor(nn.Module):
    def __init__(
        self,
        mel_channels=128,
        residual_channels=128,
        residual_layers=10,
        n_fft=1024,
        hop_length=256,
        win_length=1024,
        f_min=0,
        f_max=8000,
    ):
        super().__init__()

        self.input_projection = res_block(
            1, residual_channels, kernel_size=1, padding=0
        )

        self.residual_layers = nn.ModuleList(
            [
                res_block(
                    residual_channels,
                    residual_channels,
                    kernel_size=3,
                    padding=2 ** (i % 4),
                    dilation=2 ** (i % 4),
                )
                for i in range(residual_layers)
            ]
        )

        self.output_projection = nn.Sequential(
            res_block(residual_channels, residual_channels, kernel_size=1, padding=0),
            nn.Conv2d(residual_channels, 1, kernel_size=1, padding=0),
            nn.ReLU(),
        )
        
    def forward(self, mel, mask=None):
        mel = mel[:, None, :, :]

        if mask is not None and mask.dim() == 2:
            mask = mask[:, None, None, :]

        x = self.input_projection(mel)  # x [B, residual_channel, T]

        if mask is not None:
            x = x * mask

        for layer in self.residual_layers:
            prev = x
            x = F.silu(layer(x) + prev) / math.sqrt(2)

            if mask is not None:
                x = x * mask

        x = self.output_projection(x)

        if mask is not None:
            x = x * mask

        return x[:, 0, 0, :]
