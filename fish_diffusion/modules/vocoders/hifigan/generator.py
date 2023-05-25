from math import prod

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import remove_weight_norm, weight_norm

from fish_diffusion.modules.vocoders.nsf_hifigan.models import ResBlock1, init_weights


class HiFiGANGenerator(nn.Module):
    def __init__(
        self,
        *,
        hop_length: int = 512,
        upsample_rates: tuple[int] = (8, 8, 2, 2, 2),
        upsample_kernel_sizes: tuple[int] = (16, 16, 8, 2, 2),
        resblock_kernel_sizes: tuple[int] = (3, 7, 11),
        resblock_dilation_sizes: tuple[tuple[int]] = ((1, 3, 5), (1, 3, 5), (1, 3, 5)),
        leaky_relu_slope: float = 0.2,
        num_mels: int = 128,
        upsample_initial_channel: int = 512,
    ):
        super().__init__()

        assert (
            prod(upsample_rates) == hop_length
        ), f"hop_length must be {prod(upsample_rates)}"

        self.conv_pre = weight_norm(
            nn.Conv1d(num_mels, upsample_initial_channel, 7, 1, padding=3)
        )
        self.leaky_relu_slope = leaky_relu_slope

        self.num_upsamples = len(upsample_rates)
        self.num_kernels = len(resblock_kernel_sizes)

        self.ups = nn.ModuleList()
        for i, (u, k) in enumerate(zip(upsample_rates, upsample_kernel_sizes)):
            self.ups.append(
                weight_norm(
                    nn.ConvTranspose1d(
                        upsample_initial_channel // (2**i),
                        upsample_initial_channel // (2 ** (i + 1)),
                        k,
                        u,
                        padding=(k - u) // 2,
                    )
                )
            )

        self.resblocks = nn.ModuleList()
        for i in range(len(self.ups)):
            ch = upsample_initial_channel // (2 ** (i + 1))
            for j, (k, d) in enumerate(
                zip(resblock_kernel_sizes, resblock_dilation_sizes)
            ):
                self.resblocks.append(ResBlock1(None, ch, k, d))

        self.conv_post = weight_norm(nn.Conv1d(ch, 1, 7, 1, padding=3))
        self.ups.apply(init_weights)
        self.conv_post.apply(init_weights)

    def forward(self, x):
        x = self.conv_pre(x)
        for i in range(self.num_upsamples):
            x = F.leaky_relu(x, self.leaky_relu_slope, inplace=True)
            x = self.ups[i](x)

            xs = None
            for j in range(self.num_kernels):
                if xs is None:
                    xs = self.resblocks[i * self.num_kernels + j](x)
                else:
                    xs += self.resblocks[i * self.num_kernels + j](x)
            x = xs / self.num_kernels

        x = F.leaky_relu(x, inplace=True)
        x = self.conv_post(x)
        x = torch.tanh(x)

        return x

    def remove_weight_norm(self):
        print("Removing weight norm...")
        for l in self.ups:
            remove_weight_norm(l)
        for l in self.resblocks:
            l.remove_weight_norm()
        remove_weight_norm(self.conv_pre)
        remove_weight_norm(self.conv_post)
