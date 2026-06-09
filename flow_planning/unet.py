import logging

import torch
import torch.nn as nn
from einops.layers.torch import Rearrange
from torch import Tensor

from flow_planning.policy import sinusoidal_embedding

log = logging.getLogger(__name__)


class Downsample1d(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.conv = nn.Conv1d(dim, dim, 3, 2, 1)

    def forward(self, x):
        return self.conv(x)


class Upsample1d(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.conv = nn.ConvTranspose1d(dim, dim, 4, 2, 1)

    def forward(self, x):
        return self.conv(x)


class ConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        kernel_size = 5
        self.block = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size, padding=kernel_size // 2),
            nn.GroupNorm(8, out_channels),
            nn.Mish(),
        )

    def forward(self, x):
        return self.block(x)


class ResBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, cond_dim: int):
        super().__init__()
        self.conv1 = ConvBlock(in_channels, out_channels)
        self.conv2 = ConvBlock(out_channels, out_channels)
        self.cond_encoder = nn.Sequential(
            nn.Mish(),
            nn.Linear(cond_dim, out_channels),
            Rearrange("batch t -> batch t 1"),
        )
        self.residual_conv = (
            nn.Conv1d(in_channels, out_channels, 1)
            if in_channels != out_channels
            else nn.Identity()
        )

    def forward(self, x, cond):
        out = self.conv1(x)
        embed = self.cond_encoder(cond)
        out = self.conv2(out + embed)
        return out + self.residual_conv(x)


class ConditionalUnet1D(nn.Module):
    def __init__(self, obs_dim, act_dim, cond_dim, down_dims, device):
        super().__init__()
        input_dim = obs_dim + act_dim
        all_dims = [input_dim] + list(down_dims)
        start_dim = down_dims[0]
        in_out = list(zip(all_dims[:-1], all_dims[1:], strict=False))
        self.cond_dim = cond_dim

        # diffusion step embedding and observations
        self.cond_encoder = nn.Sequential(
            nn.Linear(cond_dim, cond_dim * 4),
            nn.Mish(),
            nn.Linear(cond_dim * 4, cond_dim),
        )

        self.down_modules = nn.ModuleList([])
        for ind, (dim_in, dim_out) in enumerate(in_out):
            is_last = ind >= (len(in_out) - 1)
            self.down_modules.append(
                nn.ModuleList(
                    [
                        ResBlock(dim_in, dim_out, cond_dim),
                        ResBlock(dim_out, dim_out, cond_dim),
                        Downsample1d(dim_out) if not is_last else nn.Identity(),
                    ]
                )
            )

        mid_dim = all_dims[-1]

        self.mid_modules = nn.ModuleList(
            [
                ResBlock(mid_dim, mid_dim, cond_dim),
                ResBlock(mid_dim, mid_dim, cond_dim),
            ]
        )

        self.up_modules = nn.ModuleList([])
        for ind, (dim_in, dim_out) in enumerate(reversed(in_out[1:])):
            is_last = ind >= (len(in_out) - 1)
            self.up_modules.append(
                nn.ModuleList(
                    [
                        ResBlock(dim_out * 2, dim_in, cond_dim),
                        ResBlock(dim_in, dim_in, cond_dim),
                        Upsample1d(dim_in) if not is_last else nn.Identity(),
                    ]
                )
            )

        self.final_conv = nn.Sequential(
            ConvBlock(start_dim, start_dim),
            nn.Conv1d(start_dim, input_dim, 1),
        )

        self.to(device)

        total_params = sum(p.numel() for p in self.parameters())
        log.info(f"Total parameters: {total_params:e}")

    def forward(self, x: Tensor, t: Tensor, data: dict[str, Tensor]) -> Tensor:
        x = x.transpose(1, 2)
        temb = sinusoidal_embedding(t.view(-1), self.cond_dim)
        global_feature = self.cond_encoder(temb)

        h = []
        for mod_list in self.down_modules:
            assert isinstance(mod_list, nn.ModuleList)
            x = mod_list[0](x, global_feature)
            x = mod_list[1](x, global_feature)
            h.append(x)
            x = mod_list[2](x)

        for mid_module in self.mid_modules:
            x = mid_module(x, global_feature)

        for mod_list in self.up_modules:
            assert isinstance(mod_list, nn.ModuleList)
            x = torch.cat((x, h.pop()), dim=1)
            x = mod_list[0](x, global_feature)
            x = mod_list[1](x, global_feature)
            x = mod_list[2](x)

        x = self.final_conv(x)
        x = x.transpose(1, 2)
        return x
