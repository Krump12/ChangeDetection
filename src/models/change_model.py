from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class SiameseUNet(nn.Module):
    def __init__(self, in_channels: int = 3, base_channels: int = 16) -> None:
        super().__init__()
        c = base_channels
        self.enc1 = ConvBlock(in_channels, c)
        self.enc2 = ConvBlock(c, c * 2)
        self.enc3 = ConvBlock(c * 2, c * 4)
        self.pool = nn.MaxPool2d(2)

        self.bridge = ConvBlock(c * 8, c * 8)
        self.up2 = nn.ConvTranspose2d(c * 8, c * 4, 2, stride=2)
        self.dec2 = ConvBlock(c * 12, c * 4)
        self.up1 = nn.ConvTranspose2d(c * 4, c * 2, 2, stride=2)
        self.dec1 = ConvBlock(c * 6, c * 2)
        self.up0 = nn.ConvTranspose2d(c * 2, c, 2, stride=2)
        self.dec0 = ConvBlock(c * 3, c)
        self.out = nn.Conv2d(c, 1, 1)

    def encode(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        return e1, e2, e3

    def forward(self, image_a: torch.Tensor, image_b: torch.Tensor) -> torch.Tensor:
        a1, a2, a3 = self.encode(image_a)
        b1, b2, b3 = self.encode(image_b)
        bridge = self.bridge(torch.cat([self.pool(a3), self.pool(b3)], dim=1))

        x = self.up2(bridge)
        x = F.interpolate(x, size=a3.shape[-2:], mode="bilinear", align_corners=False)
        x = self.dec2(torch.cat([x, a3, b3], dim=1))
        x = self.up1(x)
        x = F.interpolate(x, size=a2.shape[-2:], mode="bilinear", align_corners=False)
        x = self.dec1(torch.cat([x, a2, b2], dim=1))
        x = self.up0(x)
        x = F.interpolate(x, size=a1.shape[-2:], mode="bilinear", align_corners=False)
        x = self.dec0(torch.cat([x, a1, b1], dim=1))
        return self.out(x)


def build_model(config: dict | None = None) -> nn.Module:
    model_cfg = (config or {}).get("model", {}) if config else {}
    return SiameseUNet(
        in_channels=int(model_cfg.get("in_channels", 3)),
        base_channels=int(model_cfg.get("base_channels", 16)),
    )
