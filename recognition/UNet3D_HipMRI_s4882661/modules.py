"""
modules.py - model components using PyTorch

Contains:
    ConvBlock     : two layers
    UNet          : U-Net  for dim=2 (baseline) and 3D U-Net 
    DiceCELoss    : soft Dice + cross-entropy, for class imbalance
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


def _layers(dim):
    if dim == 2:
        return nn.Conv2d, nn.ConvTranspose2d, nn.MaxPool2d, nn.InstanceNorm2d
    if dim == 3:
        return nn.Conv3d, nn.ConvTranspose3d, nn.MaxPool3d, nn.InstanceNorm3d
    raise ValueError(f"dim must be 2 or 3, got {dim}")


class ConvBlock(nn.Module):
    """Two 3x3(x3) convolutions"""

    def __init__(self, in_channels, out_channels, dim, dropout=0.0):
        super().__init__()
        conv, _, _, norm = _layers(dim)
        layers = [
            conv(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            norm(out_channels, affine=True),
            nn.LeakyReLU(0.01, inplace=True),
            conv(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            norm(out_channels, affine=True),
            nn.LeakyReLU(0.01, inplace=True),
        ]
        if dropout > 0:
            layers.append(nn.Dropout(dropout))
        self.block = nn.Sequential(*layers)

    def forward(self, x):
        return self.block(x)


class UNet(nn.Module):
    """
    Encoder-decoder with skip connections.
    """

    def __init__(self, dim=3, in_channels=1, num_classes=6,
                 base_channels=16, depth=3, dropout=0.0):
        super().__init__()
        conv, upconv, pool, _ = _layers(dim)
        self.dim, self.depth = dim, depth
        chans = [base_channels * (2 ** i) for i in range(depth + 1)]

        self.pool = pool(kernel_size=2)

        # Encoder:.
        self.encoders = nn.ModuleList()
        prev = in_channels
        for c in chans[:-1]:
            self.encoders.append(ConvBlock(prev, c, dim, dropout))
            prev = c

        self.bottleneck = ConvBlock(chans[-2], chans[-1], dim, dropout)

        # Decoder:
        self.upconvs = nn.ModuleList()
        self.decoders = nn.ModuleList()
        for i in range(depth - 1, -1, -1):
            self.upconvs.append(upconv(chans[i + 1], chans[i], kernel_size=2, stride=2))
            self.decoders.append(ConvBlock(2 * chans[i], chans[i], dim, dropout))

        self.head = conv(chans[0], num_classes, kernel_size=1)

    def forward(self, x):
        factor = 2 ** self.depth
        if any(s % factor != 0 for s in x.shape[2:]):
            raise ValueError(
                f"Spatial size {tuple(x.shape[2:])} must be divisible by {factor}")

        skips = []
        for enc in self.encoders:
            x = enc(x)
            skips.append(x)
            x = self.pool(x)

        x = self.bottleneck(x)

        for up, dec, skip in zip(self.upconvs, self.decoders, reversed(skips)):
            x = up(x)
            x = dec(torch.cat([skip, x], dim=1))

        return self.head(x)


class DiceCELoss(nn.Module):
    """
    Combined loss = ce_weight * CrossEntropy + dice_weight * (1 - mean soft Dice).
    Soft Dice is computed per class
    """

    def __init__(self, num_classes, dice_weight=1.0, ce_weight=1.0,
                 include_background=False, eps=1e-6):
        super().__init__()
        self.num_classes = num_classes
        self.dice_weight = dice_weight
        self.ce_weight = ce_weight
        self.include_background = include_background
        self.eps = eps

    def forward(self, logits, target):
        logits = logits.float() 
        ce = F.cross_entropy(logits, target)

        probs = F.softmax(logits, dim=1)
        onehot = F.one_hot(target, self.num_classes).movedim(-1, 1).to(probs.dtype)
        reduce_dims = [0] + list(range(2, probs.dim()))  # all but channel
        intersection = (probs * onehot).sum(reduce_dims)
        denominator = probs.sum(reduce_dims) + onehot.sum(reduce_dims)
        dice = (2 * intersection + self.eps) / (denominator + self.eps)
        if not self.include_background:
            dice = dice[1:]

        return self.ce_weight * ce + self.dice_weight * (1.0 - dice.mean())


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)