from __future__ import annotations

import torch
import torch.nn as nn


class BasicBlock(nn.Module):
    """
    CIFAR-compatible ResNet basic block.

    Handles both:
    - same-resolution residual connections
    - channel/stride changes using a projection shortcut
    """

    expansion = 1

    def __init__(
        self,
        in_planes: int,
        planes: int,
        stride: int = 1,
    ):
        super().__init__()

        self.conv1 = nn.Conv2d(
            in_channels=in_planes,
            out_channels=planes,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=False,
        )

        self.bn1 = nn.BatchNorm2d(planes)

        self.relu = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(
            in_channels=planes,
            out_channels=planes,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
        )

        self.bn2 = nn.BatchNorm2d(planes)

        # Projection shortcut is required when either:
        # 1. spatial resolution changes
        # 2. number of channels changes
        if stride != 1 or in_planes != planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(
                    in_channels=in_planes,
                    out_channels=planes,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                nn.BatchNorm2d(planes),
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        identity = self.shortcut(x)

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out = out + identity
        out = self.relu(out)

        return out


class CIFARResNet18(nn.Module):
    """
    ResNet-18 adapted for CIFAR-10.

    Input:
        [B, 3, 32, 32]

    Output:
        [B, 10]
    """

    def __init__(
        self,
        num_classes: int = 10,
    ):
        super().__init__()

        self.in_planes = 64

        # CIFAR-10 stem:
        # 3x3 convolution instead of ImageNet's 7x7 convolution.
        self.conv1 = nn.Conv2d(
            in_channels=3,
            out_channels=64,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
        )

        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)

        # 64 channels, 32x32
        self.layer1 = self._make_layer(
            planes=64,
            blocks=2,
            stride=1,
        )

        # 128 channels, 16x16
        self.layer2 = self._make_layer(
            planes=128,
            blocks=2,
            stride=2,
        )

        # 256 channels, 8x8
        self.layer3 = self._make_layer(
            planes=256,
            blocks=2,
            stride=2,
        )

        # 512 channels, 4x4
        self.layer4 = self._make_layer(
            planes=512,
            blocks=2,
            stride=2,
        )

        self.avgpool = nn.AdaptiveAvgPool2d(
            (1, 1)
        )

        self.fc = nn.Linear(
            512,
            num_classes,
        )

        self._initialize_weights()

    def _make_layer(
        self,
        planes: int,
        blocks: int,
        stride: int,
    ) -> nn.Sequential:

        layers = []

        # First block handles any spatial/channel change.
        layers.append(
            BasicBlock(
                in_planes=self.in_planes,
                planes=planes,
                stride=stride,
            )
        )

        self.in_planes = planes

        # Remaining blocks preserve dimensions.
        for _ in range(1, blocks):
            layers.append(
                BasicBlock(
                    in_planes=self.in_planes,
                    planes=planes,
                    stride=1,
                )
            )

        return nn.Sequential(*layers)

    def _initialize_weights(self) -> None:

        for module in self.modules():

            if isinstance(module, nn.Conv2d):

                nn.init.kaiming_normal_(
                    module.weight,
                    mode="fan_out",
                    nonlinearity="relu",
                )

            elif isinstance(module, nn.BatchNorm2d):

                nn.init.constant_(
                    module.weight,
                    1,
                )

                nn.init.constant_(
                    module.bias,
                    0,
                )

            elif isinstance(module, nn.Linear):

                nn.init.normal_(
                    module.weight,
                    mean=0,
                    std=0.01,
                )

                nn.init.constant_(
                    module.bias,
                    0,
                )

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)

        x = torch.flatten(
            x,
            start_dim=1,
        )

        x = self.fc(x)

        return x


def create_resnet18(
    num_classes: int = 10,
) -> CIFARResNet18:
    """
    Factory function used throughout the research pipeline.
    """

    return CIFARResNet18(
        num_classes=num_classes,
    )