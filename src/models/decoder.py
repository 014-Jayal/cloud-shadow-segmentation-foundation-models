import torch
import torch.nn as nn
import torch.nn.functional as F


class SimpleDecoder(nn.Module):
    """
    Lightweight decoder for segmentation

    Converts transformer features into pixel-wise predictions
    """

    def __init__(self, in_channels=768, mid_channels=256, num_classes=3):
        super().__init__()

        self.conv1 = nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(mid_channels)

        self.conv2 = nn.Conv2d(mid_channels, mid_channels // 2, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(mid_channels // 2)

        self.out = nn.Conv2d(mid_channels // 2, num_classes, kernel_size=1)

    def forward(self, x, output_size):
        """
        x: (B, D, H, W)
        output_size: (H, W) of original image
        """

        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))

        x = self.out(x)

        # upsample back to input resolution
        x = F.interpolate(x, size=output_size, mode="bilinear", align_corners=False)

        return x


class CloudSegmentationModel(nn.Module):
    """
    Full model = Prithvi backbone + decoder
    """

    def __init__(self, backbone, decoder):
        super().__init__()

        self.backbone = backbone
        self.decoder = decoder

    def forward(self, x):
        """
        x: (B, C, H, W)
        """

        input_size = x.shape[-2:]

        features = self.backbone(x)
        out = self.decoder(features, input_size)

        return out