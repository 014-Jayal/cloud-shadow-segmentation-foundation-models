import torch
import torch.nn as nn


class PrithviAdapter(nn.Module):
    """
    Adapter for Prithvi Vision Transformer backbone

    Converts transformer tokens into spatial feature maps
    for segmentation tasks.
    """

    def __init__(self, backbone, embed_dim=768):
        super().__init__()

        self.backbone = backbone
        self.embed_dim = embed_dim

    def forward(self, x):
        """
        x: (B, C, H, W)

        returns:
        feature map (B, D, H', W')
        """

        tokens = self._forward_features(x)

        # remove CLS token if present
        if tokens.shape[1] > 1:
            tokens = tokens[:, 1:, :]

        B, N, D = tokens.shape

        # reshape tokens → spatial
        H = W = int(N ** 0.5)

        features = tokens.transpose(1, 2).contiguous()
        features = features.view(B, D, H, W)

        return features

    def _forward_features(self, x):
        """
        Handles different backbone implementations
        """

        if hasattr(self.backbone, "forward_features"):
            return self.backbone.forward_features(x)

        return self.backbone(x)


def load_prithvi_backbone(checkpoint=None):
    """
    Placeholder for loading Prithvi model

    Replace this with actual loading logic:
    - HuggingFace
    - local checkpoint
    - ISRO/NASA pretrained weights
    """

    print("[INFO] Loading Prithvi backbone...")

    backbone = None  # <-- replace later

    if checkpoint:
        print(f"[INFO] Loading weights from {checkpoint}")

    return backbone