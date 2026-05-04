import torch
import torch.nn as nn
import torch.nn.functional as F


class DiceLoss(nn.Module):
    """
    Multi-class Dice Loss
    Helps preserve structure and overlap
    """

    def __init__(self, smooth=1.0):
        super().__init__()
        self.smooth = smooth

    def forward(self, logits, targets):
        """
        logits: (B, C, H, W)
        targets: (B, H, W)
        """

        num_classes = logits.shape[1]

        probs = torch.softmax(logits, dim=1)

        # one-hot encode targets
        targets_one_hot = F.one_hot(targets, num_classes=num_classes)
        targets_one_hot = targets_one_hot.permute(0, 3, 1, 2).float()

        # flatten
        probs = probs.contiguous().view(probs.shape[0], probs.shape[1], -1)
        targets_one_hot = targets_one_hot.contiguous().view(targets_one_hot.shape[0], targets_one_hot.shape[1], -1)

        intersection = (probs * targets_one_hot).sum(dim=2)
        union = probs.sum(dim=2) + targets_one_hot.sum(dim=2)

        dice = (2 * intersection + self.smooth) / (union + self.smooth)

        loss = 1 - dice.mean()

        return loss


class FocalLoss(nn.Module):
    """
    Multi-class Focal Loss
    Handles class imbalance
    """

    def __init__(self, gamma=2.0):
        super().__init__()
        self.gamma = gamma

    def forward(self, logits, targets):
        """
        logits: (B, C, H, W)
        targets: (B, H, W)
        """

        ce_loss = F.cross_entropy(logits, targets, reduction="none")

        pt = torch.exp(-ce_loss)
        focal_loss = (1 - pt) ** self.gamma * ce_loss

        return focal_loss.mean()


class HybridLoss(nn.Module):
    """
    Combined Focal + Dice Loss
    """

    def __init__(self, alpha=0.5):
        super().__init__()

        self.focal = FocalLoss()
        self.dice = DiceLoss()
        self.alpha = alpha

    def forward(self, logits, targets):
        focal_loss = self.focal(logits, targets)
        dice_loss = self.dice(logits, targets)

        return self.alpha * focal_loss + (1 - self.alpha) * dice_loss