# focal_loss.py

import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalBCEWithLogitsLoss(nn.Module):
    """
    Focal loss version of BCEWithLogitsLoss for binary classification.
    Works exactly like BCEWithLogitsLoss but adds:
    - alpha: balances positive vs negative
    - gamma: focuses on harder cases

    Recommended defaults for insurance-like imbalanced data:
    alpha = 0.75
    gamma = 2.0
    """

    def __init__(self, alpha=0.75, gamma=2.0, reduction="mean"):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, logits, targets):
        """
        logits: (N, 1) raw scores BEFORE sigmoid
        targets: (N, 1) binary labels {0,1}
        """
        targets = targets.float()

        # Standard BCE per sample
        bce_loss = F.binary_cross_entropy_with_logits(
            logits, targets, reduction="none"
        )

        # Probabilities p = sigmoid(logits)
        probs = torch.sigmoid(logits)

        # p_t = p if y=1, else (1-p)
        pt = probs * targets + (1.0 - probs) * (1.0 - targets)

        # alpha balancing: alpha for positives, (1-alpha) for negatives
        alpha_t = self.alpha * targets + (1.0 - self.alpha) * (1.0 - targets)

        # Focal scaling
        focal_weight = alpha_t * (1.0 - pt).pow(self.gamma)

        loss = focal_weight * bce_loss

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        else:
            return loss
