"""Loss functions for model training."""

import torch
import torch.nn as nn

class FocalLoss(nn.Module):
    """Focal Loss for binary classification."""
    
    def __init__(self, gamma, pos_weight):
        """Initialize Focal Loss.
        
        Args:
            gamma: Focusing parameter
            pos_weight: Positive class weights
        """
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.w_p = pos_weight

    def forward(self, y_pred, y_true):
        """
        Compute Focal Loss.

        Args:
            y_true: True binary labels (0 or 1)
            y_pred: Predicted logits

        Returns:
            Computed focal loss
        """
        p = torch.sigmoid(y_pred)
        focal_loss_pos = - self.w_p * (1 - p) ** self.gamma * y_true * torch.log(p.clamp(min=1e-8))
        focal_loss_neg = - p ** self.gamma * (1 - y_true) * torch.log((1 - p).clamp(min=1e-8))
        return focal_loss_pos + focal_loss_neg

def get_loss_functions(config, device):
    """Get all loss functions based on configuration.
    
    Args:
        config: Configuration dictionary
        device: torch device

    Returns:
        Dictionary of loss functions
    """
    pos_weights = torch.tensor([1.0] * config["num_of_tasks"], device=device)
    
    return {
        'weighted': nn.BCEWithLogitsLoss(reduction="none", pos_weight=pos_weights),
        'non_weighted': nn.BCEWithLogitsLoss(reduction="none"),
        'focal': FocalLoss(gamma=config['gamma'], pos_weight=pos_weights),
        'mse': nn.MSELoss(),
        'cross_entropy': nn.CrossEntropyLoss(ignore_index=-1)
    }