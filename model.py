import torch
import torch.nn as nn
import segmentation_models_pytorch as smp

def build_model(cfg, weights="imagenet"):
    """Factory function to instantiate the model."""
    return smp.UnetPlusPlus(
        encoder_name=cfg.ENCODER,
        encoder_weights=weights,
        in_channels=3, 
        classes=1, 
        activation=None
    )

class ComboLoss(nn.Module):
    """Exact logic: 0.5 Dice + 0.5 BCE"""
    def __init__(self):
        super().__init__()
        self.dice = smp.losses.DiceLoss(mode='binary', from_logits=True)
        self.bce = nn.BCEWithLogitsLoss()

    def forward(self, y_pred, y_true):
        return 0.5 * self.dice(y_pred, y_true) + 0.5 * self.bce(y_pred, y_true)