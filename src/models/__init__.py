"""
Model architectures for machine unlearning.
"""

from src.models.resnet import CIFARResNet18, create_resnet18

__all__ = [
    "CIFARResNet18",
    "create_resnet18",
]
