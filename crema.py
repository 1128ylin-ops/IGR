# file: train/teaser_api.py
# NOTE: Interface-only, not runnable.

import torch
from torch.utils.data import DataLoader
from typing import Tuple, Dict

def compute_map(prob: torch.Tensor, onehot: torch.Tensor) -> float:
    """
    Args:
      prob:   [B, C], probability scores per class
      onehot: [B, C], one-hot ground truth
    Returns:
      float: mean average precision (mAP)
    """
    raise NotImplementedError("Metrics are intentionally omitted.")

def alignment_loss(p: torch.Tensor, q: torch.Tensor, temperature: float = 1.0) -> torch.Tensor:
    """
    Symmetrized alignment (e.g., JS/KL-based) between two logits.
    Shapes:
      p, q: [B, C]
    """
    raise NotImplementedError("Loss is intentionally omitted.")

def divergence_loss(z_a: torch.Tensor, z_v: torch.Tensor) -> torch.Tensor:
    """
    Decorrelation/divergence between embeddings.
    Shapes:
      z_a, z_v: [B, D]
    """
    raise NotImplementedError("Loss is intentionally omitted.")

def train_epoch(
    epoch: int,
    loader: DataLoader,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    config: Dict,
) -> Dict[str, float]:
    """
    Returns a dict of logged scalars, e.g. {'loss': ..., 'acc': ...}
    """
    raise NotImplementedError("Training logic is intentionally omitted.")

def validate_epoch(
    epoch: int,
    loader: DataLoader,
    model: torch.nn.Module,
    config: Dict,
) -> Tuple[float, float]:
    """
    Returns:
      (acc, mAP)
    """
    raise NotImplementedError("Validation logic is intentionally omitted.")
