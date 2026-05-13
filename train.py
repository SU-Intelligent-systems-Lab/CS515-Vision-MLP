import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Tuple


def train(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device
) -> Tuple[float, float]:
    """Runs one epoch of training.

    Iterates over the training DataLoader, performs forward and
    backward passes, and updates model weights.

    Args:
        model: The neural network model to train.
        loader: DataLoader providing training batches.
        optimizer: Optimizer used to update model weights.
        criterion: Loss function.
        device: Device to run computations on (cpu or cuda).

    Returns:
        Tuple of (average_loss, accuracy) for the epoch.
    """
    model.train()
    total_loss, correct, total = 0.0, 0, 0
    for inputs, targets in loader:
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        correct += outputs.argmax(1).eq(targets).sum().item()
        total += targets.size(0)
    return total_loss / len(loader), 100.0 * correct / total