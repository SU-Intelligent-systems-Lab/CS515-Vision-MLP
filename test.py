import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Tuple


def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device
) -> Tuple[float, float]:
    """Evaluates the model on a dataset.

    Runs inference over the provided DataLoader without computing
    gradients and returns loss and accuracy metrics.

    Args:
        model: The neural network model to evaluate.
        loader: DataLoader providing evaluation batches.
        criterion: Loss function.
        device: Device to run computations on (cpu or cuda).

    Returns:
        Tuple of (average_loss, accuracy) for the dataset.
    """
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    with torch.no_grad():
        for inputs, targets in loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            total_loss += loss.item()
            correct += outputs.argmax(1).eq(targets).sum().item()
            total += targets.size(0)
    return total_loss / len(loader), 100.0 * correct / total