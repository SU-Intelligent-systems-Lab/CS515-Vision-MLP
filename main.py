import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms
from typing import Tuple

from models.mixer import MLPMixer
from models.efficientnet import get_efficientnet
from train import train
from test import evaluate
from utils import save_results, plot_comparison


def get_args() -> argparse.Namespace:
    """Parses command line arguments for the training pipeline.

    Returns:
        Namespace object containing all training configuration
        arguments including model type, dataset, hyperparameters,
        and device settings.
    """
    parser = argparse.ArgumentParser(
        description="MLP-Mixer vs EfficientNet comparison"
    )
    parser.add_argument("--model", type=str, default="mixer",
                        choices=["mixer", "efficientnet", "mixer_pretrained"])
    parser.add_argument("--dataset", type=str, default="cifar10",
                        choices=["cifar10", "cifar100"])
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--pretrained", action="store_true")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--save_path", type=str, default=None)
    return parser.parse_args()


def get_data(
    batch_size: int,
    dataset: str = "cifar10"
) -> Tuple[DataLoader, DataLoader]:
    """Loads and preprocesses the specified image classification dataset.

    Applies resizing to 224x224, random horizontal flipping for
    training, and standard CIFAR normalization to both splits.

    Args:
        batch_size: Number of samples per batch.
        dataset: Dataset to load, either 'cifar10' or 'cifar100'.

    Returns:
        Tuple of (train_loader, test_loader) DataLoader objects.
    """
    transform_train = transforms.Compose([
        transforms.Resize(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465),
                             (0.2023, 0.1994, 0.2010)),
    ])
    transform_test = transforms.Compose([
        transforms.Resize(224),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465),
                             (0.2023, 0.1994, 0.2010)),
    ])
    if dataset == "cifar100":
        train_set = torchvision.datasets.CIFAR100(
            root="./data", train=True, download=True,
            transform=transform_train)
        test_set = torchvision.datasets.CIFAR100(
            root="./data", train=False, download=True,
            transform=transform_test)
    else:
        train_set = torchvision.datasets.CIFAR10(
            root="./data", train=True, download=True,
            transform=transform_train)
        test_set = torchvision.datasets.CIFAR10(
            root="./data", train=False, download=True,
            transform=transform_test)

    train_loader = DataLoader(train_set, batch_size=batch_size,
                              shuffle=True, num_workers=2)
    test_loader = DataLoader(test_set, batch_size=batch_size,
                             shuffle=False, num_workers=2)
    return train_loader, test_loader


def main() -> None:
    """Runs the full training and evaluation pipeline.

    Parses arguments, loads dataset, initializes selected
    model, trains for the specified number of epochs, saves the 
    results and optionally saves model weights to disk.
    """
    args = get_args()
    device = torch.device(
        args.device if torch.cuda.is_available() else "cpu"
    )
    print(f"Using device: {device}")

    num_classes = 100 if args.dataset == "cifar100" else 10
    train_loader, test_loader = get_data(args.batch_size, args.dataset)

    if args.model == "mixer":
        model = MLPMixer(
            image_size=224, patch_size=16, num_classes=num_classes,
            hidden_dim=512, num_layers=8,
            tokens_mlp_dim=256, channels_mlp_dim=2048
        )
    elif args.model == "mixer_pretrained":
        from models.mixer_pretrained import get_pretrained_mixer
        model = get_pretrained_mixer(num_classes=num_classes)
    else:
        model = get_efficientnet(
            num_classes=num_classes, pretrained=args.pretrained
        )

    model = model.to(device)
    criterion = nn.CrossEntropyLoss()

    lr = 1e-4 if args.model == "mixer_pretrained" else args.lr
    optimizer = torch.optim.Adam(
        model.parameters(), lr=lr, weight_decay=1e-4
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs
    )

    history = []
    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = train(
            model, train_loader, optimizer, criterion, device
        )
        val_loss, val_acc = evaluate(
            model, test_loader, criterion, device
        )
        scheduler.step()
        history.append({
            "epoch": epoch,
            "train_loss": train_loss, "train_acc": train_acc,
            "val_loss": val_loss,     "val_acc": val_acc
        })
        print(f"Epoch {epoch:>3} | "
              f"Train Loss {train_loss:.4f} Acc {train_acc:.1f}% | "
              f"Val Loss {val_loss:.4f} Acc {val_acc:.1f}%")

    pretrained_tag = "pretrained" if args.pretrained or \
        args.model == "mixer_pretrained" else "scratch"
    model_name = f"{args.model}_{pretrained_tag}_{args.dataset}"
    save_results(history, model_name)

    if args.save_path:
        torch.save(model.state_dict(), args.save_path)
    print("Done. Results saved to results/")


if __name__ == "__main__":
    main()