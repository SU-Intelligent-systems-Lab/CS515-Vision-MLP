import os
import glob
import pandas as pd
import matplotlib.pyplot as plt
from typing import List, Dict


def save_results(
    history: List[Dict[str, float]],
    model_name: str,
    save_dir: str = "results"
) -> None:
    """Saves training history to CSV and plots training curves.

    Creates a CSV file with per-epoch metrics and a PNG with
    loss and accuracy curves for both train and validation sets.

    Args:
        history: List of dicts with keys epoch, train_loss,
            train_acc, val_loss, val_acc.
        model_name: Name used for output filenames.
        save_dir: Directory to save output files to.
    """
    os.makedirs(save_dir, exist_ok=True)
    df = pd.DataFrame(history)
    df.to_csv(f"{save_dir}/{model_name}_results.csv", index=False)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    ax1.plot(df["epoch"], df["train_loss"], label="Train")
    ax1.plot(df["epoch"], df["val_loss"], label="Val")
    ax1.set_title(f"{model_name} — Loss")
    ax1.set_xlabel("Epoch")
    ax1.legend()

    ax2.plot(df["epoch"], df["train_acc"], label="Train")
    ax2.plot(df["epoch"], df["val_acc"], label="Val")
    ax2.set_title(f"{model_name} — Accuracy")
    ax2.set_xlabel("Epoch")
    ax2.legend()

    plt.tight_layout()
    plt.savefig(f"{save_dir}/{model_name}_curves.png")
    plt.close()


def plot_comparison(
    results_dir: str = "results",
    filter: str = ""
) -> None:
    """Generates a validation accuracy comparison plot for all models.

    Reads all CSV result files matching the filter string and plots
    their validation accuracy curves on a single figure.

    Args:
        results_dir: Directory containing result CSV files.
        filter: Optional substring to filter which CSVs to include.
            For example 'cifar100' to plot only CIFAR-100 results.
    """
    files = glob.glob(f"{results_dir}/*{filter}*_results.csv")
    fig, ax = plt.subplots(figsize=(8, 5))
    for f in files:
        df = pd.read_csv(f)
        name = f.split("/")[-1].replace("_results.csv", "")
        ax.plot(df["epoch"], df["val_acc"], label=name)
    ax.set_title("Validation Accuracy Comparison")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Accuracy (%)")
    ax.legend()
    plt.tight_layout()
    plt.savefig(f"{results_dir}/comparison{filter}.png")
    plt.close()