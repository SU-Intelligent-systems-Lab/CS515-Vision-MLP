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
            For example 'cifar10' to plot only CIFAR-10 results.
    """
    files = glob.glob(f"{results_dir}/*_results.csv")
    
    # Filter files that end with exactly the filter string before _results.csv
    if filter:
        files = [f for f in files if f.replace("_results.csv", "").endswith(filter)]
    
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
    plt.savefig(f"{results_dir}/comparison_{filter}.png")
    plt.close()

def plot_final_accuracy(results_dir: str = "results") -> None:
    """Generates a grouped bar chart of final validation accuracy for all models.

    Reads all CSV result files and plots the last epoch validation
    accuracy grouped by dataset for easy cross-model comparison.

    Args:
        results_dir: Directory containing result CSV files.
    """
    import numpy as np

    cifar10_files = [f for f in glob.glob(f"{results_dir}/*_results.csv")
                     if f.replace("_results.csv", "").endswith("cifar10")]
    cifar100_files = [f for f in glob.glob(f"{results_dir}/*_results.csv")
                      if f.replace("_results.csv", "").endswith("cifar100")]

    def get_final_acc(files, dataset):
        results = {}
        for f in files:
            df = pd.read_csv(f)
            name = f.split("\\")[-1].split("/")[-1].replace("_results.csv", "")
            # Strip dataset suffix and clean up label
            label = name.replace(f"_{dataset}", "").replace("_", " ")
            results[label] = df["val_acc"].iloc[-1]
        return results

    cifar10_results = get_final_acc(cifar10_files, "cifar10")
    cifar100_results = get_final_acc(cifar100_files, "cifar100")

    # Only show models that have at least a CIFAR-10 result
    all_models = sorted(cifar10_results.keys())
    x = np.arange(len(all_models))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 6))
    bars1 = ax.bar(x - width/2,
                   [cifar10_results.get(m, 0) for m in all_models],
                   width, label="CIFAR-10", color="#0D9488")

    # Only plot CIFAR-100 bars where data exists
    cifar100_vals = [cifar100_results.get(m, None) for m in all_models]
    cifar100_x = [x[i] + width/2 for i, v in enumerate(cifar100_vals) if v is not None]
    cifar100_y = [v for v in cifar100_vals if v is not None]
    bars2 = ax.bar(cifar100_x, cifar100_y, width, label="CIFAR-100", color="#1B2A4A")

    ax.bar_label(bars1, fmt="%.1f", padding=3, fontsize=9)
    ax.bar_label(bars2, fmt="%.1f", padding=3, fontsize=9)

    ax.set_title("Final Validation Accuracy by Model and Dataset", fontsize=14)
    ax.set_xlabel("Model")
    ax.set_ylabel("Accuracy (%)")
    ax.set_xticks(x)
    ax.set_xticklabels(all_models, rotation=15, ha="right", fontsize=10)
    ax.legend()
    ax.set_ylim(0, 105)
    plt.tight_layout()
    plt.savefig(f"{results_dir}/final_accuracy_comparison.png")
    plt.close()


def plot_overfitting_gap(results_dir: str = "results") -> None:
    """Generates a bar chart showing the train/val accuracy gap at epoch 30.

    A larger gap indicates more severe overfitting. Reads all CSV
    result files and plots train_acc minus val_acc for the final epoch.

    Args:
        results_dir: Directory containing result CSV files.
    """
    files = glob.glob(f"{results_dir}/*_results.csv")
    names, gaps = [], []

    for f in sorted(files):
        df = pd.read_csv(f)
        name = f.split("\\")[-1].split("/")[-1].replace("_results.csv", "")
        gap = df["train_acc"].iloc[-1] - df["val_acc"].iloc[-1]
        names.append(name)
        gaps.append(gap)

    colors = ["#DC2626" if g > 20 else "#F59E0B" if g > 10 else "#16A34A"
              for g in gaps]

    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.bar(names, gaps, color=colors)
    ax.bar_label(bars, fmt="%.1f", padding=3, fontsize=9)

    ax.set_title("Overfitting Gap at Epoch 30  (Train Acc − Val Acc)", fontsize=14)
    ax.set_xlabel("Model")
    ax.set_ylabel("Gap (%)")
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, rotation=15, ha="right")

    # Legend for color coding
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor="#DC2626", label="Severe (>20%)"),
        Patch(facecolor="#F59E0B", label="Moderate (10-20%)"),
        Patch(facecolor="#16A34A", label="Mild (<10%)"),
    ]
    ax.legend(handles=legend_elements)
    plt.tight_layout()
    plt.savefig(f"{results_dir}/overfitting_gap.png")
    plt.close()