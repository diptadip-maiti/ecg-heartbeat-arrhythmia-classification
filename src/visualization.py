import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


def save_training_curves(history, output_dir):
    output_dir = Path(output_dir)

    epochs = np.arange(1, len(history["train_loss"]) + 1)

    plt.figure(figsize=(8, 5))
    plt.plot(epochs, history["train_f1"], label="Train Macro F1")
    plt.plot(epochs, history["val_f1"], label="Validation Macro F1")
    plt.xlabel("Epoch")
    plt.ylabel("Macro F1")
    plt.title("Training vs Validation Macro F1")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / "training_f1.png", dpi=180)
    plt.close()

    plt.figure(figsize=(8, 5))
    plt.plot(epochs, history["train_loss"], label="Train Loss")
    plt.plot(epochs, history["val_loss"], label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training vs Validation Loss")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / "training_loss.png", dpi=180)
    plt.close()


def save_confusion_matrix(cm, class_names, output_path):
    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=[class_names[i] for i in range(4)],
        yticklabels=[class_names[i] for i in range(4)],
    )
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Validation Confusion Matrix")
    plt.xticks(rotation=35, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close()


def save_class_f1(per_class_f1, class_names, output_path):
    labels = [class_names[i] for i in range(4)]

    plt.figure(figsize=(9, 5))
    plt.bar(labels, per_class_f1)
    plt.ylabel("F1 Score")
    plt.ylim(0, 1)
    plt.title("Validation F1 by Class")
    plt.xticks(rotation=25, ha="right")
    plt.tight_layout()
    plt.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close()


def save_signal_examples(X_signal, y, class_names, output_path, seed=42):
    rng = np.random.default_rng(seed)
    fig, axes = plt.subplots(4, 1, figsize=(12, 10), sharex=True)

    for class_id, ax in enumerate(axes):
        indices = np.where(y == class_id)[0]
        if len(indices) == 0:
            ax.set_title(f"Class {class_id}: {class_names[class_id]} (no samples)")
            continue

        chosen = rng.choice(indices, size=min(20, len(indices)), replace=False)
        for i in chosen:
            ax.plot(X_signal[i], alpha=0.25)

        ax.set_title(f"Class {class_id}: {class_names[class_id]}")
        ax.grid(alpha=0.2)

    axes[-1].set_xlabel("Sample")
    plt.tight_layout()
    plt.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close()


def save_metrics(metrics, output_path):
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
