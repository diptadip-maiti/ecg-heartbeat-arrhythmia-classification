import argparse
import os
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR))

import numpy as np
import pandas as pd
import torch
from sklearn.utils.class_weight import compute_class_weight
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau

from src.config import (
    CLASS_NAMES, ID_COL, RR_COLS, SIGNAL_COLS, TARGET, Config
)
from src.data import ECGDataset, prepare_data, validate_schema
from src.engine import evaluate, train_one_epoch
from src.losses import FocalLoss
from src.metrics import classification_details, prediction_distribution
from src.model import ECG_CNN
from src.utils import print_environment, seed_everything
from src.visualization import (
    save_class_f1,
    save_confusion_matrix,
    save_metrics,
    save_signal_examples,
    save_training_curves,
)


def main(args):
    cfg = Config()
    seed_everything(cfg.seed)
    print_environment()
    print("Device:", cfg.device)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    train_df = pd.read_csv(Path(args.data_dir) / "train.csv")
    test_df = pd.read_csv(Path(args.data_dir) / "test.csv")
    sample_submission = pd.read_csv(Path(args.data_dir) / "sample_submission.csv")

    print("Train:", train_df.shape)
    print("Test :", test_df.shape)
    print("Sample submission:", sample_submission.shape)

    validate_schema(
        train_df, test_df,
        SIGNAL_COLS, RR_COLS, TARGET, ID_COL
    )

    print("Classes:", sorted(train_df[TARGET].unique()))
    print("\nClass distribution:")
    print(train_df[TARGET].value_counts().sort_index())

    data = prepare_data(
        train_df,
        test_df,
        SIGNAL_COLS,
        RR_COLS,
        TARGET,
        seed=cfg.seed,
        use_beat_normalization=cfg.use_beat_normalization,
    )

    train_dataset = ECGDataset(
        data["X_sig_train"], data["X_rr_train"], data["y_train"]
    )
    val_dataset = ECGDataset(
        data["X_sig_val"], data["X_rr_val"], data["y_val"]
    )

    loader_kwargs = {
        "batch_size": cfg.batch_size,
        "num_workers": cfg.num_workers,
        "pin_memory": torch.cuda.is_available(),
    }

    if cfg.num_workers > 0:
        loader_kwargs["persistent_workers"] = True

    train_loader = DataLoader(
        train_dataset, shuffle=True, drop_last=False, **loader_kwargs
    )
    val_loader = DataLoader(
        val_dataset, shuffle=False, drop_last=False, **loader_kwargs
    )

    classes = np.arange(cfg.num_classes)
    class_weights_np = compute_class_weight(
        class_weight="balanced",
        classes=classes,
        y=data["y_train"],
    )
    class_weights = torch.tensor(
        class_weights_np, dtype=torch.float32, device=cfg.device
    )

    print("\nBalanced class weights:")
    for c, w in zip(classes, class_weights_np):
        print(f"Class {c}: {w:.4f}")

    model = ECG_CNN(
        rr_dim=len(RR_COLS),
        num_classes=cfg.num_classes
    ).to(cfg.device)

    criterion = torch.nn.CrossEntropyLoss(weight=class_weights)
    optimizer = AdamW(
        model.parameters(),
        lr=cfg.learning_rate,
        weight_decay=cfg.weight_decay,
    )
    scheduler = ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=0.5,
        patience=cfg.scheduler_patience,
        min_lr=1e-6,
    )

    history = {
        "train_loss": [],
        "train_f1": [],
        "val_loss": [],
        "val_f1": [],
        "lr": [],
    }

    best_f1 = -np.inf
    epochs_without_improvement = 0

    for epoch in range(1, cfg.epochs + 1):
        train_loss, train_f1 = train_one_epoch(
            model, train_loader, criterion, optimizer,
            cfg.device, cfg.grad_clip
        )

        val_loss, val_f1, _, _, _ = evaluate(
            model, val_loader, criterion, cfg.device
        )

        scheduler.step(val_f1)
        current_lr = optimizer.param_groups[0]["lr"]

        history["train_loss"].append(train_loss)
        history["train_f1"].append(train_f1)
        history["val_loss"].append(val_loss)
        history["val_f1"].append(val_f1)
        history["lr"].append(current_lr)

        print(
            f"Epoch {epoch:02d}/{cfg.epochs} | "
            f"train loss {train_loss:.4f} | "
            f"train F1 {train_f1:.4f} | "
            f"val loss {val_loss:.4f} | "
            f"val Macro F1 {val_f1:.4f} | "
            f"lr {current_lr:.2e}"
        )

        if val_f1 > best_f1:
            best_f1 = val_f1
            epochs_without_improvement = 0

            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "best_f1": float(best_f1),
                    "rr_medians": data["rr_medians"],
                    "rr_scaler_mean": data["rr_scaler_mean"],
                    "rr_scaler_scale": data["rr_scaler_scale"],
                    "use_beat_normalization": cfg.use_beat_normalization,
                    "config": vars(cfg),
                },
                output_dir / "best_model.pt",
            )
            print(f"  -> saved best model: {best_f1:.4f}")
        else:
            epochs_without_improvement += 1

        if epochs_without_improvement >= cfg.patience:
            print("Early stopping.")
            break

    checkpoint = torch.load(
        output_dir / "best_model.pt",
        map_location=cfg.device,
        weights_only=False,
    )
    model.load_state_dict(checkpoint["model_state_dict"])

    final_val_loss, final_val_f1, val_preds, val_labels, _ = evaluate(
        model, val_loader, criterion, cfg.device
    )

    report, cm, per_class_f1 = classification_details(
        val_labels, val_preds, CLASS_NAMES
    )

    print(f"\nBest validation Macro F1: {final_val_f1:.5f}")
    print(pd.DataFrame(report).T.round(4))
    print("\nConfusion matrix:")
    print(cm)
    print("\nValidation prediction distribution:")
    print(prediction_distribution(val_preds))

    metrics = {
        "best_validation_macro_f1": float(checkpoint["best_f1"]),
        "final_validation_loss": float(final_val_loss),
        "final_validation_macro_f1": float(final_val_f1),
        "epochs_completed": len(history["train_loss"]),
        "class_f1": {
            CLASS_NAMES[i]: float(per_class_f1[i])
            for i in range(cfg.num_classes)
        },
        "validation_prediction_distribution": prediction_distribution(val_preds),
        "config": vars(cfg),
    }

    save_training_curves(history, output_dir)
    save_confusion_matrix(
        cm, CLASS_NAMES, output_dir / "confusion_matrix.png"
    )
    save_class_f1(
        per_class_f1, CLASS_NAMES, output_dir / "class_f1.png"
    )
    save_signal_examples(
        data["X_sig_train"].copy(),
        data["y_train"],
        CLASS_NAMES,
        output_dir / "signal_examples.png",
        seed=cfg.seed,
    )
    save_metrics(metrics, output_dir / "metrics.json")

    print("\nArtifacts saved to:", output_dir.resolve())


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--output-dir", default="outputs")
    args = parser.parse_args()
    main(args)
