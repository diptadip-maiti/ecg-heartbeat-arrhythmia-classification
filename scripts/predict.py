import argparse
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR))

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from src.config import Config, ID_COL, RR_COLS, SIGNAL_COLS, TARGET
from src.data import TestDataset
from src.inference import predict_test, save_submission
from src.model import ECG_CNN
from src.utils import normalize_each_beat


def main(args):
    cfg = Config()

    data_dir = Path(args.data_dir)
    checkpoint_path = Path(args.checkpoint)

    train_df = pd.read_csv(data_dir / "train.csv")
    test_df = pd.read_csv(data_dir / "test.csv")
    sample_submission = pd.read_csv(data_dir / "sample_submission.csv")

    # Reproduce training-time RR imputation.
    rr_medians = train_df[RR_COLS].median()
    test_df[RR_COLS] = test_df[RR_COLS].fillna(rr_medians)

    X_signal_test = test_df[SIGNAL_COLS].to_numpy(dtype=np.float32)
    X_rr_test = test_df[RR_COLS].to_numpy(dtype=np.float32)

    checkpoint = torch.load(
        checkpoint_path,
        map_location=cfg.device,
        weights_only=False,
    )

    if checkpoint.get("use_beat_normalization", True):
        X_signal_test = normalize_each_beat(X_signal_test)

    scaler_mean = np.asarray(checkpoint["rr_scaler_mean"], dtype=np.float32)
    scaler_scale = np.asarray(checkpoint["rr_scaler_scale"], dtype=np.float32)
    X_rr_test = (X_rr_test - scaler_mean) / np.maximum(scaler_scale, 1e-12)

    test_dataset = TestDataset(
        test_df[ID_COL].tolist(),
        X_signal_test,
        X_rr_test,
    )

    loader = DataLoader(
        test_dataset,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=(cfg.num_workers > 0),
    )

    model = ECG_CNN(
        rr_dim=len(RR_COLS),
        num_classes=cfg.num_classes,
    ).to(cfg.device)
    model.load_state_dict(checkpoint["model_state_dict"])

    test_ids, test_preds, test_probs = predict_test(
        model, loader, cfg.device
    )

    print("Predictions:", test_preds.shape)
    print("Probability matrix:", test_probs.shape)

    submission = save_submission(
        sample_submission,
        test_preds,
        TARGET,
        args.output,
    )

    print(submission.head())
    print("Submission shape:", submission.shape)
    print("Saved to:", Path(args.output).resolve())


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", default="outputs/submission.csv")
    args = parser.parse_args()
    main(args)
