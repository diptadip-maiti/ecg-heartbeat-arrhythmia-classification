import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset


class ECGDataset(Dataset):
    def __init__(self, signals, rr_features, labels=None):
        self.signals = torch.from_numpy(
            np.asarray(signals, dtype=np.float32)
        ).unsqueeze(1)
        self.rr = torch.from_numpy(
            np.asarray(rr_features, dtype=np.float32)
        )
        self.labels = (
            None if labels is None
            else torch.from_numpy(np.asarray(labels, dtype=np.int64))
        )

    def __len__(self):
        return len(self.signals)

    def __getitem__(self, idx):
        if self.labels is None:
            return self.signals[idx], self.rr[idx]
        return self.signals[idx], self.rr[idx], self.labels[idx]


def validate_schema(train_df, test_df, signal_cols, rr_cols, target, id_col):
    required_train = set(signal_cols + rr_cols + [target, id_col])
    required_test = set(signal_cols + rr_cols + [id_col])

    missing_train = required_train - set(train_df.columns)
    missing_test = required_test - set(test_df.columns)

    if missing_train:
        raise ValueError(f"Missing training columns: {sorted(missing_train)}")
    if missing_test:
        raise ValueError(f"Missing test columns: {sorted(missing_test)}")


def prepare_data(train_df, test_df, signal_cols, rr_cols, target, seed=42,
                 use_beat_normalization=True):
    rr_medians = train_df[rr_cols].median()

    train_df = train_df.copy()
    test_df = test_df.copy()

    train_df[rr_cols] = train_df[rr_cols].fillna(rr_medians)
    test_df[rr_cols] = test_df[rr_cols].fillna(rr_medians)

    if train_df[signal_cols + rr_cols].isna().any().any():
        raise ValueError("Training features still contain missing values.")
    if test_df[signal_cols + rr_cols].isna().any().any():
        raise ValueError("Test features still contain missing values.")

    X_signal = train_df[signal_cols].to_numpy(dtype=np.float32)
    X_rr = train_df[rr_cols].to_numpy(dtype=np.float32)
    y = train_df[target].to_numpy(dtype=np.int64)

    X_signal_test = test_df[signal_cols].to_numpy(dtype=np.float32)
    X_rr_test = test_df[rr_cols].to_numpy(dtype=np.float32)

    indices = np.arange(len(y))
    train_idx, val_idx = train_test_split(
        indices,
        test_size=0.20,
        random_state=seed,
        shuffle=True,
        stratify=y,
    )

    X_sig_train = X_signal[train_idx].copy()
    X_sig_val = X_signal[val_idx].copy()
    X_rr_train = X_rr[train_idx].copy()
    X_rr_val = X_rr[val_idx].copy()
    y_train = y[train_idx].copy()
    y_val = y[val_idx].copy()

    if use_beat_normalization:
        X_sig_train = normalize_each_beat(X_sig_train)
        X_sig_val = normalize_each_beat(X_sig_val)
        X_sig_test = normalize_each_beat(X_signal_test)
    else:
        X_sig_test = X_signal_test.copy()

    rr_scaler = StandardScaler()
    X_rr_train = rr_scaler.fit_transform(X_rr_train).astype(np.float32)
    X_rr_val = rr_scaler.transform(X_rr_val).astype(np.float32)
    X_rr_test = rr_scaler.transform(X_rr_test).astype(np.float32)

    return {
        "X_sig_train": X_sig_train,
        "X_sig_val": X_sig_val,
        "X_sig_test": X_sig_test,
        "X_rr_train": X_rr_train,
        "X_rr_val": X_rr_val,
        "X_rr_test": X_rr_test,
        "y_train": y_train,
        "y_val": y_val,
        "train_idx": train_idx,
        "val_idx": val_idx,
        "rr_medians": rr_medians.to_dict(),
        "rr_scaler_mean": rr_scaler.mean_.tolist(),
        "rr_scaler_scale": rr_scaler.scale_.tolist(),
    }


class TestDataset(Dataset):
    def __init__(self, ids, signals, rr_features):
        self.ids = list(ids)
        self.signals = torch.from_numpy(
            np.asarray(signals, dtype=np.float32)
        ).unsqueeze(1)
        self.rr = torch.from_numpy(
            np.asarray(rr_features, dtype=np.float32)
        )

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        return self.ids[idx], self.signals[idx], self.rr[idx]
