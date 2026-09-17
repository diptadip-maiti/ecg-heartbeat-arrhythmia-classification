import argparse
from pathlib import Path
import sys

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR))

import numpy as np
import pandas as pd

from src.config import ID_COL, RR_COLS, SIGNAL_COLS, TARGET


def main(data_dir):
    data_dir = Path(data_dir)
    train = pd.read_csv(data_dir / "train.csv")
    test = pd.read_csv(data_dir / "test.csv")
    sample = pd.read_csv(data_dir / "sample_submission.csv")

    required_train = SIGNAL_COLS + RR_COLS + [TARGET, ID_COL]
    required_test = SIGNAL_COLS + RR_COLS + [ID_COL]

    missing_train = [c for c in required_train if c not in train.columns]
    missing_test = [c for c in required_test if c not in test.columns]

    if missing_train:
        raise ValueError(f"Missing train columns: {missing_train}")
    if missing_test:
        raise ValueError(f"Missing test columns: {missing_test}")

    print("Train shape:", train.shape)
    print("Test shape:", test.shape)
    print("Sample submission shape:", sample.shape)
    print("Signal columns:", len(SIGNAL_COLS))
    print("RR columns:", RR_COLS)
    print("Classes:", sorted(train[TARGET].unique()))

    print("\nTrain missing values:")
    print(train[SIGNAL_COLS + RR_COLS + [TARGET]].isna().sum().loc[lambda s: s > 0])

    print("\nTest missing values:")
    print(test[SIGNAL_COLS + RR_COLS].isna().sum().loc[lambda s: s > 0])

    print(
        "\nTrain numeric infinities:",
        np.isinf(train.select_dtypes(include=np.number)).any().any()
    )
    print(
        "Test numeric infinities :",
        np.isinf(test.select_dtypes(include=np.number)).any().any()
    )

    assert set(train[TARGET].unique()).issubset({0, 1, 2, 3})
    assert len(train) > 0
    assert len(test) > 0

    print("\nSanity check passed.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", required=True)
    args = parser.parse_args()
    main(args.data_dir)
