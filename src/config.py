from dataclasses import dataclass
import torch


SIGNAL_COLS = [f"sig_{i}" for i in range(250)]
RR_COLS = ["pre_rr", "post_rr", "rr_ratio"]
TARGET = "label"
ID_COL = "id"

CLASS_NAMES = {
    0: "Normal",
    1: "Supraventricular Ectopic",
    2: "Ventricular Ectopic",
    3: "Fusion",
}


@dataclass
class Config:
    seed: int = 42
    batch_size: int = 512
    epochs: int = 40
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    num_workers: int = 2
    patience: int = 8
    scheduler_patience: int = 2
    grad_clip: float = 5.0
    num_classes: int = 4
    use_beat_normalization: bool = True
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
