import torch
import torch.nn as nn


class ECG_CNN(nn.Module):
    def __init__(self, rr_dim=3, num_classes=4):
        super().__init__()

        self.cnn = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=7, padding=3, bias=False),
            nn.BatchNorm1d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(2),

            nn.Conv1d(32, 64, kernel_size=5, padding=2, bias=False),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(2),

            nn.Conv1d(64, 128, kernel_size=5, padding=2, bias=False),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),

            nn.Conv1d(128, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),

            nn.AdaptiveAvgPool1d(1),
        )

        self.rr_net = nn.Sequential(
            nn.Linear(rr_dim, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(inplace=True),
            nn.Dropout(0.20),
        )

        self.classifier = nn.Sequential(
            nn.Linear(128 + 32, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(0.30),
            nn.Linear(64, num_classes),
        )

    def forward(self, signal, rr):
        x = self.cnn(signal).squeeze(-1)
        r = self.rr_net(rr)
        x = torch.cat([x, r], dim=1)
        return self.classifier(x)
