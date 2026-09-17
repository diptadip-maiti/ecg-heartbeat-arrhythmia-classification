import numpy as np
import torch
from sklearn.metrics import f1_score
from tqdm.auto import tqdm


def train_one_epoch(model, loader, criterion, optimizer, device, grad_clip=5.0):
    model.train()
    running_loss = 0.0
    all_preds = []
    all_labels = []

    for signal, rr, labels in tqdm(loader, leave=False):
        signal = signal.to(device, non_blocking=True)
        rr = rr.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        logits = model(signal, rr)
        loss = criterion(logits, labels)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()

        running_loss += loss.item() * labels.size(0)
        all_preds.append(logits.argmax(dim=1).detach().cpu())
        all_labels.append(labels.detach().cpu())

    y_pred = torch.cat(all_preds).numpy()
    y_true = torch.cat(all_labels).numpy()

    loss = running_loss / len(loader.dataset)
    f1 = f1_score(
        y_true, y_pred, average="macro",
        labels=[0, 1, 2, 3], zero_division=0
    )

    return loss, f1


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    all_preds = []
    all_labels = []
    all_probs = []

    for signal, rr, labels in loader:
        signal = signal.to(device, non_blocking=True)
        rr = rr.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        logits = model(signal, rr)
        loss = criterion(logits, labels)

        probs = torch.softmax(logits, dim=1)
        preds = probs.argmax(dim=1)

        running_loss += loss.item() * labels.size(0)
        all_preds.append(preds.cpu())
        all_labels.append(labels.cpu())
        all_probs.append(probs.cpu())

    y_pred = torch.cat(all_preds).numpy()
    y_true = torch.cat(all_labels).numpy()
    probs = torch.cat(all_probs).numpy()

    loss = running_loss / len(loader.dataset)
    f1 = f1_score(
        y_true, y_pred, average="macro",
        labels=[0, 1, 2, 3], zero_division=0
    )

    return loss, f1, y_pred, y_true, probs
