import numpy as np
import pandas as pd
import torch
from tqdm.auto import tqdm


@torch.no_grad()
def predict_test(model, loader, device):
    model.eval()

    ids = []
    predictions = []
    probabilities = []

    for batch_ids, signal, rr in tqdm(loader, desc="Predicting"):
        signal = signal.to(device, non_blocking=True)
        rr = rr.to(device, non_blocking=True)

        logits = model(signal, rr)
        probs = torch.softmax(logits, dim=1)
        preds = probs.argmax(dim=1)

        ids.extend(batch_ids)
        predictions.extend(preds.cpu().numpy())
        probabilities.append(probs.cpu().numpy())

    probabilities = np.concatenate(probabilities, axis=0)

    return ids, np.asarray(predictions), probabilities


def save_submission(sample_submission, predictions, target, output_path):
    submission = sample_submission.copy()

    if len(submission) != len(predictions):
        raise ValueError(
            f"Prediction count {len(predictions)} does not match "
            f"submission rows {len(submission)}."
        )

    if target not in submission.columns:
        raise ValueError(
            f"Expected target column '{target}' in sample submission."
        )

    submission[target] = predictions
    submission.to_csv(output_path, index=False)
    return submission
