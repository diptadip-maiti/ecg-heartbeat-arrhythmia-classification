import numpy as np
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
)


def macro_f1(y_true, y_pred):
    return f1_score(
        y_true,
        y_pred,
        average="macro",
        labels=[0, 1, 2, 3],
        zero_division=0,
    )


def classification_details(y_true, y_pred, class_names):
    labels = [0, 1, 2, 3]
    report = classification_report(
        y_true,
        y_pred,
        labels=labels,
        target_names=[class_names[i] for i in labels],
        output_dict=True,
        zero_division=0,
    )
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    per_class = f1_score(
        y_true, y_pred,
        average=None,
        labels=labels,
        zero_division=0,
    )

    return report, cm, per_class


def prediction_distribution(y_pred):
    values, counts = np.unique(y_pred, return_counts=True)
    return {int(k): int(v) for k, v in zip(values, counts)}
