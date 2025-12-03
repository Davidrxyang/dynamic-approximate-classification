#!/usr/bin/env python3
"""
Metric utilities for evaluating DAC predictions.
"""

from typing import Dict, Iterable, List


def compute_metrics(
    y_true: List[int],
    y_pred: List[int],
    label_space: Iterable[int],
) -> Dict[str, float]:
    """
    Compute accuracy, macro precision, macro recall, and macro F1-score.

    Args:
        y_true: list of ground-truth class_ids.
        y_pred: list of predicted class_ids.
        label_space: iterable of all possible class_ids.
    """
    if len(y_true) != len(y_pred):
        raise ValueError("y_true and y_pred must have the same length.")

    n = len(y_true)
    if n == 0:
        return {"accuracy": 0.0, "precision": 0.0, "recall": 0.0, "f1": 0.0}

    accuracy = sum(int(t == p) for t, p in zip(y_true, y_pred)) / n

    # Track tp/fp/fn per class for macro metrics
    stats = {lbl: {"tp": 0, "fp": 0, "fn": 0} for lbl in label_space}
    for t, p in zip(y_true, y_pred):
        if t == p:
            stats[t]["tp"] += 1
        else:
            stats[p]["fp"] += 1
            stats[t]["fn"] += 1

    precisions = []
    recalls = []
    for lbl in label_space:
        tp = stats[lbl]["tp"]
        fp = stats[lbl]["fp"]
        fn = stats[lbl]["fn"]
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        precisions.append(prec)
        recalls.append(rec)

    macro_precision = sum(precisions) / len(precisions)
    macro_recall = sum(recalls) / len(recalls)
    f1 = (
        2 * macro_precision * macro_recall / (macro_precision + macro_recall)
        if (macro_precision + macro_recall) > 0
        else 0.0
    )

    return {
        "accuracy": accuracy,
        "precision": macro_precision,
        "recall": macro_recall,
        "f1": f1,
    }


def compute_per_class_metrics(
    y_true: List[int],
    y_pred: List[int],
    label_space: Iterable[int],
) -> Dict[int, Dict[str, float]]:
    """
    Compute precision, recall, F1, and support per class_id.
    """
    if len(y_true) != len(y_pred):
        raise ValueError("y_true and y_pred must have the same length.")

    stats = {
        lbl: {"tp": 0, "fp": 0, "fn": 0, "support": 0, "predicted": 0}
        for lbl in label_space
    }

    for t, p in zip(y_true, y_pred):
        if t == p:
            stats[t]["tp"] += 1
        else:
            stats[p]["fp"] += 1
            stats[t]["fn"] += 1
        stats[t]["support"] += 1
        stats[p]["predicted"] += 1

    results: Dict[int, Dict[str, float]] = {}
    for lbl in label_space:
        tp = stats[lbl]["tp"]
        fp = stats[lbl]["fp"]
        fn = stats[lbl]["fn"]
        support = stats[lbl]["support"]
        predicted = stats[lbl]["predicted"]

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (
            2 * precision * recall / (precision + recall)
            if (precision + recall) > 0
            else 0.0
        )
        results[lbl] = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "support": float(support),
            "predicted": float(predicted),
            "correct": float(tp),
        }

    return results
