"""Comprehensive Medical Classification Metrics Suite for Diabetic Retinopathy.
Computes:
- Accuracy, Balanced Accuracy
- Macro Precision, Macro Recall, Macro-F1
- Quadratic Weighted Kappa (Cohen's Kappa with quadratic weights)
- One-vs-Rest Macro AUROC
- Ordinal Mean Absolute Error (MAE)
- Brier Multi-Class Score
- Expected Calibration Error (ECE across 10 bins)
- Per-class Precision, Recall, and F1
- Confusion Matrix
"""
import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, precision_score, 
    recall_score, f1_score, cohen_kappa_score, roc_auc_score, confusion_matrix
)
from typing import Dict, Any, List, Union

def compute_all_medical_metrics(y_true: Union[List[int], np.ndarray], 
                                y_pred_probs: Union[List[List[float]], np.ndarray],
                                num_classes: int = 5) -> Dict[str, Any]:
    """Calculates full medical grading and calibration metric suite."""
    y_true = np.asarray(y_true, dtype=int)
    y_pred_probs = np.asarray(y_pred_probs, dtype=float)
    
    if y_pred_probs.ndim == 1:
        # If integer predictions passed
        y_pred = y_pred_probs.astype(int)
        y_probs = np.eye(num_classes)[y_pred]
    else:
        # Ensure proper probability distribution
        if np.any(y_pred_probs < 0) or not np.allclose(y_pred_probs.sum(axis=1), 1.0, atol=1e-2):
            # Apply softmax
            exp_p = np.exp(y_pred_probs - np.max(y_pred_probs, axis=1, keepdims=True))
            y_probs = exp_p / exp_p.sum(axis=1, keepdims=True)
        else:
            y_probs = y_pred_probs
        y_pred = np.argmax(y_probs, axis=1)

    acc = float(accuracy_score(y_true, y_pred))
    bacc = float(balanced_accuracy_score(y_true, y_pred))
    prec_macro = float(precision_score(y_true, y_pred, average='macro', zero_division=0))
    rec_macro = float(recall_score(y_true, y_pred, average='macro', zero_division=0))
    f1_macro = float(f1_score(y_true, y_pred, average='macro', zero_division=0))
    qwk = float(cohen_kappa_score(y_true, y_pred, weights='quadratic'))
    ord_mae = float(np.mean(np.abs(y_true - y_pred)))
    
    # Per-class metrics
    prec_per_class = [float(x) for x in precision_score(y_true, y_pred, average=None, labels=list(range(num_classes)), zero_division=0)]
    rec_per_class = [float(x) for x in recall_score(y_true, y_pred, average=None, labels=list(range(num_classes)), zero_division=0)]
    f1_per_class = [float(x) for x in f1_score(y_true, y_pred, average=None, labels=list(range(num_classes)), zero_division=0)]
    cm = confusion_matrix(y_true, y_pred, labels=list(range(num_classes))).tolist()
    
    # One-vs-Rest Macro AUROC
    try:
        y_true_oh = np.eye(num_classes)[y_true]
        auroc = float(roc_auc_score(y_true_oh, y_probs, multi_class='ovr', average='macro'))
    except Exception:
        auroc = 0.5
        
    # Brier Score (multi-class mean squared probability error)
    y_true_oh = np.eye(num_classes)[y_true]
    brier = float(np.mean(np.sum((y_probs - y_true_oh) ** 2, axis=1)))
    
    # Expected Calibration Error (ECE, 10 bins)
    confidences = np.max(y_probs, axis=1)
    accuracies = (y_pred == y_true).astype(float)
    bin_boundaries = np.linspace(0, 1, 11)
    ece = 0.0
    for i in range(10):
        bin_lower = bin_boundaries[i]
        bin_upper = bin_boundaries[i + 1]
        in_bin = (confidences > bin_lower) & (confidences <= bin_upper)
        prop_in_bin = np.mean(in_bin)
        if prop_in_bin > 0:
            accuracy_in_bin = np.mean(accuracies[in_bin])
            avg_confidence_in_bin = np.mean(confidences[in_bin])
            ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin
    ece = float(ece)
    
    return {
        'accuracy': acc,
        'balanced_accuracy': bacc,
        'macro_precision': prec_macro,
        'macro_recall': rec_macro,
        'macro_f1': f1_macro,
        'quadratic_weighted_kappa': qwk,
        'ordinal_mae': ord_mae,
        'macro_auroc_ovr': auroc,
        'brier_score': brier,
        'ece_10_bins': ece,
        'per_class_precision': prec_per_class,
        'per_class_recall': rec_per_class,
        'per_class_f1': f1_per_class,
        'confusion_matrix': cm
    }
