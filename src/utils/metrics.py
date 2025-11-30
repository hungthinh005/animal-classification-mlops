"""Metrics calculation utilities."""

import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
    classification_report,
)
from typing import Dict, Tuple, List, Optional


def calculate_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: Optional[List[str]] = None,
) -> Dict[str, float]:
    """
    Calculate classification metrics.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        class_names: List of class names (optional)
        
    Returns:
        Dictionary containing various metrics
    """
    # Basic metrics
    accuracy = accuracy_score(y_true, y_pred)
    
    # Precision, recall, F1 (weighted average)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="weighted"
    )
    
    # Per-class metrics
    precision_per_class, recall_per_class, f1_per_class, support = (
        precision_recall_fscore_support(y_true, y_pred, average=None)
    )
    
    metrics = {
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "f1_score": float(f1),
    }
    
    # Add per-class metrics if class names provided
    if class_names is not None:
        for idx, class_name in enumerate(class_names):
            metrics[f"precision_{class_name}"] = float(precision_per_class[idx])
            metrics[f"recall_{class_name}"] = float(recall_per_class[idx])
            metrics[f"f1_{class_name}"] = float(f1_per_class[idx])
            metrics[f"support_{class_name}"] = int(support[idx])
    
    return metrics


def get_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> np.ndarray:
    """
    Calculate confusion matrix.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        
    Returns:
        Confusion matrix
    """
    return confusion_matrix(y_true, y_pred)


def get_classification_report(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: Optional[List[str]] = None,
) -> str:
    """
    Generate classification report.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        class_names: List of class names (optional)
        
    Returns:
        Classification report string
    """
    return classification_report(y_true, y_pred, target_names=class_names)


def compute_accuracy(outputs: torch.Tensor, labels: torch.Tensor) -> float:
    """
    Compute accuracy from model outputs and labels.
    
    Args:
        outputs: Model outputs (logits)
        labels: True labels
        
    Returns:
        Accuracy value
    """
    _, predicted = torch.max(outputs, 1)
    correct = (predicted == labels).sum().item()
    total = labels.size(0)
    return correct / total

