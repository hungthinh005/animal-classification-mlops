"""Model evaluation utilities."""

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from torch.utils.data import DataLoader
from typing import Dict, List, Tuple, Optional
from sklearn.metrics import ConfusionMatrixDisplay

from ..utils.metrics import (
    calculate_metrics,
    get_confusion_matrix,
    get_classification_report,
)


class Evaluator:
    """Model evaluator class."""
    
    def __init__(
        self,
        model: nn.Module,
        device: str = "cuda",
        class_names: Optional[List[str]] = None,
    ):
        """
        Initialize evaluator.
        
        Args:
            model: Trained model
            device: Device to run evaluation on
            class_names: List of class names
        """
        self.model = model.to(device)
        self.device = device
        self.class_names = class_names
        self.model.eval()
    
    def evaluate(
        self,
        data_loader: DataLoader,
        loss_fn: Optional[nn.Module] = None,
    ) -> Dict[str, float]:
        """
        Evaluate model on dataset.
        
        Args:
            data_loader: Data loader
            loss_fn: Loss function (optional)
            
        Returns:
            Dictionary of evaluation metrics
        """
        all_predictions = []
        all_labels = []
        total_loss = 0.0
        
        with torch.no_grad():
            for inputs, targets in data_loader:
                inputs = inputs.to(self.device)
                targets = targets.to(self.device)
                
                # Forward pass
                outputs = self.model(inputs)
                
                # Calculate loss if loss_fn provided
                if loss_fn is not None:
                    loss = loss_fn(outputs, targets)
                    total_loss += loss.item() * inputs.size(0)
                
                # Get predictions
                _, predicted = torch.max(outputs, 1)
                
                all_predictions.extend(predicted.cpu().numpy())
                all_labels.extend(targets.cpu().numpy())
        
        # Convert to numpy arrays
        y_true = np.array(all_labels)
        y_pred = np.array(all_predictions)
        
        # Calculate metrics
        metrics = calculate_metrics(y_true, y_pred, self.class_names)
        
        # Add average loss if calculated
        if loss_fn is not None:
            metrics["loss"] = total_loss / len(data_loader.dataset)
        
        return metrics
    
    def get_predictions(
        self,
        data_loader: DataLoader,
        return_probabilities: bool = False,
    ) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
        """
        Get predictions for dataset.
        
        Args:
            data_loader: Data loader
            return_probabilities: Whether to return class probabilities
            
        Returns:
            Tuple of (predictions, true_labels, probabilities)
        """
        all_predictions = []
        all_labels = []
        all_probabilities = []
        
        with torch.no_grad():
            for inputs, targets in data_loader:
                inputs = inputs.to(self.device)
                outputs = self.model(inputs)
                
                # Get predictions
                _, predicted = torch.max(outputs, 1)
                all_predictions.extend(predicted.cpu().numpy())
                all_labels.extend(targets.cpu().numpy())
                
                # Get probabilities if requested
                if return_probabilities:
                    probs = torch.softmax(outputs, dim=1)
                    all_probabilities.extend(probs.cpu().numpy())
        
        predictions = np.array(all_predictions)
        labels = np.array(all_labels)
        probabilities = np.array(all_probabilities) if return_probabilities else None
        
        return predictions, labels, probabilities
    
    def plot_confusion_matrix(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        save_path: Optional[str] = None,
        normalize: bool = False,
    ) -> plt.Figure:
        """
        Plot confusion matrix.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            save_path: Path to save figure
            normalize: Whether to normalize the confusion matrix
            
        Returns:
            Matplotlib figure
        """
        cm = get_confusion_matrix(y_true, y_pred)
        
        if normalize:
            cm = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]
            fmt = ".2f"
        else:
            fmt = "d"
        
        fig, ax = plt.subplots(figsize=(10, 8))
        
        disp = ConfusionMatrixDisplay(
            confusion_matrix=cm,
            display_labels=self.class_names,
        )
        disp.plot(ax=ax, cmap="Blues", values_format=fmt)
        
        plt.title("Confusion Matrix")
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
        
        return fig
    
    def plot_class_accuracies(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        save_path: Optional[str] = None,
    ) -> plt.Figure:
        """
        Plot per-class accuracies.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            save_path: Path to save figure
            
        Returns:
            Matplotlib figure
        """
        from sklearn.metrics import precision_recall_fscore_support
        
        # Calculate per-class metrics
        precision, recall, f1, _ = precision_recall_fscore_support(
            y_true, y_pred, average=None
        )
        
        # Create plot
        fig, ax = plt.subplots(figsize=(12, 6))
        
        x = np.arange(len(self.class_names))
        width = 0.25
        
        ax.bar(x - width, precision, width, label="Precision", alpha=0.8)
        ax.bar(x, recall, width, label="Recall", alpha=0.8)
        ax.bar(x + width, f1, width, label="F1-Score", alpha=0.8)
        
        ax.set_xlabel("Classes")
        ax.set_ylabel("Score")
        ax.set_title("Per-Class Metrics")
        ax.set_xticks(x)
        ax.set_xticklabels(self.class_names, rotation=45, ha="right")
        ax.legend()
        ax.grid(axis="y", alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
        
        return fig
    
    def print_classification_report(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
    ) -> str:
        """
        Print classification report.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            
        Returns:
            Classification report string
        """
        report = get_classification_report(y_true, y_pred, self.class_names)
        print(report)
        return report
    
    def visualize_predictions(
        self,
        data_loader: DataLoader,
        num_samples: int = 16,
        save_path: Optional[str] = None,
    ) -> plt.Figure:
        """
        Visualize model predictions on sample images.
        
        Args:
            data_loader: Data loader
            num_samples: Number of samples to visualize
            save_path: Path to save figure
            
        Returns:
            Matplotlib figure
        """
        # Get a batch of data
        images, labels = next(iter(data_loader))
        images = images[:num_samples]
        labels = labels[:num_samples]
        
        # Get predictions
        with torch.no_grad():
            outputs = self.model(images.to(self.device))
            _, predicted = torch.max(outputs, 1)
            probabilities = torch.softmax(outputs, dim=1)
        
        # Create plot
        rows = int(np.sqrt(num_samples))
        cols = int(np.ceil(num_samples / rows))
        
        fig, axes = plt.subplots(rows, cols, figsize=(15, 15))
        axes = axes.flatten()
        
        for idx in range(num_samples):
            ax = axes[idx]
            
            # Denormalize image for display
            img = images[idx].cpu().numpy().transpose(1, 2, 0)
            img = (img - img.min()) / (img.max() - img.min())
            
            ax.imshow(img)
            
            true_label = self.class_names[labels[idx]]
            pred_label = self.class_names[predicted[idx]]
            prob = probabilities[idx][predicted[idx]].item()
            
            color = "green" if labels[idx] == predicted[idx] else "red"
            ax.set_title(
                f"True: {true_label}\nPred: {pred_label} ({prob:.2f})",
                color=color,
            )
            ax.axis("off")
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
        
        return fig

