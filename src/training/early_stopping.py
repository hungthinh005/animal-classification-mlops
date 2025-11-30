"""Early stopping implementation."""

import numpy as np


class EarlyStopping:
    """
    Early stopping to stop training when validation loss doesn't improve.
    
    Attributes:
        patience: Number of epochs to wait before stopping
        min_delta: Minimum change to qualify as improvement
        counter: Current count of epochs without improvement
        best_score: Best validation score so far
        early_stop: Whether to stop training
        val_loss_min: Minimum validation loss achieved
    """
    
    def __init__(
        self,
        patience: int = 10,
        min_delta: float = 0.001,
        monitor: str = "val_loss",
        mode: str = "min",
    ):
        """
        Initialize early stopping.
        
        Args:
            patience: Number of epochs to wait for improvement
            min_delta: Minimum change to qualify as improvement
            monitor: Metric to monitor ('val_loss' or 'val_accuracy')
            mode: 'min' for loss, 'max' for accuracy
        """
        self.patience = patience
        self.min_delta = min_delta
        self.monitor = monitor
        self.mode = mode
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.val_accuracy_max = 0
    
    def __call__(self, val_loss: float, val_accuracy: float, model, save_path: str = None):
        """
        Check if training should stop.
        
        Args:
            val_loss: Current validation loss
            val_accuracy: Current validation accuracy
            model: Model to save if improved
            save_path: Path to save best model
        """
        # Determine score based on monitor
        if self.monitor == "val_loss":
            score = -val_loss
        elif self.monitor == "val_accuracy":
            score = val_accuracy
        else:
            score = -val_loss
        
        # First epoch
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, val_accuracy, model, save_path)
        
        # Check if improved
        elif score < self.best_score + self.min_delta:
            self.counter += 1
            print(f"EarlyStopping counter: {self.counter}/{self.patience}")
            
            if self.counter >= self.patience:
                self.early_stop = True
        
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, val_accuracy, model, save_path)
            self.counter = 0
    
    def save_checkpoint(
        self,
        val_loss: float,
        val_accuracy: float,
        model,
        save_path: str = None,
    ):
        """
        Save model when validation loss/accuracy improves.
        
        Args:
            val_loss: Validation loss
            val_accuracy: Validation accuracy
            model: Model to save
            save_path: Path to save model
        """
        if self.monitor == "val_loss":
            if val_loss < self.val_loss_min:
                print(
                    f"Validation loss decreased ({self.val_loss_min:.6f} -> {val_loss:.6f})"
                )
                self.val_loss_min = val_loss
                if save_path:
                    import torch
                    torch.save(model.state_dict(), save_path)
        
        elif self.monitor == "val_accuracy":
            if val_accuracy > self.val_accuracy_max:
                print(
                    f"Validation accuracy increased "
                    f"({self.val_accuracy_max:.6f} -> {val_accuracy:.6f})"
                )
                self.val_accuracy_max = val_accuracy
                if save_path:
                    import torch
                    torch.save(model.state_dict(), save_path)

