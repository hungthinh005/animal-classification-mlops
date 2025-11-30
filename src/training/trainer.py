"""Training module with MLflow tracking."""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from typing import Dict, Any, Optional, Tuple, List
from tqdm import tqdm
import mlflow
import mlflow.pytorch

from pytorch_metric_learning import losses, miners
from ..utils.metrics import compute_accuracy
from .early_stopping import EarlyStopping


class Trainer:
    """
    Trainer class for model training with MLflow tracking.
    """
    
    def __init__(
        self,
        model: nn.Module,
        config: Dict[str, Any],
        device: str = "cuda",
    ):
        """
        Initialize trainer.
        
        Args:
            model: PyTorch model
            config: Configuration dictionary
            device: Device to train on
        """
        self.model = model.to(device)
        self.config = config
        self.device = device
        
        # Setup loss functions
        self.classification_loss = nn.CrossEntropyLoss()
        
        # Metric learning loss (optional)
        loss_config = config["training"]["loss"]
        if loss_config.get("metric_learning") == "triplet_margin":
            margin = loss_config.get("triplet_margin", 0.2)
            self.metric_loss = losses.TripletMarginLoss(margin=margin)
            self.miner = miners.MultiSimilarityMiner()
        else:
            self.metric_loss = None
            self.miner = None
        
        # Loss weights
        self.weight_ce = loss_config.get("loss_weight_ce", 1.0)
        self.weight_triplet = loss_config.get("loss_weight_triplet", 1.0)
        
        # Setup optimizer
        optimizer_config = config["training"]["optimizer"]
        if optimizer_config["type"] == "adam":
            self.optimizer = optim.Adam(
                model.parameters(),
                lr=config["training"]["learning_rate"],
                weight_decay=config["training"].get("weight_decay", 0),
                betas=optimizer_config.get("betas", (0.9, 0.999)),
                eps=optimizer_config.get("eps", 1e-8),
            )
        else:
            self.optimizer = optim.SGD(
                model.parameters(),
                lr=config["training"]["learning_rate"],
                weight_decay=config["training"].get("weight_decay", 0),
                momentum=0.9,
            )
        
        # Setup learning rate scheduler
        scheduler_config = config["training"]["scheduler"]
        if scheduler_config["type"] == "reduce_on_plateau":
            self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                mode=scheduler_config.get("mode", "min"),
                factor=scheduler_config.get("factor", 0.5),
                patience=scheduler_config.get("patience", 5),
                min_lr=scheduler_config.get("min_lr", 1e-7),
            )
        elif scheduler_config["type"] == "step":
            self.scheduler = optim.lr_scheduler.StepLR(
                self.optimizer,
                step_size=scheduler_config.get("step_size", 10),
                gamma=scheduler_config.get("gamma", 0.1),
            )
        else:
            self.scheduler = None
        
        # Setup early stopping
        if config["training"]["early_stopping"]["enabled"]:
            self.early_stopping = EarlyStopping(
                patience=config["training"]["early_stopping"]["patience"],
                min_delta=config["training"]["early_stopping"]["min_delta"],
                monitor=config["training"]["early_stopping"]["monitor"],
            )
        else:
            self.early_stopping = None
        
        # Training history
        self.history = {
            "train_loss": [],
            "train_accuracy": [],
            "val_loss": [],
            "val_accuracy": [],
            "learning_rates": [],
        }
    
    def train_epoch(self, train_loader: DataLoader) -> Tuple[float, float]:
        """
        Train for one epoch.
        
        Args:
            train_loader: Training data loader
            
        Returns:
            Tuple of (average_loss, accuracy)
        """
        self.model.train()
        
        total_loss = 0.0
        correct = 0
        total = 0
        
        pbar = tqdm(train_loader, desc="Training")
        for inputs, targets in pbar:
            inputs = inputs.to(self.device)
            targets = targets.to(self.device)
            
            # Zero gradients
            self.optimizer.zero_grad()
            
            # Forward pass
            outputs = self.model(inputs)
            
            # Calculate loss
            class_loss = self.classification_loss(outputs, targets)
            
            # Add metric learning loss if enabled
            if self.metric_loss is not None and self.miner is not None:
                hard_pairs = self.miner(outputs, targets)
                metric_l = self.metric_loss(outputs, targets, hard_pairs)
                loss = self.weight_ce * class_loss + self.weight_triplet * metric_l
            else:
                loss = class_loss
            
            # Backward pass
            loss.backward()
            self.optimizer.step()
            
            # Track metrics
            total_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs, 1)
            total += targets.size(0)
            correct += (predicted == targets).sum().item()
            
            # Update progress bar
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})
        
        avg_loss = total_loss / len(train_loader.dataset)
        accuracy = correct / total
        
        return avg_loss, accuracy
    
    def validate(self, val_loader: DataLoader) -> Tuple[float, float]:
        """
        Validate model.
        
        Args:
            val_loader: Validation data loader
            
        Returns:
            Tuple of (average_loss, accuracy)
        """
        self.model.eval()
        
        total_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            pbar = tqdm(val_loader, desc="Validation")
            for inputs, targets in pbar:
                inputs = inputs.to(self.device)
                targets = targets.to(self.device)
                
                # Forward pass
                outputs = self.model(inputs)
                
                # Calculate loss
                loss = self.classification_loss(outputs, targets)
                
                # Track metrics
                total_loss += loss.item() * inputs.size(0)
                _, predicted = torch.max(outputs, 1)
                total += targets.size(0)
                correct += (predicted == targets).sum().item()
                
                # Update progress bar
                pbar.set_postfix({"loss": f"{loss.item():.4f}"})
        
        avg_loss = total_loss / len(val_loader.dataset)
        accuracy = correct / total
        
        return avg_loss, accuracy
    
    def train(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        epochs: int,
        checkpoint_path: Optional[str] = "models/best_model.pth",
        use_mlflow: bool = True,
    ) -> Dict[str, List[float]]:
        """
        Train model for multiple epochs.
        
        Args:
            train_loader: Training data loader
            val_loader: Validation data loader
            epochs: Number of epochs
            checkpoint_path: Path to save best model
            use_mlflow: Whether to use MLflow tracking
            
        Returns:
            Training history dictionary
        """
        # Setup MLflow
        if use_mlflow:
            mlflow.set_tracking_uri(self.config["mlflow"]["tracking_uri"])
            mlflow.set_experiment(self.config["mlflow"]["experiment_name"])
            
            with mlflow.start_run():
                # Log parameters
                mlflow.log_params({
                    "model": self.config["model"]["architecture"],
                    "batch_size": self.config["training"]["batch_size"],
                    "learning_rate": self.config["training"]["learning_rate"],
                    "epochs": epochs,
                    "optimizer": self.config["training"]["optimizer"]["type"],
                })
                
                return self._train_loop(
                    train_loader, val_loader, epochs, checkpoint_path, use_mlflow=True
                )
        else:
            return self._train_loop(
                train_loader, val_loader, epochs, checkpoint_path, use_mlflow=False
            )
    
    def _train_loop(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        epochs: int,
        checkpoint_path: Optional[str],
        use_mlflow: bool = False,
    ) -> Dict[str, List[float]]:
        """Internal training loop."""
        
        for epoch in range(1, epochs + 1):
            print(f"\nEpoch {epoch}/{epochs}")
            print("-" * 50)
            
            # Train
            train_loss, train_acc = self.train_epoch(train_loader)
            
            # Validate
            val_loss, val_acc = self.validate(val_loader)
            
            # Get current learning rate
            current_lr = self.optimizer.param_groups[0]["lr"]
            
            # Update history
            self.history["train_loss"].append(train_loss)
            self.history["train_accuracy"].append(train_acc)
            self.history["val_loss"].append(val_loss)
            self.history["val_accuracy"].append(val_acc)
            self.history["learning_rates"].append(current_lr)
            
            # Print metrics
            print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}")
            print(f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}")
            print(f"Learning Rate: {current_lr:.6f}")
            
            # Log to MLflow
            if use_mlflow:
                mlflow.log_metrics({
                    "train_loss": train_loss,
                    "train_accuracy": train_acc,
                    "val_loss": val_loss,
                    "val_accuracy": val_acc,
                    "learning_rate": current_lr,
                }, step=epoch)
            
            # Update learning rate
            if self.scheduler is not None:
                if isinstance(self.scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(val_loss)
                else:
                    self.scheduler.step()
            
            # Early stopping
            if self.early_stopping is not None:
                self.early_stopping(val_loss, val_acc, self.model, checkpoint_path)
                
                if self.early_stopping.early_stop:
                    print(f"\nEarly stopping triggered at epoch {epoch}")
                    break
        
        # Log model to MLflow
        if use_mlflow:
            mlflow.pytorch.log_model(self.model, "model")
        
        return self.history

