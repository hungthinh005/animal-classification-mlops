"""Model factory for creating different model architectures."""

import torch.nn as nn
from typing import Dict, Any
from .cnn_models import CNNDualConv, CNNSingleConv
from .resnet_model import ResNetClassifier, ResNetWithProjection


def create_model(config: Dict[str, Any], use_metric_learning: bool = False) -> nn.Module:
    """
    Create model based on configuration.
    
    Args:
        config: Configuration dictionary
        use_metric_learning: Whether to use metric learning variant
        
    Returns:
        PyTorch model
        
    Raises:
        ValueError: If architecture is not supported
    """
    model_config = config["model"]
    architecture = model_config["architecture"]
    num_classes = config["data"]["num_classes"]
    dropout_rate = model_config.get("dropout_rate", 0.5)
    
    if architecture == "cnn_dual":
        model = CNNDualConv(
            num_classes=num_classes,
            dropout_rate=dropout_rate,
        )
    
    elif architecture == "cnn_single":
        model = CNNSingleConv(
            num_classes=num_classes,
            dropout_rate=dropout_rate,
        )
    
    elif architecture == "resnet50":
        pretrained = model_config.get("pretrained", True)
        freeze_backbone = model_config.get("freeze_backbone", False)
        
        if use_metric_learning:
            embedding_dim = config["training"].get("embedding_dim", 128)
            model = ResNetWithProjection(
                num_classes=num_classes,
                embedding_dim=embedding_dim,
                pretrained=pretrained,
            )
        else:
            model = ResNetClassifier(
                num_classes=num_classes,
                pretrained=pretrained,
                freeze_backbone=freeze_backbone,
                dropout_rate=dropout_rate,
            )
    
    else:
        raise ValueError(
            f"Unsupported architecture: {architecture}. "
            "Choose from: 'cnn_dual', 'cnn_single', 'resnet50'"
        )
    
    return model


def load_model(
    model_path: str,
    config: Dict[str, Any],
    device: str = "cpu",
    use_metric_learning: bool = False,
) -> nn.Module:
    """
    Load trained model from checkpoint.
    
    Args:
        model_path: Path to model checkpoint
        config: Configuration dictionary
        device: Device to load model on
        use_metric_learning: Whether model uses metric learning
        
    Returns:
        Loaded model
    """
    import torch
    
    # Create model architecture
    model = create_model(config, use_metric_learning=use_metric_learning)
    
    # Load weights
    checkpoint = torch.load(model_path, map_location=device)
    
    # Handle different checkpoint formats
    if isinstance(checkpoint, dict):
        if "model_state_dict" in checkpoint:
            model.load_state_dict(checkpoint["model_state_dict"])
        elif "state_dict" in checkpoint:
            model.load_state_dict(checkpoint["state_dict"])
        else:
            model.load_state_dict(checkpoint)
    else:
        model.load_state_dict(checkpoint)
    
    model.to(device)
    model.eval()
    
    return model


def save_model(
    model: nn.Module,
    save_path: str,
    optimizer: Any = None,
    epoch: int = None,
    metrics: Dict[str, float] = None,
) -> None:
    """
    Save model checkpoint.
    
    Args:
        model: PyTorch model
        save_path: Path to save checkpoint
        optimizer: Optimizer state (optional)
        epoch: Current epoch (optional)
        metrics: Metrics dictionary (optional)
    """
    import torch
    from pathlib import Path
    
    # Ensure directory exists
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    
    # Create checkpoint dictionary
    checkpoint = {
        "model_state_dict": model.state_dict(),
    }
    
    if optimizer is not None:
        checkpoint["optimizer_state_dict"] = optimizer.state_dict()
    
    if epoch is not None:
        checkpoint["epoch"] = epoch
    
    if metrics is not None:
        checkpoint["metrics"] = metrics
    
    # Save checkpoint
    torch.save(checkpoint, save_path)

