"""Custom CNN model architectures."""

import torch
import torch.nn as nn
from typing import List


class CNNDualConv(nn.Module):
    """
    Dual-convolution CNN architecture (VGG-style).
    
    Features:
    - Multiple blocks with two Conv2d layers each
    - BatchNorm2d for stable training
    - MaxPool2d for downsampling
    - Dropout for regularization
    """
    
    def __init__(
        self,
        num_classes: int = 10,
        input_channels: int = 3,
        hidden_dims: List[int] = [8, 16, 32],
        dropout_rate: float = 0.5,
    ):
        """
        Initialize CNNDualConv model.
        
        Args:
            num_classes: Number of output classes
            input_channels: Number of input channels (3 for RGB)
            hidden_dims: List of hidden dimensions for conv blocks
            dropout_rate: Dropout rate
        """
        super(CNNDualConv, self).__init__()
        
        self.num_classes = num_classes
        
        # Build convolutional blocks
        layers = []
        in_channels = input_channels
        
        for out_channels in hidden_dims:
            layers.append(self._make_conv_block(in_channels, out_channels))
            in_channels = out_channels
        
        self.conv_layers = nn.Sequential(*layers)
        
        # Calculate the size after convolutions (224 -> 28 after 3 blocks)
        # Each MaxPool2d reduces size by half
        feature_size = 224 // (2 ** len(hidden_dims))
        flattened_size = hidden_dims[-1] * feature_size * feature_size
        
        # Fully connected layers
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(dropout_rate),
            nn.Linear(flattened_size, 512),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(512, num_classes),
        )
    
    def _make_conv_block(self, in_channels: int, out_channels: int) -> nn.Sequential:
        """Create a convolutional block with two conv layers."""
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding="same"),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding="same"),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        x = self.conv_layers(x)
        x = self.classifier(x)
        return x


class CNNSingleConv(nn.Module):
    """
    Single-convolution CNN architecture.
    
    Features:
    - Simple architecture with one conv layer per block
    - ReLU activation
    - MaxPool2d for downsampling
    - Minimal regularization
    """
    
    def __init__(
        self,
        num_classes: int = 10,
        input_channels: int = 3,
        hidden_dims: List[int] = [16, 32, 64],
        dropout_rate: float = 0.5,
    ):
        """
        Initialize CNNSingleConv model.
        
        Args:
            num_classes: Number of output classes
            input_channels: Number of input channels (3 for RGB)
            hidden_dims: List of hidden dimensions for conv blocks
            dropout_rate: Dropout rate
        """
        super(CNNSingleConv, self).__init__()
        
        self.num_classes = num_classes
        
        # Build convolutional layers
        layers = []
        in_channels = input_channels
        
        for out_channels in hidden_dims:
            layers.extend(
                [
                    nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
                    nn.ReLU(),
                    nn.MaxPool2d(kernel_size=2, stride=2),
                ]
            )
            in_channels = out_channels
        
        self.conv_layers = nn.Sequential(*layers)
        
        # Calculate the size after convolutions
        feature_size = 224 // (2 ** len(hidden_dims))
        flattened_size = hidden_dims[-1] * feature_size * feature_size
        
        # Fully connected layers
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(dropout_rate),
            nn.Linear(flattened_size, 500),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(500, num_classes),
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        x = self.conv_layers(x)
        x = self.classifier(x)
        return x


def get_model_summary(model: nn.Module, input_size: tuple = (1, 3, 224, 224)) -> str:
    """
    Get model summary.
    
    Args:
        model: PyTorch model
        input_size: Input tensor size
        
    Returns:
        Model summary string
    """
    try:
        from torchinfo import summary
        return str(summary(model, input_size=input_size, verbose=0))
    except ImportError:
        # Fallback if torchinfo not available
        return str(model)

