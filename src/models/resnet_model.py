"""ResNet-based model architecture."""

import torch
import torch.nn as nn
from torchvision import models
from typing import Optional


class ResNetClassifier(nn.Module):
    """
    ResNet-50 based classifier with transfer learning.
    
    Features:
    - Pretrained ResNet-50 backbone from ImageNet
    - Custom classification head
    - Optional backbone freezing for fine-tuning
    """
    
    def __init__(
        self,
        num_classes: int = 10,
        pretrained: bool = True,
        freeze_backbone: bool = False,
        dropout_rate: float = 0.5,
    ):
        """
        Initialize ResNet classifier.
        
        Args:
            num_classes: Number of output classes
            pretrained: Whether to use pretrained weights
            freeze_backbone: Whether to freeze backbone layers
            dropout_rate: Dropout rate (not used in default ResNet head)
        """
        super(ResNetClassifier, self).__init__()
        
        self.num_classes = num_classes
        
        # Load pretrained ResNet-50
        if pretrained:
            weights = models.ResNet50_Weights.DEFAULT
            self.backbone = models.resnet50(weights=weights)
        else:
            self.backbone = models.resnet50(weights=None)
        
        # Freeze backbone if specified
        if freeze_backbone:
            self._freeze_backbone()
        
        # Replace final fully connected layer
        num_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Linear(num_features, num_classes)
    
    def _freeze_backbone(self):
        """Freeze all layers except the final classification layer."""
        for name, param in self.backbone.named_parameters():
            if "fc" not in name:  # Don't freeze final layer
                param.requires_grad = False
    
    def unfreeze_backbone(self, num_layers: Optional[int] = None):
        """
        Unfreeze backbone layers for fine-tuning.
        
        Args:
            num_layers: Number of layers to unfreeze from the end (None = all)
        """
        if num_layers is None:
            # Unfreeze all layers
            for param in self.backbone.parameters():
                param.requires_grad = True
        else:
            # Unfreeze last N layers
            layers = list(self.backbone.children())
            for layer in layers[-num_layers:]:
                for param in layer.parameters():
                    param.requires_grad = True
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        return self.backbone(x)
    
    def get_feature_extractor(self) -> nn.Module:
        """
        Get the feature extractor (backbone without classification head).
        
        Returns:
            Feature extractor module
        """
        return nn.Sequential(*list(self.backbone.children())[:-1])


class ResNetWithProjection(nn.Module):
    """
    ResNet with projection head for metric learning.
    
    Useful for triplet loss and other metric learning approaches.
    """
    
    def __init__(
        self,
        num_classes: int = 10,
        embedding_dim: int = 128,
        pretrained: bool = True,
    ):
        """
        Initialize ResNet with projection head.
        
        Args:
            num_classes: Number of output classes
            embedding_dim: Dimension of embedding space
            pretrained: Whether to use pretrained weights
        """
        super(ResNetWithProjection, self).__init__()
        
        # Load pretrained ResNet-50
        if pretrained:
            weights = models.ResNet50_Weights.DEFAULT
            backbone = models.resnet50(weights=weights)
        else:
            backbone = models.resnet50(weights=None)
        
        # Remove final classification layer
        self.feature_extractor = nn.Sequential(*list(backbone.children())[:-1])
        
        # Get feature dimension
        num_features = backbone.fc.in_features
        
        # Projection head for embeddings
        self.projection = nn.Sequential(
            nn.Flatten(),
            nn.Linear(num_features, embedding_dim),
            nn.ReLU(),
        )
        
        # Classification head
        self.classifier = nn.Linear(embedding_dim, num_classes)
    
    def forward(self, x: torch.Tensor, return_embedding: bool = False) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input tensor
            return_embedding: Whether to return embedding instead of logits
            
        Returns:
            Classification logits or embeddings
        """
        features = self.feature_extractor(x)
        embeddings = self.projection(features)
        
        if return_embedding:
            return embeddings
        
        logits = self.classifier(embeddings)
        return logits

