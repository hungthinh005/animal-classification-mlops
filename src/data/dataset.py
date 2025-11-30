"""Dataset and data loading utilities."""

import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
from typing import Tuple, Optional, Dict, Any
from PIL import Image


class ConvertToRGB:
    """Convert image to RGB mode."""
    
    def __call__(self, img: Image.Image) -> Image.Image:
        """Convert image to RGB if not already."""
        if img.mode != "RGB":
            img = img.convert("RGB")
        return img


class AnimalDataset(datasets.ImageFolder):
    """
    Custom dataset for animal images.
    Extends ImageFolder with additional functionality.
    """
    
    def __init__(
        self,
        root: str,
        transform: Optional[transforms.Compose] = None,
        num_classes: Optional[int] = None,
    ):
        """
        Initialize dataset.
        
        Args:
            root: Root directory path
            transform: Image transformations
            num_classes: Number of classes to use (optional)
        """
        super().__init__(root=root, transform=transform)
        
        # Filter to num_classes if specified
        if num_classes is not None:
            self.samples = [
                (path, label)
                for path, label in self.samples
                if label < num_classes
            ]
            self.targets = [label for _, label in self.samples]
            self.classes = self.classes[:num_classes]
            self.class_to_idx = {
                cls: idx for cls, idx in self.class_to_idx.items() if idx < num_classes
            }
    
    def get_class_distribution(self) -> Dict[str, int]:
        """Get distribution of classes in dataset."""
        from collections import Counter
        label_counts = Counter(self.targets)
        return {self.classes[label]: count for label, count in label_counts.items()}


def get_transforms(
    config: Dict[str, Any],
    is_training: bool = True,
    compute_stats: bool = False,
) -> transforms.Compose:
    """
    Get image transformations based on configuration.
    
    Args:
        config: Configuration dictionary
        is_training: Whether transforms are for training
        compute_stats: Whether to compute mean/std (no normalization)
        
    Returns:
        Composed transforms
    """
    image_size = tuple(config["data"]["image_size"])
    
    transform_list = [
        ConvertToRGB(),
        transforms.Resize(image_size),
        transforms.ToTensor(),
    ]
    
    if is_training and not compute_stats:
        # Training augmentations
        aug_config = config["augmentation"]["train"]
        
        if aug_config.get("horizontal_flip", False):
            transform_list.insert(-1, transforms.RandomHorizontalFlip())
        
        rotation = aug_config.get("rotation_degrees", 0)
        if rotation > 0:
            transform_list.insert(-1, transforms.RandomRotation(rotation))
        
        color_jitter = aug_config.get("color_jitter")
        if color_jitter:
            transform_list.insert(
                -1,
                transforms.ColorJitter(
                    brightness=color_jitter.get("brightness", 0),
                    contrast=color_jitter.get("contrast", 0),
                    saturation=color_jitter.get("saturation", 0),
                ),
            )
    
    # Add normalization if not computing stats
    if not compute_stats:
        # Use ImageNet stats as default or custom stats if available
        mean = config.get("normalization", {}).get("mean", [0.485, 0.456, 0.406])
        std = config.get("normalization", {}).get("std", [0.229, 0.224, 0.225])
        
        if config["augmentation"]["train" if is_training else "test"].get("normalize", True):
            transform_list.append(transforms.Normalize(mean=mean, std=std))
    
    return transforms.Compose(transform_list)


def compute_dataset_statistics(data_loader: DataLoader) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute mean and standard deviation of dataset.
    
    Args:
        data_loader: DataLoader for the dataset
        
    Returns:
        Tuple of (mean, std) tensors
    """
    from tqdm import tqdm
    
    channels_sum, channels_squared_sum, num_batches = 0, 0, 0
    
    for data, _ in tqdm(data_loader, desc="Computing statistics"):
        channels_sum += torch.mean(data, dim=[0, 2, 3])
        channels_squared_sum += torch.mean(data**2, dim=[0, 2, 3])
        num_batches += 1
    
    mean = channels_sum / num_batches
    std = (channels_squared_sum / num_batches - mean**2) ** 0.5
    
    return mean, std


def get_dataloaders(
    config: Dict[str, Any],
    train_dataset,
    val_dataset,
    test_dataset: Optional[Any] = None,
) -> Tuple[DataLoader, DataLoader, Optional[DataLoader]]:
    """
    Create data loaders for train, validation, and test sets.
    
    Args:
        config: Configuration dictionary
        train_dataset: Training dataset
        val_dataset: Validation dataset
        test_dataset: Test dataset (optional)
        
    Returns:
        Tuple of (train_loader, val_loader, test_loader)
    """
    batch_size = config["training"]["batch_size"]
    num_workers = config["training"].get("num_workers", 0)
    
    # Set random seed
    g = torch.Generator()
    g.manual_seed(config["project"]["random_seed"])
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        generator=g,
        pin_memory=True,
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    
    test_loader = None
    if test_dataset is not None:
        test_loader = DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
        )
    
    return train_loader, val_loader, test_loader

