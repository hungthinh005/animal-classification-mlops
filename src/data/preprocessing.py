"""Data preprocessing utilities."""

import os
import shutil
import random
import numpy as np
from pathlib import Path
from typing import Tuple, List
from sklearn.model_selection import train_test_split


def prepare_data(
    source_dir: str,
    target_dir: str,
    test_ratio: float = 0.2,
    random_state: int = 42,
    num_classes: int = 10,
) -> Tuple[str, str]:
    """
    Prepare data by splitting into train and test sets.
    
    Args:
        source_dir: Source directory containing class folders
        target_dir: Target directory for train/test split
        test_ratio: Ratio of test data
        random_state: Random seed for reproducibility
        num_classes: Number of classes to use
        
    Returns:
        Tuple of (train_dir, test_dir) paths
    """
    # Create target directories
    train_dir = os.path.join(target_dir, "train")
    test_dir = os.path.join(target_dir, "test")
    
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)
    
    # Set random seed
    random.seed(random_state)
    np.random.seed(random_state)
    
    # Get class folders (first num_classes)
    class_folders = sorted(
        [f for f in os.listdir(source_dir) if os.path.isdir(os.path.join(source_dir, f))]
    )[:num_classes]
    
    print(f"Processing {len(class_folders)} classes: {class_folders}")
    
    # Process each class
    for class_name in class_folders:
        print(f"Processing class: {class_name}")
        
        # Create class folders
        os.makedirs(os.path.join(train_dir, class_name), exist_ok=True)
        os.makedirs(os.path.join(test_dir, class_name), exist_ok=True)
        
        # Get all files for this class
        class_path = os.path.join(source_dir, class_name)
        files = [
            f
            for f in os.listdir(class_path)
            if os.path.isfile(os.path.join(class_path, f))
        ]
        
        # Split files
        train_files, test_files = train_test_split(
            files, test_size=test_ratio, random_state=random_state
        )
        
        # Copy files to train directory
        for file in train_files:
            src = os.path.join(class_path, file)
            dst = os.path.join(train_dir, class_name, file)
            shutil.copy2(src, dst)
        
        # Copy files to test directory
        for file in test_files:
            src = os.path.join(class_path, file)
            dst = os.path.join(test_dir, class_name, file)
            shutil.copy2(src, dst)
        
        print(
            f"  - Split {len(files)} files into {len(train_files)} train and "
            f"{len(test_files)} test"
        )
    
    return train_dir, test_dir


def split_dataset(
    dataset,
    val_ratio: float = 0.2,
    random_state: int = 42
) -> Tuple:
    """
    Split dataset into train and validation sets.
    
    Args:
        dataset: PyTorch dataset
        val_ratio: Validation ratio
        random_state: Random seed
        
    Returns:
        Tuple of (train_dataset, val_dataset)
    """
    from torch.utils.data import random_split
    import torch
    
    # Set random seed
    g = torch.Generator()
    g.manual_seed(random_state)
    
    # Calculate split sizes
    train_size = 1.0 - val_ratio
    
    # Split dataset
    train_dataset, val_dataset = random_split(
        dataset, [train_size, val_ratio], generator=g
    )
    
    return train_dataset, val_dataset


def get_class_distribution(dataset) -> dict:
    """
    Get class distribution in dataset.
    
    Args:
        dataset: PyTorch dataset
        
    Returns:
        Dictionary mapping class indices to counts
    """
    from collections import Counter
    
    labels = [dataset[i][1] for i in range(len(dataset))]
    return dict(Counter(labels))


def download_kaggle_dataset(dataset_name: str, download_path: str) -> str:
    """
    Download dataset from Kaggle.
    
    Args:
        dataset_name: Kaggle dataset name
        download_path: Path to download dataset
        
    Returns:
        Path to downloaded dataset
    """
    try:
        import kagglehub
        
        # Download dataset
        dataset_path = kagglehub.dataset_download(dataset_name)
        print(f"Dataset downloaded to: {dataset_path}")
        
        # Move files to target location
        os.makedirs(download_path, exist_ok=True)
        
        for item in os.listdir(dataset_path):
            src = os.path.join(dataset_path, item)
            dst = os.path.join(download_path, item)
            if os.path.isdir(src):
                shutil.copytree(src, dst, dirs_exist_ok=True)
            else:
                shutil.copy2(src, dst)
        
        return download_path
        
    except ImportError:
        raise ImportError(
            "kagglehub is required to download datasets. "
            "Install it with: pip install kagglehub"
        )

