"""Pytest configuration and fixtures."""

import pytest
import torch
import numpy as np
from pathlib import Path
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.config import load_config
from src.models.model_factory import create_model


@pytest.fixture
def config():
    """Load test configuration."""
    return load_config("configs/config.yaml")


@pytest.fixture
def device():
    """Get device for testing."""
    return "cuda" if torch.cuda.is_available() else "cpu"


@pytest.fixture
def sample_image():
    """Create a sample image tensor."""
    return torch.randn(1, 3, 224, 224)


@pytest.fixture
def sample_batch():
    """Create a sample batch of images."""
    return torch.randn(8, 3, 224, 224)


@pytest.fixture
def sample_labels():
    """Create sample labels."""
    return torch.randint(0, 10, (8,))


@pytest.fixture
def cnn_model(config):
    """Create a CNN model for testing."""
    config["model"]["architecture"] = "cnn_single"
    return create_model(config)


@pytest.fixture
def resnet_model(config):
    """Create a ResNet model for testing."""
    config["model"]["architecture"] = "resnet50"
    config["model"]["pretrained"] = False
    return create_model(config)

