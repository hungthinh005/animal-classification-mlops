"""Training script for animal classification."""

import sys
from pathlib import Path
import argparse
import torch

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

from src.utils.config import load_config
from src.utils.logger import setup_logger
from src.data.dataset import AnimalDataset, get_transforms, get_dataloaders
from src.data.preprocessing import split_dataset
from src.models.model_factory import create_model
from src.training.trainer import Trainer


def main(args):
    """Main training function."""
    
    # Setup logger
    logger = setup_logger(
        name="train",
        log_file="logs/training.log",
        level="INFO",
    )
    
    logger.info("Starting training...")
    
    # Load configuration
    config = load_config(args.config)
    logger.info(f"Loaded configuration from {args.config}")
    
    # Override config with command line arguments
    if args.architecture:
        config["model"]["architecture"] = args.architecture
    if args.epochs:
        config["training"]["epochs"] = args.epochs
    if args.batch_size:
        config["training"]["batch_size"] = args.batch_size
    if args.learning_rate:
        config["training"]["learning_rate"] = args.learning_rate
    
    # Set device
    device = "cuda" if torch.cuda.is_available() and not args.cpu else "cpu"
    logger.info(f"Using device: {device}")
    
    # Create datasets
    logger.info("Loading datasets...")
    
    train_transform = get_transforms(config, is_training=True)
    val_transform = get_transforms(config, is_training=False)
    
    # Load full training dataset
    full_train_dataset = AnimalDataset(
        root=args.train_dir,
        transform=train_transform,
        num_classes=config["data"]["num_classes"],
    )
    
    # Split into train and validation
    train_dataset, val_dataset = split_dataset(
        full_train_dataset,
        val_ratio=config["data"]["val_size"],
        random_state=config["project"]["random_seed"],
    )
    
    # Update validation dataset transform
    val_dataset.dataset.transform = val_transform
    
    logger.info(f"Train samples: {len(train_dataset)}")
    logger.info(f"Validation samples: {len(val_dataset)}")
    
    # Create data loaders
    train_loader, val_loader, _ = get_dataloaders(
        config, train_dataset, val_dataset
    )
    
    # Create model
    logger.info(f"Creating model: {config['model']['architecture']}")
    model = create_model(config)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Total parameters: {total_params:,}")
    logger.info(f"Trainable parameters: {trainable_params:,}")
    
    # Create trainer
    trainer = Trainer(model, config, device=device)
    
    # Train
    logger.info("Starting training...")
    history = trainer.train(
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=config["training"]["epochs"],
        checkpoint_path=args.checkpoint_path,
        use_mlflow=not args.no_mlflow,
    )
    
    logger.info("Training completed!")
    logger.info(f"Best validation accuracy: {max(history['val_accuracy']):.4f}")
    logger.info(f"Best validation loss: {min(history['val_loss']):.4f}")
    
    # Save final model
    if args.save_final:
        final_path = args.checkpoint_path.replace("best_model", "final_model")
        torch.save(model.state_dict(), final_path)
        logger.info(f"Final model saved to {final_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train animal classification model")
    
    # Data arguments
    parser.add_argument(
        "--train-dir",
        type=str,
        default="data/processed/train",
        help="Path to training data directory",
    )
    
    # Model arguments
    parser.add_argument(
        "--architecture",
        type=str,
        choices=["cnn_dual", "cnn_single", "resnet50"],
        help="Model architecture",
    )
    parser.add_argument(
        "--checkpoint-path",
        type=str,
        default="models/best_model.pth",
        help="Path to save best model",
    )
    
    # Training arguments
    parser.add_argument(
        "--config",
        type=str,
        default="configs/config.yaml",
        help="Path to configuration file",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        help="Number of epochs",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        help="Batch size",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        help="Learning rate",
    )
    parser.add_argument(
        "--cpu",
        action="store_true",
        help="Use CPU instead of GPU",
    )
    parser.add_argument(
        "--no-mlflow",
        action="store_true",
        help="Disable MLflow tracking",
    )
    parser.add_argument(
        "--save-final",
        action="store_true",
        help="Save final model in addition to best model",
    )
    
    args = parser.parse_args()
    
    main(args)

