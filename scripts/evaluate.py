"""Evaluation script for animal classification."""

import sys
from pathlib import Path
import argparse
import torch
import matplotlib.pyplot as plt

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

from src.utils.config import load_config
from src.utils.logger import setup_logger
from src.data.dataset import AnimalDataset, get_transforms, get_dataloaders
from src.models.model_factory import load_model
from src.evaluation.evaluator import Evaluator


def main(args):
    """Main evaluation function."""
    
    # Setup logger
    logger = setup_logger(
        name="evaluate",
        log_file="logs/evaluation.log",
        level="INFO",
    )
    
    logger.info("Starting evaluation...")
    
    # Load configuration
    config = load_config(args.config)
    logger.info(f"Loaded configuration from {args.config}")
    
    # Set device
    device = "cuda" if torch.cuda.is_available() and not args.cpu else "cpu"
    logger.info(f"Using device: {device}")
    
    # Load test dataset
    logger.info("Loading test dataset...")
    test_transform = get_transforms(config, is_training=False)
    test_dataset = AnimalDataset(
        root=args.test_dir,
        transform=test_transform,
        num_classes=config["data"]["num_classes"],
    )
    
    logger.info(f"Test samples: {len(test_dataset)}")
    
    # Create data loader
    _, _, test_loader = get_dataloaders(
        config, None, None, test_dataset
    )
    
    # Load model
    logger.info(f"Loading model from {args.model_path}")
    model = load_model(args.model_path, config, device)
    
    # Create evaluator
    class_names = config["data"]["class_names"]
    evaluator = Evaluator(model, device=device, class_names=class_names)
    
    # Evaluate
    logger.info("Evaluating model...")
    metrics = evaluator.evaluate(test_loader, loss_fn=torch.nn.CrossEntropyLoss())
    
    # Print results
    logger.info("=" * 50)
    logger.info("Evaluation Results:")
    logger.info("=" * 50)
    logger.info(f"Accuracy: {metrics['accuracy']:.4f}")
    logger.info(f"Precision: {metrics['precision']:.4f}")
    logger.info(f"Recall: {metrics['recall']:.4f}")
    logger.info(f"F1-Score: {metrics['f1_score']:.4f}")
    logger.info(f"Loss: {metrics['loss']:.4f}")
    
    # Get predictions
    predictions, labels, probabilities = evaluator.get_predictions(
        test_loader, return_probabilities=True
    )
    
    # Print classification report
    logger.info("\nClassification Report:")
    evaluator.print_classification_report(labels, predictions)
    
    # Generate visualizations
    if args.visualize:
        logger.info("\nGenerating visualizations...")
        
        # Confusion matrix
        fig_cm = evaluator.plot_confusion_matrix(
            labels,
            predictions,
            save_path="outputs/confusion_matrix.png",
            normalize=args.normalize_cm,
        )
        plt.close(fig_cm)
        logger.info("Saved confusion matrix to outputs/confusion_matrix.png")
        
        # Per-class accuracies
        fig_acc = evaluator.plot_class_accuracies(
            labels,
            predictions,
            save_path="outputs/class_accuracies.png",
        )
        plt.close(fig_acc)
        logger.info("Saved class accuracies to outputs/class_accuracies.png")
        
        # Sample predictions
        fig_pred = evaluator.visualize_predictions(
            test_loader,
            num_samples=16,
            save_path="outputs/sample_predictions.png",
        )
        plt.close(fig_pred)
        logger.info("Saved sample predictions to outputs/sample_predictions.png")
    
    logger.info("\nEvaluation completed!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate animal classification model")
    
    parser.add_argument(
        "--test-dir",
        type=str,
        default="data/processed/test",
        help="Path to test data directory",
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default="models/best_model.pth",
        help="Path to trained model",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/config.yaml",
        help="Path to configuration file",
    )
    parser.add_argument(
        "--cpu",
        action="store_true",
        help="Use CPU instead of GPU",
    )
    parser.add_argument(
        "--visualize",
        action="store_true",
        help="Generate visualization plots",
    )
    parser.add_argument(
        "--normalize-cm",
        action="store_true",
        help="Normalize confusion matrix",
    )
    
    args = parser.parse_args()
    
    main(args)
