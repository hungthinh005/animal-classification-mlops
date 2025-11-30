"""Prediction script for animal classification."""

import sys
from pathlib import Path
import argparse
import torch
from PIL import Image
import json

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

from src.utils.config import load_config
from src.utils.logger import setup_logger
from src.data.dataset import get_transforms
from src.models.model_factory import load_model
from src.evaluation.predictor import Predictor


def main(args):
    """Main prediction function."""
    
    # Setup logger
    logger = setup_logger(
        name="predict",
        log_file="logs/prediction.log" if not args.quiet else None,
        level="INFO" if not args.quiet else "WARNING",
    )
    
    logger.info("Starting prediction...")
    
    # Load configuration
    config = load_config(args.config)
    logger.info(f"Loaded configuration from {args.config}")
    
    # Set device
    device = "cuda" if torch.cuda.is_available() and not args.cpu else "cpu"
    logger.info(f"Using device: {device}")
    
    # Load model
    logger.info(f"Loading model from {args.model_path}")
    model = load_model(args.model_path, config, device)
    
    # Get transforms and class names
    transform = get_transforms(config, is_training=False)
    class_names = config["data"]["class_names"]
    
    # Create predictor
    predictor = Predictor(
        model=model,
        class_names=class_names,
        transform=transform,
        device=device,
    )
    
    # Predict based on mode
    if args.batch:
        # Batch prediction
        logger.info(f"Running batch prediction on directory: {args.input}")
        image_files = list(Path(args.input).glob("*.jpg")) + \
                     list(Path(args.input).glob("*.jpeg")) + \
                     list(Path(args.input).glob("*.png"))
        
        logger.info(f"Found {len(image_files)} images")
        
        results = predictor.predict_batch(
            [str(f) for f in image_files],
            batch_size=args.batch_size,
        )
        
        # Save or display results
        output_data = []
        for img_path, result in zip(image_files, results):
            output_data.append({
                "image": str(img_path),
                "predicted_class": result["predicted_class"],
                "confidence": float(result["confidence"]),
            })
            
            if not args.quiet:
                print(f"{img_path.name}: {result['predicted_class']} ({result['confidence']:.4f})")
        
        # Save to JSON if output specified
        if args.output:
            with open(args.output, 'w') as f:
                json.dump(output_data, f, indent=2)
            logger.info(f"Results saved to {args.output}")
    
    else:
        # Single image prediction
        logger.info(f"Running prediction on image: {args.input}")
        
        result = predictor.predict_image(args.input, top_k=args.top_k)
        
        # Display results
        print(f"\nPrediction Results:")
        print(f"=" * 50)
        print(f"Predicted Class: {result['predicted_class']}")
        print(f"Confidence: {result['confidence']:.4f}")
        print(f"\nTop {args.top_k} Predictions:")
        for i, (class_name, prob) in enumerate(result['top_predictions'], 1):
            print(f"  {i}. {class_name}: {prob:.4f}")
        
        # Save to JSON if output specified
        if args.output:
            with open(args.output, 'w') as f:
                json.dump(result, f, indent=2)
            logger.info(f"Results saved to {args.output}")
    
    logger.info("Prediction completed!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Predict animal class from image(s)")
    
    parser.add_argument(
        "input",
        type=str,
        help="Path to image file or directory (for batch mode)",
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
        "--output",
        type=str,
        help="Path to save results (JSON format)",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="Number of top predictions to return",
    )
    parser.add_argument(
        "--batch",
        action="store_true",
        help="Batch prediction mode (process directory)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Batch size for batch prediction",
    )
    parser.add_argument(
        "--cpu",
        action="store_true",
        help="Use CPU instead of GPU",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress output messages",
    )
    
    args = parser.parse_args()
    
    main(args)
