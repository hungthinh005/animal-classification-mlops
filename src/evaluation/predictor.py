"""Prediction utilities for inference."""

import torch
import torch.nn as nn
import numpy as np
from PIL import Image
from typing import Dict, List, Tuple, Optional, Union
from pathlib import Path


class Predictor:
    """Model predictor for inference."""
    
    def __init__(
        self,
        model: nn.Module,
        class_names: List[str],
        transform,
        device: str = "cuda",
    ):
        """
        Initialize predictor.
        
        Args:
            model: Trained model
            class_names: List of class names
            transform: Image transformation pipeline
            device: Device to run inference on
        """
        self.model = model.to(device)
        self.class_names = class_names
        self.transform = transform
        self.device = device
        self.model.eval()
    
    def predict_image(
        self,
        image: Union[str, Path, Image.Image],
        top_k: int = 5,
    ) -> Dict[str, Union[str, float, List[Tuple[str, float]]]]:
        """
        Predict class for a single image.
        
        Args:
            image: Image file path or PIL Image
            top_k: Number of top predictions to return
            
        Returns:
            Dictionary containing prediction results
        """
        # Load image if path provided
        if isinstance(image, (str, Path)):
            image = Image.open(image).convert("RGB")
        
        # Transform image
        image_tensor = self.transform(image).unsqueeze(0).to(self.device)
        
        # Make prediction
        with torch.no_grad():
            outputs = self.model(image_tensor)
            probabilities = torch.softmax(outputs, dim=1)
        
        # Get top-k predictions
        top_probs, top_indices = torch.topk(probabilities[0], top_k)
        
        # Format results
        top_predictions = [
            (self.class_names[idx.item()], prob.item())
            for idx, prob in zip(top_indices, top_probs)
        ]
        
        # Get top prediction
        predicted_class = self.class_names[top_indices[0].item()]
        confidence = top_probs[0].item()
        
        return {
            "predicted_class": predicted_class,
            "confidence": confidence,
            "top_predictions": top_predictions,
        }
    
    def predict_batch(
        self,
        images: List[Union[str, Path, Image.Image]],
        batch_size: int = 32,
    ) -> List[Dict[str, Union[str, float]]]:
        """
        Predict classes for multiple images.
        
        Args:
            images: List of image paths or PIL Images
            batch_size: Batch size for processing
            
        Returns:
            List of prediction dictionaries
        """
        results = []
        
        # Process in batches
        for i in range(0, len(images), batch_size):
            batch_images = images[i : i + batch_size]
            batch_tensors = []
            
            # Load and transform images
            for image in batch_images:
                if isinstance(image, (str, Path)):
                    image = Image.open(image).convert("RGB")
                
                image_tensor = self.transform(image)
                batch_tensors.append(image_tensor)
            
            # Stack into batch
            batch = torch.stack(batch_tensors).to(self.device)
            
            # Make predictions
            with torch.no_grad():
                outputs = self.model(batch)
                probabilities = torch.softmax(outputs, dim=1)
                _, predicted = torch.max(outputs, 1)
            
            # Format results
            for idx in range(len(batch_images)):
                pred_idx = predicted[idx].item()
                confidence = probabilities[idx][pred_idx].item()
                
                results.append({
                    "predicted_class": self.class_names[pred_idx],
                    "confidence": confidence,
                })
        
        return results
    
    def predict_from_array(
        self,
        image_array: np.ndarray,
        top_k: int = 5,
    ) -> Dict[str, Union[str, float, List[Tuple[str, float]]]]:
        """
        Predict class from numpy array.
        
        Args:
            image_array: Image as numpy array (H, W, C)
            top_k: Number of top predictions to return
            
        Returns:
            Dictionary containing prediction results
        """
        # Convert numpy array to PIL Image
        if image_array.dtype != np.uint8:
            image_array = (image_array * 255).astype(np.uint8)
        
        image = Image.fromarray(image_array)
        
        return self.predict_image(image, top_k=top_k)
    
    def predict_with_gradcam(
        self,
        image: Union[str, Path, Image.Image],
        target_layer: Optional[nn.Module] = None,
    ) -> Tuple[Dict, np.ndarray]:
        """
        Predict with GradCAM visualization.
        
        Note: This is a placeholder for GradCAM implementation.
        Full implementation would require pytorch-grad-cam library.
        
        Args:
            image: Image file path or PIL Image
            target_layer: Target layer for GradCAM
            
        Returns:
            Tuple of (prediction_dict, gradcam_heatmap)
        """
        # Get prediction
        prediction = self.predict_image(image)
        
        # Placeholder for GradCAM heatmap
        # In production, implement using pytorch-grad-cam
        heatmap = np.zeros((224, 224))
        
        return prediction, heatmap
    
    @staticmethod
    def load_predictor(
        model_path: str,
        config: Dict,
        device: str = "cuda",
    ) -> "Predictor":
        """
        Load predictor from saved model.
        
        Args:
            model_path: Path to saved model
            config: Configuration dictionary
            device: Device to load model on
            
        Returns:
            Predictor instance
        """
        from ..models.model_factory import load_model
        from ..data.dataset import get_transforms
        
        # Load model
        model = load_model(model_path, config, device)
        
        # Get transforms (test mode, no augmentation)
        transform = get_transforms(config, is_training=False)
        
        # Get class names
        class_names = config["data"]["class_names"]
        
        return Predictor(model, class_names, transform, device)

