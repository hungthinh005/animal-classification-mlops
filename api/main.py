"""FastAPI application for model serving."""

import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import List, Optional
import torch
from PIL import Image
import io

from src.utils.config import load_config
from src.evaluation.predictor import Predictor


# Initialize FastAPI app
app = FastAPI(
    title="Animal Classification API",
    description="API for multiclass animal image classification",
    version="0.1.0",
)

# Global variables for model and predictor
predictor: Optional[Predictor] = None
config: Optional[dict] = None


class PredictionResponse(BaseModel):
    """Response model for predictions."""
    
    predicted_class: str
    confidence: float
    top_predictions: List[dict]


class HealthResponse(BaseModel):
    """Response model for health check."""
    
    status: str
    model_loaded: bool
    device: str


@app.on_event("startup")
async def startup_event():
    """Load model on startup."""
    global predictor, config
    
    try:
        # Load configuration
        config = load_config("configs/config.yaml")
        
        # Determine device
        device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Load predictor
        model_path = config["api"]["model_path"]
        predictor = Predictor.load_predictor(model_path, config, device)
        
        print(f"Model loaded successfully on {device}")
    except Exception as e:
        print(f"Error loading model: {e}")
        predictor = None


@app.get("/", response_model=dict)
async def root():
    """Root endpoint."""
    return {
        "message": "Animal Classification API",
        "version": "0.1.0",
        "endpoints": {
            "health": "/health",
            "predict": "/predict",
            "predict_batch": "/predict/batch",
        },
    }


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    return HealthResponse(
        status="healthy" if predictor is not None else "unhealthy",
        model_loaded=predictor is not None,
        device=device,
    )


@app.post("/predict", response_model=PredictionResponse)
async def predict(file: UploadFile = File(...)):
    """
    Predict animal class for uploaded image.
    
    Args:
        file: Image file
        
    Returns:
        Prediction results
    """
    if predictor is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    # Validate file type
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image")
    
    # Check file size
    max_size = config["api"].get("max_image_size", 10 * 1024 * 1024)  # 10MB
    
    try:
        # Read image
        contents = await file.read()
        
        if len(contents) > max_size:
            raise HTTPException(
                status_code=400,
                detail=f"File size exceeds maximum of {max_size / (1024*1024):.1f}MB",
            )
        
        # Convert to PIL Image
        image = Image.open(io.BytesIO(contents)).convert("RGB")
        
        # Make prediction
        result = predictor.predict_image(image, top_k=5)
        
        # Format top predictions
        top_preds = [
            {"class": cls, "confidence": float(conf)}
            for cls, conf in result["top_predictions"]
        ]
        
        return PredictionResponse(
            predicted_class=result["predicted_class"],
            confidence=float(result["confidence"]),
            top_predictions=top_preds,
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")


@app.post("/predict/batch")
async def predict_batch(files: List[UploadFile] = File(...)):
    """
    Predict animal classes for multiple images.
    
    Args:
        files: List of image files
        
    Returns:
        List of prediction results
    """
    if predictor is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    if len(files) > 32:
        raise HTTPException(status_code=400, detail="Maximum 32 images per request")
    
    results = []
    
    for file in files:
        try:
            # Validate file type
            if not file.content_type.startswith("image/"):
                results.append({
                    "filename": file.filename,
                    "error": "Not an image file",
                })
                continue
            
            # Read and predict
            contents = await file.read()
            image = Image.open(io.BytesIO(contents)).convert("RGB")
            result = predictor.predict_image(image, top_k=3)
            
            results.append({
                "filename": file.filename,
                "predicted_class": result["predicted_class"],
                "confidence": float(result["confidence"]),
            })
        
        except Exception as e:
            results.append({
                "filename": file.filename,
                "error": str(e),
            })
    
    return JSONResponse(content={"results": results})


@app.get("/classes")
async def get_classes():
    """Get list of supported classes."""
    if config is None:
        raise HTTPException(status_code=503, detail="Configuration not loaded")
    
    return {
        "classes": config["data"]["class_names"],
        "num_classes": config["data"]["num_classes"],
    }


if __name__ == "__main__":
    import uvicorn
    
    uvicorn.run(
        "main:app",
        host=config["api"]["host"] if config else "0.0.0.0",
        port=config["api"]["port"] if config else 8000,
        reload=True,
    )

