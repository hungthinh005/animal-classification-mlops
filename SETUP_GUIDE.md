# Animal Classification MLOps - Setup Guide

## 🎯 Quick Setup (5 Minutes)

### 1. Environment Setup

```bash
# Create virtual environment
python -m venv venv

# Activate (Windows)
venv\Scripts\activate

# Activate (Linux/Mac)
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Download Dataset

```bash
# Using the provided script
python -c "
from src.data.preprocessing import download_kaggle_dataset
download_kaggle_dataset(
    'iamsouravbanerjee/animal-image-dataset-90-different-animals',
    'data/raw'
)
"
```

### 3. Prepare Data

```bash
python -c "
from src.data.preprocessing import prepare_data
prepare_data(
    source_dir='data/raw/animals/animals',
    target_dir='data/processed',
    test_ratio=0.2,
    num_classes=10
)
"
```

### 4. Train Model

```bash
# Quick training with default settings
python scripts/train.py --epochs 10

# Full training with ResNet-50
python scripts/train.py \
    --architecture resnet50 \
    --epochs 50 \
    --batch-size 32 \
    --learning-rate 0.001
```

### 5. Evaluate Model

```bash
python scripts/evaluate.py \
    --model-path models/best_model.pth \
    --save-plots \
    --save-predictions
```

### 6. Start API Server

```bash
cd api
uvicorn main:app --reload

# Test the API
python test_client.py path/to/test/image.jpg
```

## 🐳 Docker Setup (Recommended for Production)

### Build and Run

```bash
# Build the image
docker build -t animal-classification:latest .

# Run the container
docker run -p 8000:8000 \
    -v $(pwd)/models:/app/models \
    animal-classification:latest
```

### Using Docker Compose (with MLflow)

```bash
# Start all services
docker-compose up -d

# Access API: http://localhost:8000
# Access MLflow: http://localhost:5000

# View logs
docker-compose logs -f api

# Stop services
docker-compose down
```

## 📊 Using MLflow

### Start MLflow Server

```bash
mlflow ui --backend-store-uri mlruns
```

Access at: http://localhost:5000

### Track Experiments

MLflow automatically tracks:
- Model architecture
- Hyperparameters
- Training/validation metrics
- Model artifacts

## 🧪 Running Tests

```bash
# Run all tests
pytest

# Run with coverage report
pytest --cov=src --cov-report=html

# Run specific test file
pytest tests/test_models.py -v

# View coverage report
open htmlcov/index.html
```

## 📝 Configuration

Edit `configs/config.yaml` to customize:

### Model Settings
```yaml
model:
  architecture: "resnet50"  # cnn_dual, cnn_single, resnet50
  pretrained: true
  freeze_backbone: false
  dropout_rate: 0.5
```

### Training Settings
```yaml
training:
  batch_size: 32
  epochs: 50
  learning_rate: 0.001
  weight_decay: 0.0001
```

### Data Augmentation
```yaml
augmentation:
  train:
    horizontal_flip: true
    rotation_degrees: 10
    color_jitter:
      brightness: 0.2
      contrast: 0.2
      saturation: 0.2
```

## 🚀 Usage Examples

### Training with Custom Config

```python
from src.utils.config import load_config, update_config
from src.training.trainer import Trainer

# Load and modify config
config = load_config("configs/config.yaml")
config = update_config(config, {
    "training": {"batch_size": 64, "learning_rate": 0.0001}
})

# Train model
trainer = Trainer(model, config, device="cuda")
history = trainer.train(train_loader, val_loader, epochs=50)
```

### Making Predictions

```python
from src.evaluation.predictor import Predictor

# Load predictor
predictor = Predictor.load_predictor(
    "models/best_model.pth",
    config,
    device="cuda"
)

# Predict single image
result = predictor.predict_image("path/to/image.jpg", top_k=5)
print(f"Predicted: {result['predicted_class']} "
      f"({result['confidence']:.2%})")
```

### Batch Predictions

```python
# Predict multiple images
image_paths = ["img1.jpg", "img2.jpg", "img3.jpg"]
results = predictor.predict_batch(image_paths, batch_size=32)

for img, result in zip(image_paths, results):
    print(f"{img}: {result['predicted_class']} "
          f"({result['confidence']:.2%})")
```

### API Usage

```python
import requests

# Health check
response = requests.get("http://localhost:8000/health")
print(response.json())

# Predict image
with open("cat.jpg", "rb") as f:
    files = {"file": f}
    response = requests.post(
        "http://localhost:8000/predict",
        files=files
    )
print(response.json())
```

## 🔧 Troubleshooting

### CUDA Out of Memory
- Reduce batch size in config
- Use gradient accumulation
- Try mixed precision training

### Slow Training
- Enable GPU (`--cpu` flag if GPU issues)
- Increase num_workers in config
- Use smaller model architecture

### Poor Performance
- Train for more epochs
- Adjust learning rate
- Increase data augmentation
- Try different model architecture
- Check for data imbalance

### API Not Starting
- Ensure model file exists at specified path
- Check port 8000 is not in use
- Verify all dependencies installed

## 📚 Additional Resources

### Project Structure
- See `README.md` for complete documentation
- Check notebooks for exploratory analysis
- Review tests for usage examples

### Customization
- Add new model architectures in `src/models/`
- Implement custom losses in `src/training/`
- Add data augmentation in `src/data/`

### Deployment
- See `Dockerfile` for containerization
- Check `.github/workflows/` for CI/CD
- Review `docker-compose.yml` for orchestration

## 🤝 Getting Help

1. Check this guide first
2. Review error logs in `logs/`
3. Check MLflow for training issues
4. Review test output for functionality
5. Open an issue on GitHub

---

**Happy Training! 🚀**

