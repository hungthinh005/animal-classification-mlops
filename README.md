# Animal Classification MLOps

> Production-ready multiclass image classification using CNNs and Transfer Learning with complete MLOps infrastructure.

[![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)
[![Docker](https://img.shields.io/badge/Docker-Ready-green.svg)](https://www.docker.com/)
[![MLflow](https://img.shields.io/badge/MLflow-Tracking-orange.svg)](https://mlflow.org/)

---

## Overview

End-to-end MLOps system for classifying animal images across 10+ species using deep learning, featuring multiple CNN architectures, automated training pipelines, and REST API deployment.

**Key Features**:
- Multiple model architectures (ResNet-50, Custom CNNs)
- Transfer learning with 98% accuracy
- REST API for real-time predictions
- MLflow experiment tracking
- Docker deployment
- CI/CD pipelines

---

## Performance

| Model | Accuracy | F1-Score | Parameters | Training Time (GPU) |
|-------|----------|----------|------------|---------------------|
| **ResNet-50** | **98%** | **0.97** | 23.5M | ~15 min |
| CNNDualConv | 65% | 0.63 | 2.5M | ~20 min |
| CNNSingleConv | 45% | 0.42 | 1.2M | ~10 min |

*Tested on 10-class animal classification (480 training images, 50 epochs)*

---

## Architecture

### System Components

```
┌────────────────────────────────────────────────────────────┐
│                    Docker Environment                       │
│                                                             │
│  ┌──────────────┐        ┌──────────────┐                 │
│  │   FastAPI    │        │    MLflow    │                 │
│  │ (Port 8000)  │        │ (Port 5000)  │                 │
│  │              │        │              │                 │
│  │ - Prediction │        │ - Tracking   │                 │
│  │ - Batch API  │        │ - Registry   │                 │
│  └──────────────┘        └──────────────┘                 │
│                                                             │
└────────────────────────────────────────────────────────────┘
```

### ML Pipeline

```
┌─────────────┐    ┌──────────────┐    ┌─────────────┐    ┌──────────────┐
│   Dataset   │ -> │ Preprocessing │ -> │   Training  │ -> │  Deployment  │
│  (Kaggle)   │    │ Augmentation  │    │   Models    │    │   (Docker)   │
└─────────────┘    └──────────────┘    └─────────────┘    └──────────────┘
       |                   |                    |                   |
       v                   v                    v                   v
  90 classes         Resize 224x224      ResNet-50/CNNs      FastAPI + MLflow
  5400+ images       Normalization      Early Stopping         Monitoring
                     Data Augment       MLflow Tracking
```

---

## Quick Start

### 🐍 Local Python Setup

#### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

#### 2. Start Services

```bash
# Terminal 1: MLflow UI
mlflow ui --backend-store-uri mlruns --host 0.0.0.0 --port 5000

# Terminal 2: FastAPI (after training)
cd api
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

**Or use PowerShell scripts** (Windows):
```powershell
.\start_services.ps1    # Start both MLflow + API
.\start_mlflow.ps1      # MLflow only
.\start_api.ps1         # API only
```

#### 3. Prepare Dataset

```python
from src.data.preprocessing import download_kaggle_dataset, split_dataset

# Download from Kaggle
download_kaggle_dataset(
    "iamsouravbanerjee/animal-image-dataset-90-different-animals",
    "data/raw"
)

# Split into train/test
split_dataset(
    source_dir="data/raw/animals/animals",
    target_dir="data/processed",
    test_ratio=0.2,
    num_classes=10  # Use first 10 classes
)
```

#### 4. Train Model

```bash
# Recommended: ResNet-50 with transfer learning
python scripts/train.py --config configs/config.yaml --architecture resnet50 --epochs 50

# Fast training: Custom CNN
python scripts/train.py --architecture cnn_dual --epochs 30 --batch-size 32
```

**Track experiments** at http://localhost:5000

#### 5. Make Predictions

```bash
# Single image
python scripts/predict.py --model-path models/best_model.pth --image-path path/to/image.jpg

# Using API
curl -X POST "http://localhost:8000/predict" -F "file=@animal.jpg"
```

---

### 🐳 Docker Setup

#### Start All Services

```bash
docker-compose up -d
```

Services available:
- **API**: http://localhost:8000
- **API Docs**: http://localhost:8000/docs
- **MLflow**: http://localhost:5000

#### Stop Services

```bash
docker-compose down
```

#### View Logs

```bash
docker-compose logs -f
```

---

## Project Structure

```
animal-classification-mlops/
├── api/                        # FastAPI application
│   ├── main.py                # REST API endpoints
│   └── test_client.py         # API testing
├── configs/
│   └── config.yaml            # Configuration
├── src/
│   ├── data/                  # Data processing
│   │   ├── dataset.py        # PyTorch Dataset
│   │   └── preprocessing.py  # Data pipeline
│   ├── models/                # Model architectures
│   │   ├── cnn_models.py     # Custom CNNs
│   │   ├── resnet_model.py   # ResNet-50
│   │   └── model_factory.py  # Model creation
│   ├── training/              # Training pipeline
│   │   ├── trainer.py        # Training loop
│   │   └── early_stopping.py # Early stopping
│   ├── evaluation/            # Evaluation & prediction
│   │   ├── evaluator.py
│   │   └── predictor.py
│   └── utils/                 # Utilities
│       ├── config.py
│       ├── logger.py
│       └── metrics.py
├── scripts/                   # CLI scripts
│   ├── train.py
│   ├── evaluate.py
│   └── predict.py
├── tests/                     # Unit tests
├── notebooks/                 # Jupyter notebooks
├── data/                      # Data storage
├── models/                    # Saved models
├── Dockerfile                 # Docker image
├── docker-compose.yml         # Multi-service setup
└── requirements.txt           # Dependencies
```

---

## Configuration

Edit `configs/config.yaml` to customize:

```yaml
# Model settings
model:
  architecture: "resnet50"      # resnet50, cnn_dual, cnn_single
  pretrained: true              # Use ImageNet weights
  dropout_rate: 0.5

# Training settings
training:
  batch_size: 32
  epochs: 50
  learning_rate: 0.001
  
  # Early stopping
  early_stopping:
    enabled: true
    patience: 10
    min_delta: 0.001

# Data settings
data:
  num_classes: 10               # Number of animal classes
  image_size: [224, 224]
  test_size: 0.2
```

---

## API Reference

### Endpoints

#### `GET /health`
Check API health and model status.

```bash
curl http://localhost:8000/health
```

Response:
```json
{
  "status": "healthy",
  "model_loaded": true,
  "device": "cuda"
}
```

#### `POST /predict`
Predict single image.

```bash
curl -X POST "http://localhost:8000/predict" \
  -F "file=@cat.jpg"
```

Response:
```json
{
  "predicted_class": "cat",
  "confidence": 0.95,
  "top_predictions": [
    {"class": "cat", "confidence": 0.95},
    {"class": "dog", "confidence": 0.03},
    {"class": "bear", "confidence": 0.01}
  ]
}
```

#### `POST /predict/batch`
Predict multiple images.

```bash
curl -X POST "http://localhost:8000/predict/batch" \
  -F "files=@cat.jpg" \
  -F "files=@dog.jpg"
```

#### `GET /classes`
Get supported animal classes.

**Interactive Docs**: http://localhost:8000/docs

---

## Supported Animal Classes

Default configuration (10 classes):
1. 🦌 Antelope
2. 🦡 Badger
3. 🦇 Bat
4. 🐻 Bear
5. 🐝 Bee
6. 🪲 Beetle
7. 🦬 Bison
8. 🐗 Boar
9. 🦋 Butterfly
10. 🐱 Cat

*Extend to 90 classes by updating `configs/config.yaml`*

---

## MLflow Experiment Tracking

### View Experiments

```bash
mlflow ui --backend-store-uri mlruns --port 5000
```

Open http://localhost:5000 to:
- Compare model runs
- View metrics & parameters
- Download artifacts
- Track model versions

### Log Custom Experiments

```python
import mlflow

mlflow.set_experiment("animal-classification")

with mlflow.start_run():
    mlflow.log_param("model", "resnet50")
    mlflow.log_param("batch_size", 32)
    mlflow.log_metric("accuracy", 0.98)
    mlflow.log_artifact("model.pth")
```

---

## Development

### Run Tests

```bash
# All tests
pytest

# With coverage
pytest --cov=src --cov-report=html

# Specific test
pytest tests/test_models.py -v
```

### Code Quality

```bash
# Format code
black src/ tests/

# Lint
flake8 src/ tests/

# Type checking
mypy src/
```

---

## Training Options

### Command Line Arguments

```bash
python scripts/train.py \
  --config configs/config.yaml \
  --architecture resnet50 \
  --epochs 50 \
  --batch-size 32 \
  --learning-rate 0.001 \
  --device cuda
```

### Architecture Comparison

| Architecture | Use Case | Speed | Accuracy |
|-------------|----------|-------|----------|
| **ResNet-50** | Production, best accuracy | Medium | ⭐⭐⭐⭐⭐ |
| **CNNDualConv** | Balanced | Fast | ⭐⭐⭐ |
| **CNNSingleConv** | Quick experiments | Very Fast | ⭐⭐ |

---

## Monitoring & Logging

### Application Logs

```bash
tail -f logs/training.log
```

### MLflow Metrics

Automatically tracked:
- Training/validation loss
- Accuracy, F1-score, Precision, Recall
- Learning rate
- Model parameters
- Training time

---

## Deployment

### Docker Production Build

```bash
# Build image
docker build -t animal-classification:latest .

# Run container
docker run -p 8000:8000 animal-classification:latest

# Push to registry
docker tag animal-classification:latest registry/animal-classification:v1.0
docker push registry/animal-classification:v1.0
```

### Environment Variables

```bash
PYTHONPATH=/app
CUDA_VISIBLE_DEVICES=0
MODEL_PATH=models/best_model.pth
```

---

## CI/CD Pipeline

GitHub Actions workflow (`.github/workflows/ci-cd.yml`):

- ✅ Automated testing on push/PR
- ✅ Code quality checks (black, flake8)
- ✅ Docker image building
- ✅ Model deployment

---

## Troubleshooting

### Common Issues

**Issue**: CUDA out of memory
```bash
# Solution: Reduce batch size
python scripts/train.py --batch-size 16
```

**Issue**: Port already in use
```bash
# Find process
netstat -ano | findstr :5000

# Kill process
taskkill /PID <pid> /F
```

**Issue**: Model not found
```bash
# Verify model exists
ls models/best_model.pth

# Check config path
cat configs/config.yaml | grep model_path
```

---

## Performance Tips

1. **Use GPU**: 15-20x faster than CPU
   ```bash
   python scripts/train.py --device cuda
   ```

2. **Transfer Learning**: Start with pretrained ResNet-50
   ```yaml
   model:
     architecture: resnet50
     pretrained: true
   ```

3. **Data Augmentation**: Improves generalization
   ```yaml
   augmentation:
     horizontal_flip: true
     rotation_degrees: 10
     color_jitter: true
   ```

4. **Early Stopping**: Prevents overfitting
   ```yaml
   early_stopping:
     enabled: true
     patience: 10
   ```

---

## Dataset

**Source**: [Animal Image Dataset (90 Different Animals)](https://www.kaggle.com/datasets/iamsouravbanerjee/animal-image-dataset-90-different-animals)

**Statistics**:
- 90 animal classes
- 60+ images per class
- 5400+ total images
- RGB images, various sizes

**License**: Check Kaggle dataset page

---

## Technologies

- **Framework**: PyTorch 2.0+
- **API**: FastAPI + Uvicorn
- **Tracking**: MLflow
- **Containerization**: Docker + Docker Compose
- **CI/CD**: GitHub Actions
- **Testing**: pytest
- **Code Quality**: black, flake8, mypy

---

## Contributing

1. Fork the repository
2. Create feature branch (`git checkout -b feature/new-feature`)
3. Commit changes (`git commit -m 'Add new feature'`)
4. Push to branch (`git push origin feature/new-feature`)
5. Open Pull Request

---

## License

This project is licensed under the MIT License - see [LICENSE](LICENSE) file.

---

## Acknowledgments

- Dataset: [Animal Image Dataset](https://www.kaggle.com/datasets/iamsouravbanerjee/animal-image-dataset-90-different-animals)
- PyTorch team for the deep learning framework
- FastAPI for the web framework
- MLflow for experiment tracking

---

## Contact

For questions or feedback, please open an issue on GitHub.

---

**Built with ❤️ for MLOps excellence**
