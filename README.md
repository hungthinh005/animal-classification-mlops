# Animal Classification MLOps

A production-ready MLOps project for multiclass animal image classification using Convolutional Neural Networks (CNNs) and Transfer Learning with ResNet-50.

## 🎯 Project Overview

This project demonstrates a complete ML lifecycle implementation for classifying animal images across multiple species. It includes data preprocessing, model training with multiple architectures, evaluation, and deployment via REST API.

### Key Features

- **Multiple Model Architectures**: Custom CNNs and ResNet-50 with transfer learning
- **MLflow Integration**: Experiment tracking and model registry
- **FastAPI Deployment**: REST API for real-time predictions
- **Comprehensive Testing**: Unit tests for all components
- **Docker Support**: Containerized deployment
- **CI/CD Pipeline**: Automated testing and deployment

## 📁 Project Structure

```
animal-classification-mlops/
├── api/                      # FastAPI application
│   ├── main.py              # API endpoints
│   └── test_client.py       # API test client
├── configs/                  # Configuration files
│   └── config.yaml          # Main configuration
├── data/                     # Data directory
│   ├── raw/                 # Raw data
│   ├── processed/           # Processed data
│   └── predictions/         # Prediction outputs
├── models/                   # Saved models
├── notebooks/                # Jupyter notebooks
├── src/                      # Source code
│   ├── data/                # Data processing
│   │   ├── dataset.py
│   │   └── preprocessing.py
│   ├── models/              # Model architectures
│   │   ├── cnn_models.py
│   │   ├── resnet_model.py
│   │   └── model_factory.py
│   ├── training/            # Training modules
│   │   ├── trainer.py
│   │   └── early_stopping.py
│   ├── evaluation/          # Evaluation modules
│   │   ├── evaluator.py
│   │   └── predictor.py
│   └── utils/               # Utilities
│       ├── config.py
│       ├── logger.py
│       └── metrics.py
├── tests/                    # Test suite
├── logs/                     # Training logs
├── outputs/                  # Output artifacts
├── Dockerfile               # Docker configuration
├── docker-compose.yml       # Docker Compose configuration
├── requirements.txt         # Python dependencies
├── pyproject.toml           # Project metadata
└── README.md                # This file
```

## 🚀 Getting Started

### Prerequisites

- Python 3.9+
- CUDA-capable GPU (optional, for faster training)
- Docker (optional, for containerized deployment)

### Installation

1. **Clone the repository**

```bash
git clone https://github.com/yourusername/animal-classification-mlops.git
cd animal-classification-mlops
```

2. **Create virtual environment**

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies**

```bash
pip install -r requirements.txt
```

### Quick Start

#### 1. Data Preparation

Download and prepare the dataset:

```python
from src.data.preprocessing import download_kaggle_dataset, prepare_data

# Download dataset
download_kaggle_dataset(
    "iamsouravbanerjee/animal-image-dataset-90-different-animals",
    "data/raw"
)

# Split into train/test
prepare_data(
    source_dir="data/raw/animals/animals",
    target_dir="data/processed",
    test_ratio=0.2,
    num_classes=10
)
```

#### 2. Train Model

```python
from src.utils.config import load_config
from src.data.dataset import AnimalDataset, get_transforms, get_dataloaders
from src.models.model_factory import create_model
from src.training.trainer import Trainer

# Load configuration
config = load_config("configs/config.yaml")

# Create datasets
transform = get_transforms(config, is_training=True)
train_dataset = AnimalDataset("data/processed/train", transform=transform, num_classes=10)

# Create model
model = create_model(config)

# Train
trainer = Trainer(model, config, device="cuda")
history = trainer.train(train_loader, val_loader, epochs=50)
```

Or use the training script:

```bash
python scripts/train.py --config configs/config.yaml
```

#### 3. Evaluate Model

```python
from src.evaluation.evaluator import Evaluator

evaluator = Evaluator(model, device="cuda", class_names=class_names)
metrics = evaluator.evaluate(test_loader)
print(metrics)
```

#### 4. Run API Server

```bash
cd api
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

Test the API:

```bash
python api/test_client.py path/to/image.jpg
```

## 🏗️ Model Architectures

### 1. CNNDualConv (VGG-style)
- Multiple convolutional blocks with batch normalization
- Dropout for regularization
- Suitable for smaller datasets

### 2. CNNSingleConv
- Lightweight architecture
- Faster training
- Lower parameter count

### 3. ResNet-50 (Transfer Learning)
- Pretrained on ImageNet
- Fine-tuned for animal classification
- Best performance on the benchmark

## 📊 Experiment Tracking

This project uses MLflow for experiment tracking:

```bash
# Start MLflow UI
mlflow ui --backend-store-uri mlruns

# Access at http://localhost:5000
```

## 🐳 Docker Deployment

### Build and Run

```bash
# Build image
docker build -t animal-classification:latest .

# Run container
docker run -p 8000:8000 animal-classification:latest
```

### Using Docker Compose

```bash
docker-compose up -d
```

## 🧪 Testing

Run the test suite:

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=src --cov-report=html

# Run specific test file
pytest tests/test_models.py
```

## 📈 Performance

| Model | Accuracy | F1-Score | Parameters | Training Time |
|-------|----------|----------|------------|---------------|
| CNNDualConv | 65% | 0.63 | 2.5M | 20 min |
| CNNSingleConv | 45% | 0.42 | 1.2M | 10 min |
| ResNet-50 | 98% | 0.97 | 23.5M | 15 min |

*Tested on 10-class animal classification with 480 training images*

## 📝 Configuration

Edit `configs/config.yaml` to customize:

- Model architecture
- Training hyperparameters
- Data augmentation settings
- API configuration
- MLflow settings

## 🔄 CI/CD Pipeline

This project includes GitHub Actions workflows for:

- Automated testing on push/PR
- Code quality checks (black, flake8)
- Docker image building
- Model deployment

## 📚 API Documentation

### Endpoints

- `GET /` - API information
- `GET /health` - Health check
- `POST /predict` - Predict single image
- `POST /predict/batch` - Predict multiple images
- `GET /classes` - Get supported classes

### Example Request

```python
import requests

url = "http://localhost:8000/predict"
files = {"file": open("cat.jpg", "rb")}
response = requests.post(url, files=files)
print(response.json())
```

### Example Response

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

## 🤝 Contributing

Contributions are welcome! Please:

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Dataset: [Animal Image Dataset](https://www.kaggle.com/datasets/iamsouravbanerjee/animal-image-dataset-90-different-animals)
- PyTorch team for the deep learning framework
- FastAPI for the modern web framework
- MLflow for experiment tracking

## 📧 Contact

For questions or feedback, please open an issue on GitHub.

---

**Built with ❤️ for MLOps excellence**

