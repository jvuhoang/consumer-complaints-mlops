#!/bin/bash

# ============================================================================
# Complete Project Setup Script
# ============================================================================

set -e  # Exit on error

echo "============================================================================"
echo "🚀 CONSUMER COMPLAINTS MLOPS - FULL PROJECT SETUP"
echo "============================================================================"
echo ""

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# ============================================================================
# CHECK PREREQUISITES
# ============================================================================
echo "📋 Step 1: Checking prerequisites..."

# Check if git is installed
if ! command -v git &> /dev/null; then
    echo -e "${RED}❌ Git is not installed. Please install git first.${NC}"
    exit 1
fi

# Check if python is installed
if ! command -v python3 &> /dev/null; then
    echo -e "${RED}❌ Python 3 is not installed. Please install Python 3.9+${NC}"
    exit 1
fi

# Check Python version
PYTHON_VERSION=$(python3 --version | cut -d " " -f 2 | cut -d "." -f 1,2)
echo "  ✓ Python version: $PYTHON_VERSION"

# Check if gcloud is installed
if ! command -v gcloud &> /dev/null; then
    echo -e "${YELLOW}⚠️  gcloud is not installed. You'll need it for GCP integration.${NC}"
    echo "    Install from: https://cloud.google.com/sdk/docs/install"
else
    echo "  ✓ gcloud is installed"
fi

echo -e "${GREEN}✅ Prerequisites check complete${NC}"
echo ""

# ============================================================================
# CREATE PROJECT STRUCTURE
# ============================================================================
echo "📁 Step 2: Creating project structure..."

# Create main directories
mkdir -p .github/workflows
mkdir -p src/{data,models,training,evaluation,deployment,pipelines}
mkdir -p tests/{unit,integration,smoke,e2e,data_validation}
mkdir -p scripts
mkdir -p docker
mkdir -p configs
mkdir -p notebooks
mkdir -p docs
mkdir -p deployments
mkdir -p reports/{evaluation,data_validation}

# Create __init__.py files
find src -type d -exec touch {}/__init__.py \;
find tests -type d -exec touch {}/__init__.py \;

echo -e "${GREEN}✅ Project structure created${NC}"
echo ""

# ============================================================================
# CREATE CONFIGURATION FILES
# ============================================================================
echo "⚙️  Step 3: Creating configuration files..."

# .gitignore
cat > .gitignore << 'EOF'
# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
venv/
env/
ENV/
*.egg-info/
dist/
build/

# Jupyter
.ipynb_checkpoints
*.ipynb_checkpoints

# IDEs
.vscode/
.idea/
*.swp
*.swo

# OS
.DS_Store
Thumbs.db

# GCP
*.json
!configs/*.json
key.json
credentials/

# Model artifacts
models/
*.h5
*.pb
*.pkl
saved_model/

# Logs
logs/
*.log

# Test coverage
.coverage
htmlcov/
.pytest_cache/
test-results.xml

# Environment
.env
.env.local

# Deployment records
deployments/*.json
EOF

# requirements.txt
cat > requirements.txt << 'EOF'
# Core ML libraries
tensorflow>=2.15.0
tensorflow-hub>=0.15.0
tensorflow-text>=2.15.0

# Google Cloud
google-cloud-aiplatform>=1.38.0
google-cloud-bigquery>=3.13.0
google-cloud-storage>=2.10.0

# Data processing
pandas>=2.0.0
numpy>=1.24.0
scikit-learn>=1.3.0

# Utilities
pyyaml>=6.0
python-dotenv>=1.0.0
click>=8.1.0
joblib>=1.3.0

# Monitoring
prometheus-client>=0.18.0
EOF

# requirements-dev.txt
cat > requirements-dev.txt << 'EOF'
# Testing
pytest>=7.4.0
pytest-cov>=4.1.0
pytest-mock>=3.11.0

# Linting and formatting
black>=23.7.0
flake8>=6.1.0
pylint>=2.17.0
isort>=5.12.0
mypy>=1.5.0
bandit>=1.7.5

# Documentation
sphinx>=7.1.0
sphinx-rtd-theme>=1.3.0

# Development tools
ipython>=8.14.0
jupyter>=1.0.0
pre-commit>=3.3.0
EOF

# .flake8
cat > .flake8 << 'EOF'
[flake8]
max-line-length = 100
exclude = 
    .git,
    __pycache__,
    venv,
    env,
    build,
    dist,
    *.egg-info
ignore = 
    E203,
    E501,
    W503,
per-file-ignores =
    __init__.py:F401
EOF

# pyproject.toml
cat > pyproject.toml << 'EOF'
[tool.black]
line-length = 100
target-version = ['py39', 'py310', 'py311']
include = '\.pyi?$'

[tool.isort]
profile = "black"
line_length = 100

[tool.pytest.ini_options]
testpaths = ["tests"]
python_files = "test_*.py"
python_classes = "Test*"
python_functions = "test_*"
markers = [
    "unit: Unit tests",
    "integration: Integration tests",
    "smoke: Smoke tests",
    "e2e: End-to-end tests",
]
EOF

# Makefile
cat > Makefile << 'EOF'
.PHONY: help install install-dev test lint format clean

help:
	@echo "Available commands:"
	@echo "  make install        Install production dependencies"
	@echo "  make install-dev    Install development dependencies"
	@echo "  make test          Run all tests"
	@echo "  make lint          Run linting checks"
	@echo "  make format        Format code"
	@echo "  make clean         Clean build artifacts"

install:
	pip install --upgrade pip
	pip install -r requirements.txt

install-dev:
	pip install --upgrade pip
	pip install -r requirements.txt
	pip install -r requirements-dev.txt

test:
	pytest tests/ -v --cov=src --cov-report=html --cov-report=term

test-unit:
	pytest tests/unit/ -v

test-integration:
	pytest tests/integration/ -v

lint:
	black --check src/ tests/
	isort --check-only src/ tests/
	flake8 src/ tests/
	pylint src/ || true
	mypy src/ || true

format:
	black src/ tests/ scripts/
	isort src/ tests/ scripts/

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} + || true
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.pyo" -delete
	rm -rf build/ dist/ .pytest_cache/ .coverage htmlcov/
EOF

# README.md
cat > README.md << 'EOF'
# Consumer Complaints MLOps Pipeline

Multi-class text classification for consumer complaints using BERT and Universal Sentence Encoder.

## 🚀 Quick Start

```bash
# Install dependencies
make install-dev

# Run tests
make test

# Train model
python src/training/train.py \
    --project-id YOUR_PROJECT_ID \
    --sample-size 5000 \
    --epochs 5 \
    --output-path ./models
```

## 📁 Project Structure

```
├── .github/workflows/    # CI/CD pipelines
├── src/                  # Source code
│   ├── data/            # Data loading and preprocessing
│   ├── models/          # Model definitions
│   ├── training/        # Training scripts
│   ├── evaluation/      # Evaluation scripts
│   └── deployment/      # Deployment scripts
├── tests/               # Tests
├── scripts/             # Utility scripts
└── configs/             # Configuration files
```

## 🧪 Testing

```bash
# Run all tests
make test

# Run specific test types
make test-unit
make test-integration
```

## 🎨 Code Quality

```bash
# Check code quality
make lint

# Auto-format code
make format
```

## 🚀 Deployment

See [docs/deployment.md](docs/deployment.md) for deployment instructions.

## 📚 Documentation

- [Setup Guide](docs/setup.md)
- [Training Guide](docs/training.md)
- [Deployment Guide](docs/deployment.md)

## 📄 License

Apache License 2.0
EOF

echo -e "${GREEN}✅ Configuration files created${NC}"
echo ""

# ============================================================================
# CREATE SAMPLE SOURCE FILES
# ============================================================================
echo "📝 Step 4: Creating sample source files..."

# src/models/__init__.py
cat > src/models/__init__.py << 'EOF'
"""Model definitions for consumer complaints classifier"""

from .bert_classifier import build_bert_classifier
from .use_classifier import build_use_classifier

__all__ = ['build_bert_classifier', 'build_use_classifier']
EOF

# src/models/use_classifier.py
cat > src/models/use_classifier.py << 'EOF'
"""Universal Sentence Encoder classifier"""

import tensorflow as tf
import tensorflow_hub as hub
from tensorflow.keras import layers, Model


def build_use_classifier(num_classes: int) -> Model:
    """
    Build Universal Sentence Encoder classifier
    
    Args:
        num_classes: Number of output classes
        
    Returns:
        Compiled Keras model
    """
    encoder_url = "https://tfhub.dev/google/universal-sentence-encoder/4"
    encoder = hub.KerasLayer(encoder_url, trainable=True, name='use_encoder')
    
    text_input = layers.Input(shape=(), dtype=tf.string, name='text')
    embedding = encoder(text_input)
    
    x = layers.Dense(256, activation='relu')(embedding)
    x = layers.Dropout(0.3)(x)
    x = layers.Dense(128, activation='relu')(x)
    x = layers.Dropout(0.2)(x)
    
    if num_classes == 2:
        output = layers.Dense(1, activation='sigmoid', name='output')(x)
    else:
        output = layers.Dense(num_classes, activation='softmax', name='output')(x)
    
    return Model(inputs=text_input, outputs=output, name='USE_Classifier')
EOF

echo -e "${GREEN}✅ Sample source files created${NC}"
echo ""

# ============================================================================
# CREATE SAMPLE TESTS
# ============================================================================
echo "🧪 Step 5: Creating sample tests..."

# tests/unit/test_models.py
cat > tests/unit/test_models.py << 'EOF'
"""Unit tests for model building"""

import pytest
import tensorflow as tf
from src.models.use_classifier import build_use_classifier


def test_use_model_builds():
    """Test USE model builds without errors"""
    model = build_use_classifier(num_classes=18)
    assert model is not None
    assert isinstance(model, tf.keras.Model)


def test_use_model_prediction():
    """Test USE model can make predictions"""
    model = build_use_classifier(num_classes=18)
    sample_text = ["This is a test complaint"]
    prediction = model.predict(sample_text, verbose=0)
    assert prediction.shape == (1, 18)
    assert (prediction >= 0).all()
    assert (prediction <= 1).all()
EOF

# tests/conftest.py
cat > tests/conftest.py << 'EOF'
"""Pytest configuration and fixtures"""

import pytest


@pytest.fixture
def sample_complaint():
    """Sample complaint text for testing"""
    return "I have a problem with my credit card charges"


@pytest.fixture
def sample_complaints():
    """Multiple sample complaints for testing"""
    return [
        "I dispute the charges on my credit report",
        "My mortgage payment was not processed correctly",
        "I am being harassed by debt collectors"
    ]
EOF

echo -e "${GREEN}✅ Sample tests created${NC}"
echo ""

# ============================================================================
# CREATE DOCUMENTATION
# ============================================================================
echo "📚 Step 6: Creating documentation..."

cat > docs/setup.md << 'EOF'
# Setup Guide

## Prerequisites

- Python 3.9+
- Google Cloud Platform account
- gcloud CLI installed

## Installation

1. Clone the repository
2. Create virtual environment
3. Install dependencies

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
make install-dev
```

## GCP Setup

See [deployment.md](deployment.md) for GCP configuration.
EOF

cat > docs/training.md << 'EOF'
# Training Guide

## Local Training

```bash
python src/training/train.py \
    --project-id YOUR_PROJECT \
    --sample-size 5000 \
    --epochs 5 \
    --output-path ./models
```

## Vertex AI Training

See [deployment.md](deployment.md) for cloud training.
EOF

echo -e "${GREEN}✅ Documentation created${NC}"
echo ""

# ============================================================================
# INITIALIZE GIT
# ============================================================================
echo "🔧 Step 7: Initializing Git repository..."

if [ ! -d ".git" ]; then
    git init
    git branch -M main
    git add .
    git commit -m "Initial commit: Complete MLOps project structure"
    echo -e "${GREEN}✅ Git repository initialized${NC}"
else
    echo -e "${YELLOW}⚠️  Git repository already exists${NC}"
fi

echo ""

# ============================================================================
# CREATE VIRTUAL ENVIRONMENT
# ============================================================================
echo "🐍 Step 8: Creating virtual environment..."

read -p "Do you want to create a virtual environment now? (y/n) " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    python3 -m venv venv
    echo -e "${GREEN}✅ Virtual environment created${NC}"
    echo ""
    echo "Activate it with:"
    echo "  source venv/bin/activate  # Linux/Mac"
    echo "  venv\\Scripts\\activate     # Windows"
else
    echo "Skipped virtual environment creation"
fi

echo ""

# ============================================================================
# COMPLETION
# ============================================================================
echo "============================================================================"
echo -e "${GREEN}🎉 PROJECT SETUP COMPLETE!${NC}"
echo "============================================================================"
echo ""
echo "📁 Project structure created in: $(pwd)"
echo ""
echo "🚀 Next steps:"
echo ""
echo "1. Activate virtual environment:"
echo "   source venv/bin/activate"
echo ""
echo "2. Install dependencies:"
echo "   make install-dev"
echo ""
echo "3. Run tests to verify setup:"
echo "   make test"
echo ""
echo "4. Setup GCP integration:"
echo "   - Create service account"
echo "   - Add GitHub secrets"
echo "   - See: docs/deployment.md"
echo ""
echo "5. Push to GitHub:"
echo "   git remote add origin https://github.com/YOUR_USERNAME/consumer-complaints-mlops.git"
echo "   git push -u origin main"
echo ""
echo "============================================================================"
echo "📚 Documentation: docs/"
echo "🧪 Run tests: make test"
echo "🎨 Format code: make format"
echo "🔍 Check quality: make lint"
echo "============================================================================"