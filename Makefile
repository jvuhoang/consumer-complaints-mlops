# Consumer Complaints MLOps - Makefile
# This Makefile provides convenient commands for development, testing, and deployment

.PHONY: help
help:
	@echo "Consumer Complaints MLOps - Available Commands:"
	@echo "=========================================="
	@echo "Setup & Environment:"
	@echo "  make install          : Install all dependencies"
	@echo "  make install-dev      : Install development dependencies"
	@echo "  make venv             : Create virtual environment"
	@echo "  make setup-gcp        : Run GCP environment setup script"
	@echo ""
	@echo "Code Quality:"
	@echo "  make format           : Format code with black and isort"
	@echo "  make lint             : Run linting with flake8"
	@echo "  make type-check       : Run type checking with mypy"
	@echo "  make style            : Run all code quality checks"
	@echo ""
	@echo "Testing:"
	@echo "  make test             : Run all tests"
	@echo "  make test-unit        : Run unit tests only"
	@echo "  make test-integration : Run integration tests only"
	@echo "  make test-coverage    : Run tests with coverage report"
	@echo "  make test-watch       : Run tests in watch mode"
	@echo ""
	@echo "Application:"
	@echo "  make run              : Run Flask application"
	@echo "  make run-prod         : Run with Gunicorn (production)"
	@echo "  make test-api         : Test API endpoints"
	@echo ""
	@echo "Vertex AI Deployment:"
	@echo "  make upload-model     : Upload pre-trained model to Vertex AI"
	@echo "  make create-endpoint  : Create new Vertex AI endpoint"
	@echo "  make deploy-endpoint  : Deploy model to Vertex AI endpoint"
	@echo "  make list-endpoints   : List all Vertex AI endpoints"
	@echo "  make list-models      : List all models in Vertex AI"
	@echo "  make undeploy-model   : Undeploy model from endpoint"
	@echo ""
	@echo "Cloud Run Deployment (Optional):"
	@echo "  make build-docker     : Build Docker image for Flask app"
	@echo "  make push-docker      : Push Docker image to registry"
	@echo "  make deploy-cloudrun  : Deploy Flask app to Cloud Run"
	@echo ""
	@echo "Data Management:"
	@echo "  make download-data    : Download dataset from source"
	@echo "  make process-data     : Process and prepare data"
	@echo "  make upload-data      : Upload data to BigQuery"
	@echo ""
	@echo "Utilities:"
	@echo "  make clean            : Clean temporary files"
	@echo "  make clean-all        : Clean everything including venv"
	@echo "  make logs             : View application logs"
	@echo "  make verify           : Verify environment setup"

# Variables
PYTHON := python3
PIP := pip
VENV := venv
PROJECT_NAME := consumer-complaints-mlops
REGION := us-central1
PORT := 8080

# Get GCP project ID and other configs from environment or gcloud
GCP_PROJECT_ID ?= $(shell gcloud config get-value project 2>/dev/null)
MODEL_PATH ?= ./models/bilstm_cnn_model.h5
MODEL_DISPLAY_NAME ?= consumer-complaints-bilstm-cnn
ENDPOINT_DISPLAY_NAME ?= consumer-complaints-endpoint

###################
# Setup & Environment
###################

.PHONY: venv
venv:
	@echo "Creating virtual environment..."
	$(PYTHON) -m venv $(VENV)
	@echo "Virtual environment created at ./$(VENV)"
	@echo "Activate it with: source $(VENV)/bin/activate"

.ONESHELL:
.PHONY: install
install:
	@echo "Installing dependencies..."
	$(PIP) install --upgrade pip setuptools wheel
	$(PIP) install -r requirements.txt
	@echo "Dependencies installed successfully!"

.ONESHELL:
.PHONY: install-dev
install-dev: install
	@echo "Installing development dependencies..."
	$(PIP) install -r requirements-dev.txt
	@echo "Development dependencies installed!"

.PHONY: setup-gcp
setup-gcp:
	@echo "Running GCP environment setup..."
	@if [ -f "setup_env.sh" ]; then \
		chmod +x setup_env.sh; \
		./setup_env.sh; \
	else \
		echo "Error: setup_env.sh not found!"; \
		exit 1; \
	fi

###################
# Code Quality
###################

.PHONY: format
format:
	@echo "Formatting code with black..."
	black src/ tests/ *.py
	@echo "Sorting imports with isort..."
	isort src/ tests/ *.py
	@echo "Code formatting complete!"

.PHONY: lint
lint:
	@echo "Running flake8 linter..."
	flake8 src/ tests/ *.py --max-line-length=100 --exclude=$(VENV)
	@echo "Linting complete!"

.PHONY: type-check
type-check:
	@echo "Running mypy type checker..."
	mypy src/ --ignore-missing-imports
	@echo "Type checking complete!"

.PHONY: style
style: format lint type-check
	@echo "All code quality checks passed!"

###################
# Testing
###################

.PHONY: test
test:
	@echo "Running all tests..."
	pytest tests/ -v
	@echo "Tests complete!"

.PHONY: test-unit
test-unit:
	@echo "Running unit tests..."
	pytest tests/ -v -m "not integration"
	@echo "Unit tests complete!"

.PHONY: test-integration
test-integration:
	@echo "Running integration tests..."
	pytest tests/ -v -m integration
	@echo "Integration tests complete!"

###################
# Application
###################

.PHONY: run
run:
	@echo "Starting Flask application..."
	@echo "Application will be available at http://localhost:$(PORT)"
	$(PYTHON) app.py

.PHONY: test-api
test-api:
	@echo "Testing API endpoints..."
	@echo "\nTesting health endpoint..."
	curl -s http://localhost:$(PORT)/health | python -m json.tool
	@echo "\n\nTesting prediction endpoint..."
	curl -s -X POST http://localhost:$(PORT)/predict \
		-H "Content-Type: application/json" \
		-d '{"complaint_text": "I was charged an incorrect fee on my credit card"}' \
		| python -m json.tool

###################
# Vertex AI Deployment
###################

.PHONY: upload-model
upload-model:
	@echo "Uploading pre-trained model to Vertex AI Model Registry..."
	@if [ -z "$(GCP_PROJECT_ID)" ]; then \
		echo "Error: GCP_PROJECT_ID not set!"; \
		exit 1; \
	fi
	@if [ ! -f "$(MODEL_PATH)" ]; then \
		echo "Error: Model file not found at $(MODEL_PATH)"; \
		exit 1; \
	fi
	@echo "Uploading model: $(MODEL_DISPLAY_NAME)"
	gcloud ai models upload \
		--region=$(REGION) \
		--display-name=$(MODEL_DISPLAY_NAME) \
		--container-image-uri=us-docker.pkg.dev/vertex-ai/prediction/tf2-cpu.2-15:latest \
		--artifact-uri=gs://$(GCS_BUCKET)/models/$(MODEL_DISPLAY_NAME)/ \
		--project=$(GCP_PROJECT_ID)
	@echo "Model uploaded successfully to Vertex AI!"

.PHONY: create-endpoint
create-endpoint:
	@echo "Creating new Vertex AI endpoint..."
	@if [ -z "$(GCP_PROJECT_ID)" ]; then \
		echo "Error: GCP_PROJECT_ID not set!"; \
		exit 1; \
	fi
	gcloud ai endpoints create \
		--region=$(REGION) \
		--display-name=$(ENDPOINT_DISPLAY_NAME) \
		--project=$(GCP_PROJECT_ID)
	@echo "Endpoint created successfully!"
	@echo "Run 'make list-endpoints' to see the endpoint ID"

.PHONY: deploy-endpoint
deploy-endpoint:
	@echo "Deploying model to Vertex AI endpoint..."
	@if [ -z "$(GCP_PROJECT_ID)" ]; then \
		echo "Error: GCP_PROJECT_ID not set!"; \
		exit 1; \
	fi
	@if [ -z "$(ENDPOINT_ID)" ]; then \
		echo "Error: ENDPOINT_ID not set!"; \
		echo "Usage: make deploy-endpoint ENDPOINT_ID=<endpoint-id> MODEL_ID=<model-id>"; \
		exit 1; \
	fi
	@if [ -z "$(MODEL_ID)" ]; then \
		echo "Error: MODEL_ID not set!"; \
		echo "Usage: make deploy-endpoint ENDPOINT_ID=<endpoint-id> MODEL_ID=<model-id>"; \
		exit 1; \
	fi
	gcloud ai endpoints deploy-model $(ENDPOINT_ID) \
		--region=$(REGION) \
		--model=$(MODEL_ID) \
		--display-name=$(MODEL_DISPLAY_NAME)-deployment \
		--machine-type=n1-standard-4 \
		--min-replica-count=1 \
		--max-replica-count=3 \
		--traffic-split=0=100 \
		--project=$(GCP_PROJECT_ID)
	@echo "Model deployed to endpoint successfully!"
	@echo "Update your .env file with: VERTEX_ENDPOINT_ID=projects/$(GCP_PROJECT_ID)/locations/$(REGION)/endpoints/$(ENDPOINT_ID)"

.PHONY: list-endpoints
list-endpoints:
	@echo "Listing all Vertex AI endpoints..."
	@gcloud ai endpoints list \
		--region=$(REGION) \
		--project=$(GCP_PROJECT_ID) \
		--format="table(name,displayName,createTime,updateTime)"

.PHONY: list-models
list-models:
	@echo "Listing all models in Vertex AI..."
	@gcloud ai models list \
		--region=$(REGION) \
		--project=$(GCP_PROJECT_ID) \
		--format="table(name,displayName,createTime,versionDescription)"

.PHONY: undeploy-model
undeploy-model:
	@echo "Undeploying model from endpoint..."
	@if [ -z "$(ENDPOINT_ID)" ]; then \
		echo "Error: ENDPOINT_ID not set!"; \
		echo "Usage: make undeploy-model ENDPOINT_ID=<endpoint-id> DEPLOYED_MODEL_ID=<deployed-model-id>"; \
		exit 1; \
	fi
	@if [ -z "$(DEPLOYED_MODEL_ID)" ]; then \
		echo "Error: DEPLOYED_MODEL_ID not set!"; \
		echo "Usage: make undeploy-model ENDPOINT_ID=<endpoint-id> DEPLOYED_MODEL_ID=<deployed-model-id>"; \
		exit 1; \
	fi
	gcloud ai endpoints undeploy-model $(ENDPOINT_ID) \
		--region=$(REGION) \
		--deployed-model-id=$(DEPLOYED_MODEL_ID) \
		--project=$(GCP_PROJECT_ID)
	@echo "Model undeployed successfully!"


###################
# Utilities
###################

.PHONY: clean
clean:
	@echo "Cleaning temporary files..."
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -delete
	find . -type d -name "*.egg-info" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name ".coverage" -delete
	find . -type d -name "htmlcov" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".mypy_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name ".DS_Store" -delete
	@echo "Cleanup complete!"

.PHONY: clean-all
clean-all: clean
	@echo "Removing virtual environment..."
	rm -rf $(VENV)
	@echo "Full cleanup complete!"

.PHONY: logs
logs:
	@echo "Viewing application logs..."
	@if [ -f "logs/app.log" ]; then \
		tail -f logs/app.log; \
	else \
		echo "No log file found at logs/app.log"; \
	fi

.PHONY: verify
verify:
	@echo "Verifying environment setup..."
	@if [ -f "verify_setup.py" ]; then \
		$(PYTHON) verify_setup.py; \
	else \
		@echo "Running basic verification..."; \
		$(PYTHON) -c "import tensorflow; import google.cloud.aiplatform; print('✓ All imports successful')"; \
	fi

###################
# CI/CD Helpers
###################

.PHONY: ci-test
ci-test: style test-coverage
	@echo "CI tests complete!"

.PHONY: pre-commit
pre-commit: style test
	@echo "Pre-commit checks passed!"

###################
# Quick Commands
###################

.PHONY: dev
dev: install-dev verify
	@echo "Development environment ready!"
	@echo "Run 'make run' to start the application"

.PHONY: deploy-vertex-ai
deploy-vertex-ai:
	@echo "=========================================="
	@echo "Complete Vertex AI Deployment Workflow"
	@echo "=========================================="
	@echo ""
	@echo "Step 1: Upload your pre-trained model..."
	@$(MAKE) upload-model
	@echo ""
	@echo "Step 2: List models to get MODEL_ID..."
	@$(MAKE) list-models
	@echo ""
	@echo "Step 3: Create an endpoint (if you don't have one)..."
	@read -p "Create new endpoint? (y/n): " answer; \
	if [ "$answer" = "y" ]; then \
		$(MAKE) create-endpoint; \
	fi
	@echo ""
	@echo "Step 4: Deploy model to endpoint"
	@echo "Run the following command with your IDs:"
	@echo "  make deploy-endpoint ENDPOINT_ID=<endpoint-id> MODEL_ID=<model-id>"
	@echo ""
	@echo "Step 5: Update your .env file with the endpoint ID"
	@echo "Step 6: Test with 'make run' and 'make test-api'"

.PHONY: prod
prod: install test deploy-cloudrun
	@echo "Production deployment complete!"
	@echo "Flask API is now running on Cloud Run"
	@echo "Models are served via Vertex AI endpoints"

.PHONY: all
all: install-dev test upload-model create-endpoint
	@echo "Full pipeline complete!"
	@echo "Note: Don't forget to deploy your model to the endpoint with:"
	@echo "  make deploy-endpoint ENDPOINT_ID=<id> MODEL_ID=<id>"