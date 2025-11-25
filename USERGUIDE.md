# Consumer Complaint Category Routing Pipeline (TensorFlow/GCP MLOps)

## Project Overview
This project implements a scalable, end-to-end Machine Learning Operations (MLOps) pipeline for multi-class classification of consumer complaints. The primary goal is to automatically predict the correct category of a new complaint, enabling financial institutions to route the customer immediately to the correct internal department (e.g., Credit Card, Mortgage, Debt Collection).

By using deep learning ([Tensorflow](https://www.tensorflow.org/)/[Keras](https://keras.io/)), [Google Cloud services](https://console.cloud.google.com), and [Vertex AI](https://cloud.google.com/vertex-ai) platform the system ensures highly accurate predictions. It leverages a Standard LSTM for baseline sequence modeling and an Advanced BiLSTM + CNN hybrid architecture to capture both long-term dependencies and local textual features.


## Key Features
*   Advanced NLP Architectures: Implements two specific deep learning strategies using TensorFlow/Keras:

- Standard LSTM: A Long Short-Term Memory network for handling sequential text data.

- BiLSTM + CNN: A hybrid model combining Bidirectional LSTMs (for past/future context) with Convolutional Neural Networks (for extracting local n-gram features) to maximize classification accuracy.

*   CI/CD Automation: GitHub Actions manage the CI/CD process, automatically building the Docker image, pushing it to Google Artifact Registry, and triggering model training/deployment on every code update.

*   Scalable MLOps: Vertex AI is used for managed model training, hosting a centralized Model Registry, and deploying the prediction model to a scalable, low-latency Endpoint.

*   Data Lake & Persistence: Google BigQuery serves as the central data warehouse for storing raw consumer complaints, feature data, and final prediction outcomes.

*   Web Serving Layer: A Flask API provides the front-end interface for querying the Vertex AI endpoint, making predictions accessible to internal applications.


## Technology Stack

| Category    | Technology           | Purpose |
| :-------------: |:-------------:| :---------------------------:|
| ML Framework   | TensorFlow, Keras | Development of LSTM and BiLSTM+CNN models. |
| Cloud Platform     | Google Cloud Console     |   Primary cloud environment and resource management. |
| Data Warehouse | Google BigQuery    |   Model Registry, Model Training, and scalable Prediction Endpoints. |
| CI/CD | GitHub, Git Actions | DSource control and automated build/deploy workflows. |
| Application Layer    | Flask     |  Lightweight Python web framework for the prediction API. |
| Containerization | Docker    |    Packaging the application and environment for consistent deployment. |

## Quick Start Setup
Follow these steps to set up the project locally and connect to your Google Cloud environment.

**Prerequisites**

1. Python 3.8+
2. Docker
3. A Google Cloud Project with Billing Enabled.
4. The gcloud CLI installed and authenticated.
5. Necessary GCP APIs enabled (Vertex AI API, BigQuery API, Cloud Build API, Artifact Registry API).

**Step 1: Clone the Repository**

```bash
git clone [https://github.com/jvuhoang/consumer-complaints-mlops.git](https://github.com/jvuhoang/consumer-complaints-mlops.git)
cd consumer-complaints-mlops
```

**Step 2: Configure Environment Variables**
Set the following environment variables, typically as secrets in GitHub for CI/CD, and locally for development:

| Variable    | Description          | 
| :-------------: |:-------------:|
| GCP_PROJECT_ID  | Your Google Cloud Project ID. |
| GCP_REGION    | The region for Vertex AI (e.g., us-central1).   | 
| BQ_DATASET_NAME | BigQuery dataset name where tables reside (e.g., complaint_data).  |   
| ARTIFACT_REGISTRY_REPO | Name of the Docker repository in Artifact Registry.   |   


**Step 3: Run the Flask Prediction Server (Local Testing)**

To run the local API that connects to a deployed Vertex AI Endpoint:
1. Install dependencies: pip install -r requirements.txt 
2. Set endpoint details: export VERTEX_ENDPOINT_ID=<Your_Endpoint_ID>
3. Run the application: python app.py


**Step 4: Trigger the MLOps Pipeline**

The full MLOps pipeline is executed via GitHub Actions:
1. Commit and push your changes to the main branch on GitHub.
2. The GitHub Action workflow will:
*   Build the training container (packaging the LSTM and BiLSTM+CNN code).
*   Push the image to Google Artifact Registry.
*   Trigger a managed training job. 
*   Deploy the resulting model to a Vertex AI Endpoint if performance metrics are met.


## User Guide: 

For detailed instructions on setting up the GCP infrastructure, creating BigQuery tables, and configuring GitHub Actions for CI/CD, please refer to the comprehensive [USER_GUIDE.md]().


## Other files:
## Testing
```bash
pytest tests/
```

## Training
```bash
python src/training/train.py
```
# Test CI/CD
# CI/CD Test
