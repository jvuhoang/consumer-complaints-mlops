# Consumer Complaint Category Routing Pipeline (TensorFlow/GCP MLOps)

## Project Overview
This project implements a scalable, end-to-end Machine Learning Operations (MLOps) pipeline for multi-class classification of consumer complaints. The primary goal is to automatically predict the correct category of a new complaint, enabling financial institutions to route the customer immediately to the correct internal department (e.g., Credit Card, Mortgage, Debt Collection).

By using deep learning ([Tensorflow](https://www.tensorflow.org/)/[Keras](https://keras.io/)), (LSTM and BiLSTM+CNN architectures), [Google Cloud Platform](https://console.cloud.google.com), and [Vertex AI](https://cloud.google.com/vertex-ai) platform the system ensures highly accurate predictions. It leverages a Standard LSTM for baseline sequence modeling and an Advanced BiLSTM + CNN hybrid architecture to capture both long-term dependencies and local textual features.

## Key Features
*   **Advanced NLP Architectures:** Implements two specific deep learning strategies using TensorFlow/Keras:
      - Standard LSTM: A Long Short-Term Memory network for handling sequential text data.
      - BiLSTM + CNN: A hybrid model combining Bidirectional LSTMs (for past/future context) with Convolutional Neural Networks (for extracting local n-gram features) to maximize classification accuracy.

*   **CI/CD Automation:** GitHub Actions manage the CI/CD process, automate tests, linting and code quality check before deploying model. Pipeline is triggered after every push to the repository. 

*   **Scalable MLOps:** Vertex AI is used for hosting a centralized Model Registry, and deploying the prediction model to a scalable, low-latency Endpoint.

*   **Data Lake & Persistence:** Google BigQuery serves as the central data warehouse for storing raw consumer complaints, feature data, and final prediction outcomes.

*   **Web Serving Layer:** A Flask API provides the front-end interface for querying the Vertex AI endpoint, making predictions accessible to internal applications.

## Dataset Management

**Consumer Complaints**
Source: originally comes from the US Consumer Finance Complaints

Link of dataset: https://huggingface.co/datasets/milesbutler/consumer_complaints

Data is related to consumer complaints about financial services

Nearly 278k records (rows)

18 Attributions:
*   Date received
*   Product, sub-product
*   Issue, sub-issue
*   Consumer complaint
*   Company public response
*   Company information: company, states, zip code
*   Tag
*   Consumer consent provided
*   Submitted via
*   Date sent to Company
*   Company response to consumer
*   Timely response
*   Consumer disputed
*   Complaint ID


## 🏗️ Architecture

```
User Request
    ↓
Flask API (app.py)
    ↓
Google Cloud Vertex AI Endpoint
    ↓
ML Model (BiLSTM+CNN / LSTM)
    ↓
Prediction Result
    ↓
BigQuery (Optional logging)
```

**Technology Stack:**

| Component | Technology | Purpose |
|-----------|-----------|---------|
| ML Framework | TensorFlow 2.15, Keras | Model development and training |
| Cloud Platform | Google Cloud Platform | Infrastructure and services |
| ML Platform | Vertex AI | Model registry and serving |
| Data Warehouse | BigQuery | Data storage and analytics |
| API Framework | Flask | Web API for predictions |
| CI/CD | GitHub Actions | Automated testing and deployment |

---


## Quick Start Setup
Follow these steps to set up the project locally and connect to your Google Cloud environment.

**Prerequisites**

1. Python 3.11
2. A Google Cloud Project with Billing Enabled.
3. The gcloud CLI installed and authenticated.
4. Necessary GCP APIs enabled (Vertex AI API, BigQuery API, Cloud Build API, Artifact Registry API).

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
| REGION    | The region for Vertex AI (e.g., us-central1).   | 
| GCP_SERVICE_ACCOUNT | The Service Account email ID on Google Cloud Platform used for operations.  |   
| GCP_WORKLOAD_IDENTITY_PROVIDER | The full identifier for the GCP Workload Identity Provider (for passwordless auth).   |   
| GCS_BUCKET | The name of the Google Cloud Storage Bucket used for artifacts.  |   
| VERTEX AI ENDPOINT | ID of Vertex AI Endpoint used to deploy web app  |   

**Step 3: Trigger the MLOps Pipeline**

The full MLOps pipeline is executed via GitHub Actions:
1. Commit and push your changes to the main branch on GitHub.
2. The GitHub Action workflow will:
*   Run pre-commit hooks (linters, code formatters, unit test).
*   Trigger a pre-trained model (LSTM and BiLSTM+CNN).
*   Upload model to Google Cloud.
*   Register model with Vertex AI. 
*   Deploy the resulting model to a Vertex AI Endpoint.
*   Send notification.

**Step 4: Run the Flask Prediction Server (Local Testing)**

To run the local API that connects to a deployed Vertex AI Endpoint:
1. Install dependencies: pip install -r requirements.txt 
2. Set endpoint details: export VERTEX_ENDPOINT_ID=<Your_Endpoint_ID>
3. Run the application: python app.py

## Model Monitoring

After a model is deployed in for prediction serving, continuous monitoring is set up to ensure that the model continue to perform as expected. Configure [monitor.yml](https://github.com/jvuhoang/consumer-complaints-mlops/blob/7f9ddef6c70a482398d84939b586c69b9abcdf7f/.github/workflows/monitor.yml) for performance metrics monitoring and drift detection:

1. Set performance metrics and drift threshold.
2. Set a monitoring frequency (default is every 24 hours).
3. Create alert if needed.


## Metadata Tracking

Parameters, metrics, artifacts and metadata stored by `Vertex AI` in [Cloud Console](https://console.cloud.google.com/vertex-ai).


## User Guide 

For detailed instructions on setting up the GCP infrastructure, creating BigQuery tables, and configuring GitHub Actions for CI/CD, please refer to the comprehensive [USER_GUIDE.md](https://github.com/jvuhoang/consumer-complaints-mlops/blob/main/USERGUIDE.md).


## Python Notebook for Model Training
Python Notebook: [DTI6302_Consumer_Complaints_TensorFlow.ipynb](https://github.com/jvuhoang/consumer-complaints-mlops/blob/main/DTI6302_Consumer_Complaints_TensorFlow-2.ipynb)
