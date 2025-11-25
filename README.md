# Consumer Complaint Category Routing Pipeline (TensorFlow/GCP MLOps)

## Project Overview
This project implements a scalable, end-to-end Machine Learning Operations (MLOps) pipeline for multi-class classification of consumer complaints. The primary goal is to automatically predict the correct category of a new complaint, enabling financial institutions to route the customer immediately to the correct internal department (e.g., Credit Card, Mortgage, Debt Collection).

By using deep learning ([Tensorflow](https://www.tensorflow.org/)/[Keras](https://keras.io/)), [Google Cloud services](https://console.cloud.google.com), and [Vertex AI](https://cloud.google.com/vertex-ai) platform the system ensures highly accurate predictions. It leverages a Standard LSTM for baseline sequence modeling and an Advanced BiLSTM + CNN hybrid architecture to capture both long-term dependencies and local textual features.


## Key Features
*   **Advanced NLP Architectures:** Implements two specific deep learning strategies using TensorFlow/Keras:
      - Standard LSTM: A Long Short-Term Memory network for handling sequential text data.
      - BiLSTM + CNN: A hybrid model combining Bidirectional LSTMs (for past/future context) with Convolutional Neural Networks (for extracting local n-gram features) to maximize classification accuracy.
*   **CI/CD Automation:** GitHub Actions manage the CI/CD process, automatically building the Docker image, pushing it to Google Artifact Registry, and triggering model training/deployment on every code update.
*   **Scalable MLOps:** Vertex AI is used for managed model training, hosting a centralized Model Registry, and deploying the prediction model to a scalable, low-latency Endpoint.
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

## Technology Stack

| Category    | Technology           | Purpose |
| :------------- |:-------------:| :---------------------------:|
| ML Framework   | TensorFlow, Keras | Development of LSTM and BiLSTM+CNN models. |
| Cloud Platform     | Google Cloud Console     |   Primary cloud environment and resource management. |
| Data Warehouse | Google BigQuery    |   Model Registry, Model Training, and scalable Prediction Endpoints. |
| CI/CD | GitHub, Git Actions | DSource control and automated build/deploy workflows. |
| Application Layer    | Flask     |  Lightweight Python web framework for the prediction API. |


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
| REGION    | The region for Vertex AI (e.g., us-central1).   | 
| GCP_SERVICE_ACCOUNT | Service Account ID on Google Cloud Platform  |   
| GCP_WORKLOAD_IDENTITY_PROVIDER | Google Cloud Platform Workload Identity Provider   |   
| GCS_BUCKET | Number of Google Cloud Service Bucket   |   


**Step 3: Trigger the MLOps Pipeline**

The full MLOps pipeline is executed via GitHub Actions:
1. Commit and push your changes to the main branch on GitHub.
2. The GitHub Action workflow will:
*   Run pre-commit hooks (linters, code formatters, unit test).
*   Trigger a training job (run LSTM and BiLSTM+CNN code).
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

After a model is deployed in for prediction serving, continuous monitoring is set up to ensure that the model continue to perform as expected. Configure [Vertex AI Model Monitoring](https://cloud.google.com/vertex-ai/docs/model-monitoring/overview?hl=nn) for skew and drift detection:

1. Set skew and drift threshold.
2. Create a monitoring job for all the models under and endpoint.
3. List the monitoring jobs.
4. List artifacts produced by monitoring job.
5. Pause and delete the monitoring job.


## Metadata Tracking

Parameters, metrics, artifacts and metadata stored by `Vertex AI` in [Cloud Console](https://console.cloud.google.com/vertex-ai).


## User Guide: 

For detailed instructions on setting up the GCP infrastructure, creating BigQuery tables, and configuring GitHub Actions for CI/CD, please refer to the comprehensive [USER_GUIDE.md]().

