# Consumer Complaint Category Routing: Detailed User Guide

This guide provides a comprehensive walkthrough for setting up, configuring, and executing the MLOps pipeline for the Consumer Complaint Category Routing Pipeline.

This project utilizes TensorFlow/Keras for Deep Learning (LSTM & BiLSTM+CNN), Google Cloud Vertex AI for MLOps, and GitHub Actions for CI/CD automation.



## 1. Environment & Infrastructure Setup

**1.1. Prerequisites**
Ensure your local or cloud environment meets the following requirements:
*   Python: Version 3.8+
*   Google Cloud Platform (GCP):
        *   A Project with Billing Enabled.
        *   gcloud CLI installed and authenticated.
        *   APIs Enabled: Vertex AI API, BigQuery API, Cloud Build API, Artifact Registry API.


**1.2. Dataset Information**

The model is trained on the Consumer Finance Complaints dataset.

*   Source: HuggingFace - milesbutler/consumer_complaints
*   Volume: Approximately 278,000 records.
*   Key Attributes:
        *   Date received
        *   Product / Sub-product (Target Category)
        *   Issue / Sub-issue
        *   Consumer complaint narrative (Input Text)
        *   Company public response
        *   Company information (Name, State, Zip)
        *   Tags, Consent, Submission method
        *   Timely response, Consumer disputed, Complaint ID


## 2. Configuration & Credentials

**2.1. Cloning the Repository**

```bash
git clone [https://github.com/jvuhoang/consumer-complaints-mlops.git](https://github.com/jvuhoang/consumer-complaints-mlops.git)
cd consumer-complaints-mlops
```

**2.2. Environment Variables**

To enable the CI/CD pipeline and local development, you must configure the following environment variables. For GitHub Actions, add these as Repository Secrets.

| Variable    | Description          | 
| :-------------: |:-------------:|
| GCP_PROJECT_ID  | Your Google Cloud Project ID. |
| REGION    | The region for Vertex AI (e.g., us-central1).   | 
| GCP_SERVICE_ACCOUNT | The Service Account email ID on Google Cloud Platform used for operations.  |   
| GCP_WORKLOAD_IDENTITY_PROVIDER | The full identifier for the GCP Workload Identity Provider (for passwordless auth).   |   
| GCS_BUCKET | The name of the Google Cloud Storage Bucket used for artifacts.  | 


## 3. The MLOps Pipeline (CI/CD)

The pipeline is fully automated via GitHub Actions. It is triggered automatically when code is pushed to the main branch.

**3.1. Pipeline Stages**

1. Pre-commit Hooks:
*   Runs linters and code formatters.
*   Executes unit tests to ensure code integrity.

2. Training Job Trigger:
*   Builds the training container.
*   Executes the training job on Vertex AI using the Standard LSTM and Advanced BiLSTM + CNN architectures.

3. Model Upload:
*   Uploads the trained model artifacts to Google Cloud Storage.

4. Model Registration:
*   Registers the model version in the Vertex AI Model Registry.

5. Deployment:
*   Deploys the registered model to a Vertex AI Endpoint for real-time prediction.

6. Notification:
*   Sends a notification (e.g., email or Slack) regarding the deployment status.


## 4. Local Testing & Prediction Server

To test the application logic locally or run the web interface:
**1. Install Dependencies:**
```bash
pip install -r requirements.txt
```

**2. Configure Endpoint: Set the ID of your deployed Vertex AI endpoint.**
```bash
export VERTEX_ENDPOINT_ID=<Your_Endpoint_ID>
```

**3. Run the Flask App:**
```bash
python app.py
```
The application will launch, allowing you to send complaint text and receive category predictions via the API.


## 5. Model Monitoring & Metadata

After deployment, the system utilizes Google Cloud's native tools to ensure reliability.
**5.1. Vertex AI Model Monitoring**

To detect performance degradation, Model Monitoring is configured for the deployed endpoint:

*   Skew & Drift Detection: Alerts are triggered if the input data distribution deviates significantly from the training data.

*   Monitoring Jobs:
        *   Create monitoring jobs for all models under an endpoint.
        *   List active monitoring jobs.
        *   Pause or delete jobs when no longer needed.
*   Artifacts: Access statistical artifacts produced by the monitoring jobs for deep analysis.

**5.2. Metadata Tracking**

All pipeline executions are tracked using Vertex AI Metadata:

*   Parameters: Hyperparameters used during training.
*   Metrics: Accuracy, Loss, and other evaluation metrics.
*   Artifacts: Lineage tracking for Datasets, Models, and Endpoints.

All metadata is visualizable and queryable via the Google Cloud Console.






