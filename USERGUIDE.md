# Consumer Complaint Category Routing: Detailed User Guide

This guide provides a comprehensive walkthrough for setting up, configuring, and executing the MLOps pipeline for the Consumer Complaint Category Routing Pipeline.

This project utilizes TensorFlow/Keras for Deep Learning (LSTM & BiLSTM+CNN), Google Cloud Vertex AI for MLOps, and GitHub Actions for CI/CD automation.



## 1. Environment & Infrastructure Setup

**1.1. Prerequisites**
Ensure your local or cloud environment meets the following requirements:
*   Python: Version 3.11+
*   Google Cloud Platform (GCP):
        *   A Project with Billing Enabled.
        *   gcloud CLI installed and authenticated.
        *   APIs Enabled: Vertex AI API, BigQuery API, Cloud Build API, Artifact Registry API.

### Required Software

- **Python 3.11 or higher**
  ```bash
  python --version  # Should output Python 3.11.x or higher
  ```

- **Git**
  ```bash
  git --version
  ```

- **Google Cloud SDK (gcloud CLI)**
  - Download: https://cloud.google.com/sdk/docs/install
  - Verify installation:
    ```bash
    gcloud --version
    ```

### Google Cloud Requirements

**GCP Project** with billing enabled
**Required APIs** enabled:
   ```bash
   gcloud services enable aiplatform.googleapis.com
   gcloud services enable bigquery.googleapis.com
   gcloud services enable cloudbuild.googleapis.com
   gcloud services enable artifactregistry.googleapis.com
   gcloud services enable storage.googleapis.com
   ```

**Service Account** with permissions:
   - Vertex AI User
   - BigQuery Data Editor
   - Storage Object Admin

**Vertex AI Endpoint** deployed with your model


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

**2.1: Clone the Repository**

```bash
git clone https://github.com/jvuhoang/consumer-complaints-mlops.git
cd consumer-complaints-mlops
```

**2.2: Create Python Virtual Environment**

**On macOS/Linux:**
```bash
# Create virtual environment
python3 -m venv venv

# Activate virtual environment
source venv/bin/activate
```

**On Windows:**
```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
venv\Scripts\activate
```

**Verify activation** - you should see `(venv)` at the beginning of your terminal prompt.

**2.3: Install Python Dependencies**

```bash
# Upgrade pip
pip install --upgrade pip

# Install required packages
pip install -r requirements.txt

# Install development dependencies (optional, for testing/linting)
pip install -r requirements-dev.txt
```

**2.4: Authenticate with Google Cloud**

**Option A: Using gcloud CLI (Recommended for Local Development)**

```bash
# Login to Google Cloud
gcloud auth login

# Set your project
gcloud config set project YOUR_PROJECT_ID

# Generate application default credentials
gcloud auth application-default login
```

**Option B: Using Service Account Key File**

a. Create a service account:
   ```bash
   gcloud iam service-accounts create mlops-app \
     --display-name="MLOps Application Service Account"
   ```

b. Grant necessary permissions:
   ```bash
   export PROJECT_ID=$(gcloud config get-value project)
   
   gcloud projects add-iam-policy-binding $PROJECT_ID \
     --member="serviceAccount:mlops-app@${PROJECT_ID}.iam.gserviceaccount.com" \
     --role="roles/aiplatform.user"
   
   gcloud projects add-iam-policy-binding $PROJECT_ID \
     --member="serviceAccount:mlops-app@${PROJECT_ID}.iam.gserviceaccount.com" \
     --role="roles/bigquery.dataEditor"
   ```

c. Download the service account key:
   ```bash
   mkdir -p credentials
   
   gcloud iam service-accounts keys create credentials/gcp-key.json \
     --iam-account=mlops-app@${PROJECT_ID}.iam.gserviceaccount.com
   ```

**SECURITY WARNING**: Never commit the `credentials/` directory to git!


**2.5 Configure Environment Variables**

To enable the CI/CD pipeline and local development, you must configure the following environment variables. For GitHub Actions, add these as Repository Secrets.

| Variable    | Description          | 
| :-------------: |:-------------:|
| GCP_PROJECT_ID  | Your Google Cloud Project ID. |
| REGION    | The region for Vertex AI (e.g., us-central1).   | 
| GCP_SERVICE_ACCOUNT | The Service Account email ID on Google Cloud Platform used for operations.  |   
| GCP_WORKLOAD_IDENTITY_PROVIDER | The full identifier for the GCP Workload Identity Provider (for passwordless auth).   |   
| GCS_BUCKET | The name of the Google Cloud Storage Bucket used for artifacts.  | 
| VERTEX AI ENDPOINT | ID of Vertex AI Endpoint used to deploy web app  |  

Create a `.env` file in the project root:

```bash
# Copy the example file
cp .env.example .env  # If you have an example file

# Or create manually
touch .env
```

Add the following configuration to `.env`:

```bash
# Google Cloud Project Configuration
GCP_PROJECT_ID=your-project-id-here
REGION=us-central1

# Vertex AI Endpoint
VERTEX_ENDPOINT_ID=projects/YOUR_PROJECT_ID/locations/REGION/endpoints/ENDPOINT_ID

# Service Account Configuration
GCP_SERVICE_ACCOUNT=mlops-app@your-project-id.iam.gserviceaccount.com
GOOGLE_APPLICATION_CREDENTIALS=./credentials/gcp-key.json

# Workload Identity (for GitHub Actions CI/CD)
GCP_WORKLOAD_IDENTITY_PROVIDER=projects/PROJECT_NUMBER/locations/global/workloadIdentityPools/POOL_ID/providers/PROVIDER_ID

# Google Cloud Storage
GCS_BUCKET=your-bucket-name

# Optional: BigQuery Configuration
BIGQUERY_DATASET=consumer_complaints
BIGQUERY_TABLE=predictions

# Flask Configuration
FLASK_ENV=development
FLASK_DEBUG=True
```

**How to find your Vertex AI Endpoint ID:**

```bash
# List all endpoints in your project
gcloud ai endpoints list --region=us-central1

# Copy the full endpoint resource name
# Format: projects/PROJECT_NUMBER/locations/REGION/endpoints/ENDPOINT_ID
```

### Step 6: Verify Setup

Run the verification script to check all configurations:

```bash
python -c "
import os
from google.cloud import aiplatform

print('✓ Python environment OK')

if os.getenv('GCP_PROJECT_ID'):
    print('✓ GCP_PROJECT_ID configured')
else:
    print('✗ GCP_PROJECT_ID not set')

if os.getenv('VERTEX_ENDPOINT_ID'):
    print('✓ VERTEX_ENDPOINT_ID configured')
else:
    print('✗ VERTEX_ENDPOINT_ID not set')

print('✓ Google Cloud libraries imported successfully')
print('
Setup verification complete!')
"
```

---

## 🏃 Running the Application

### Start the Flask Application

```bash
# Ensure virtual environment is activated
# You should see (venv) in your terminal

# Run the Flask app
python app.py
```

**Expected output:**
```
 * Serving Flask app 'app'
 * Debug mode: on
WARNING: This is a development server. Do not use it in a production deployment.
 * Running on http://127.0.0.1:8080
Press CTRL+C to quit
```



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
**4.1. Install Dependencies:**
```bash
pip install -r requirements.txt
```

**4.2. Configure Endpoint: Set the ID of your deployed Vertex AI endpoint.**
```bash
export VERTEX_ENDPOINT_ID=<Your_Endpoint_ID>
```

**4.3. Run the Flask App:**
```bash
python app.py
```
The application will launch, allowing you to send complaint text and receive category predictions via the API.




## 5. Model Monitoring & Metadata

### GitHub Actions Workflow

The project includes automated CI/CD pipelines in `.github/workflows/`:

a.**`mlops-pipeline.yml`**: Runs on push to `main`
   - Linting and code quality checks
   - Unit tests
   - Security scans
   - Package pretrained model
   - Uploads model to Vertex AI
   - Deploys to endpoint
   - Sends notifications

b.**`monitor.yml`**: Runs on schedule (daily)
   - Model performance monitoring
   - Drift detection
   - Alert generation



### Vertex AI Monitoring

```bash
# View endpoint metrics
gcloud ai endpoints describe ENDPOINT_ID \
  --region=us-central1

# View model deployment status
gcloud ai models list \
  --region=us-central1 \
  --filter="displayName:consumer-complaints"
```

### BigQuery Analytics

Query prediction logs:

```sql
-- View recent predictions
SELECT 
  timestamp,
  complaint_text,
  predicted_category,
  confidence_score
FROM `project.dataset.predictions`
ORDER BY timestamp DESC
LIMIT 100;

-- Category distribution
SELECT 
  predicted_category,
  COUNT(*) as count,
  AVG(confidence_score) as avg_confidence
FROM `project.dataset.predictions`
GROUP BY predicted_category
ORDER BY count DESC;
```






