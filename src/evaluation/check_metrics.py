import argparse
import sys
import numpy as np
import pandas as pd
import logging
from datasets import load_dataset
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics import f1_score, precision_score, recall_score, hamming_loss
# Import the necessary Google Cloud SDK module
from google.cloud import aiplatform

# --- Configuration ---
PROJECT_ID =  "${{ secrets.GCP_PROJECT_ID }}"
REGION =  "${{ secrets.REGION }}"
# ---------------------

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate Multi-Label Model Metrics from Vertex AI")
    parser.add_argument("--endpoint-id", type=str, required=True, help="Vertex AI Endpoint ID (e.g., 1234567890)")
    parser.add_argument("--dataset", type=str, default="milesbutler/consumer_complaints", help="Hugging Face dataset path")
    parser.add_argument("--split", type=str, default="test", help="Dataset split to use (e.g., test)")
    parser.add_argument("--alert-threshold", type=float, default=0.70, help="F1-Micro threshold for alerting")
    parser.add_argument("--batch-size", type=int, default=100, help="Number of samples to pull from the dataset and send for prediction.")
    return parser.parse_args()

def load_data(dataset_name, split, batch_size):
    """Loads the dataset and prepares data for prediction."""
    logger.info(f"📥 Loading dataset: {dataset_name}...")
    try:
        dataset = load_dataset(dataset_name)
    except Exception as e:
        logger.error(f"Failed to load dataset: {e}")
        sys.exit(1)

    if split not in dataset:
        logger.error(f"Split '{split}' not found. Exiting.")
        sys.exit(1)

    data = dataset[split].select(range(batch_size))
    df = data.to_pandas()
    
    label_col = next((col for col in ['Product', 'product', 'label', 'labels'] if col in df.columns), None)
    text_col = next((col for col in ['Consumer Complaint', 'text', 'narrative', 'consumer_complaint_narrative'] if col in df.columns), None)
    
    if not label_col or not text_col:
        logger.error(f"Could not find required columns. Label col: {label_col}, Text col: {text_col}")
        sys.exit(1)
        
    logger.info(f"✅ Data loaded. Using text column: '{text_col}' and label column: '{label_col}'")
    
    # Prepare true labels (y_true)
    df['target_list'] = df[label_col].apply(lambda x: [x] if isinstance(x, str) else x)
    
    # Prepare input instances for Vertex AI (assuming the model expects text strings)
    instances = df[text_col].tolist()
    
    return df, 'target_list', instances

def get_predictions(endpoint_id, instances):
    """Queries the Vertex AI Endpoint for predictions."""
    
    aiplatform.init(project=PROJECT_ID, location=REGION)
    
    logger.info(f"⚡️ Querying Vertex AI Endpoint: {endpoint_id} with {len(instances)} instances...")
    
    # Fetch the Endpoint object
    endpoint = aiplatform.Endpoint(endpoint_name=endpoint_id)

    # Convert the list of text instances into the format expected by the Vertex SDK
    # Each instance must be a dictionary if you have multiple features, 
    # but for simple text classification, it's often a list of dictionaries like:
    # [{"text": "complaint text 1"}, {"text": "complaint text 2"}]
    # NOTE: You may need to customize this input format based on your model's requirement!
    prediction_instances = [{"text": text} for text in instances]

    try:
        # Call the prediction service
        response = endpoint.predict(instances=prediction_instances)
        
        # The response structure depends entirely on your model's output format.
        # We assume the model returns a list of lists, where each inner list contains
        # the predicted class labels (e.g., [['Credit Card', 'Billing'], ['Mortgage']]).
        # If your model returns logits/probabilities, you'll need a post-processing step here.
        
        # This extracts the 'predictions' field, which is usually the custom output.
        y_pred = response.predictions 
        
        logger.info(f"✅ Successfully received {len(y_pred)} predictions.")
        return y_pred
        
    except Exception as e:
        logger.error(f"❌ Error during prediction request to Vertex AI: {e}")
        # Fail the monitoring job if the prediction service is unreachable or errors
        sys.exit(1)

def calculate_metrics(y_true, y_pred, threshold):
    """Calculates and reports multi-label metrics."""
    # (Remains the same as before, using MultiLabelBinarizer and calculating metrics)
    mlb = MultiLabelBinarizer()
    
    y_true_bin = mlb.fit_transform(y_true)
    
    try:
        y_pred_bin = mlb.transform(y_pred)
    except ValueError:
        # Re-fit for robustness in production monitoring
        all_labels = list(y_true) + list(y_pred)
        mlb = MultiLabelBinarizer()
        mlb.fit(all_labels)
        y_true_bin = mlb.transform(y_true)
        y_pred_bin = mlb.transform(y_pred)

    logger.info("📊 Calculating metrics...")
    
    f1_micro = f1_score(y_true_bin, y_pred_bin, average='micro')
    precision = precision_score(y_true_bin, y_pred_bin, average='micro')
    hamming = hamming_loss(y_true_bin, y_pred_bin)

    print("\n" + "="*40)
    print("🤖 MODEL PERFORMANCE REPORT (Vertex AI)")
    print("="*40)
    print(f"Dataset Size:   {len(y_true)}")
    print("-" * 40)
    print(f"✅ F1 Score (Micro):    {f1_micro:.4f}")
    print(f"   Precision (Micro):   {precision:.4f}")
    print(f"📉 Hamming Loss:        {hamming:.4f} (Lower is better)")
    print("="*40 + "\n")

    if f1_micro < threshold:
        logger.error(f"❌ Alert! F1-Micro ({f1_micro:.4f}) is below threshold ({threshold}).")
        sys.exit(1) 
    else:
        logger.info(f"✅ Performance is healthy (Above {threshold}).")

def main():
    args = parse_args()
    
    # 1. Load Data and prepare instances for prediction
    df, label_col_name, prediction_instances = load_data(
        args.dataset, args.split, args.batch_size
    )
    
    # 2. Get Ground Truth
    y_true = df[label_col_name].tolist()
    
    # 3. Get Predictions from Vertex AI
    y_pred = get_predictions(args.endpoint_id, prediction_instances)
    
    # 4. Run Checks
    calculate_metrics(y_true, y_pred, args.alert_threshold)

if __name__ == "__main__":
    main()