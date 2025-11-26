import argparse
import sys
import logging
import pandas as pd
import numpy as np
from datasets import load_dataset
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics import f1_score, precision_score, recall_score, hamming_loss
from google.cloud import aiplatform

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate Multi-Label Model Metrics from Vertex AI")
    
    # REQUIRED: Vertex AI Connection Details
    parser.add_argument("--project-id", type=str, required=True, help="Google Cloud Project ID")
    parser.add_argument("--region", type=str, default="us-central1", help="Google Cloud region (e.g., us-central1)")
    parser.add_argument("--endpoint-id", type=str, required=True, help="Vertex AI Endpoint ID")
    
    # Dataset Config
    parser.add_argument("--dataset", type=str, default="milesbutler/consumer_complaints", help="Hugging Face dataset path")
    parser.add_argument("--split", type=str, default="test", help="Dataset split to use")
    parser.add_argument("--batch-size", type=int, default=100, help="Number of samples to predict")
    
    # Alerting
    parser.add_argument("--alert-threshold", type=float, default=0.70, help="F1-Micro threshold for alerting")
    
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

    # Take only the first N rows for the spot check
    data = dataset[split].select(range(batch_size))
    df = data.to_pandas()
    
    # Dynamic column finding
    label_col = next((col for col in ['Product', 'product', 'label', 'labels'] if col in df.columns), None)
    text_col = next((col for col in ['Consumer Complaint', 'text', 'narrative', 'consumer_complaint_narrative'] if col in df.columns), None)
    
    if not label_col or not text_col:
        logger.error(f"Could not find required columns. Label col: {label_col}, Text col: {text_col}")
        sys.exit(1)
        
    logger.info(f"✅ Data loaded. Using text column: '{text_col}' and label column: '{label_col}'")
    
    # Prepare true labels (y_true) - Wrap single strings in list for MultiLabelBinarizer
    df['target_list'] = df[label_col].apply(lambda x: [x] if isinstance(x, str) else x)
    
    # Prepare input instances for Vertex AI
    instances = df[text_col].tolist()
    
    return df, 'target_list', instances

def get_predictions(project_id, region, endpoint_id, instances):
    """Queries the Vertex AI Endpoint for predictions."""
    
    # Initialize connection using the passed arguments
    aiplatform.init(project=project_id, location=region)
    
    logger.info(f"⚡️ Querying Vertex AI Endpoint: {endpoint_id} with {len(instances)} instances...")
    
    try:
        endpoint = aiplatform.Endpoint(endpoint_name=endpoint_id)
        
        # Vertex AI limits payload size, so we batch if necessary. 
        # Since batch_size is small (100) in args, we might send it all at once, 
        # but robust code splits it just in case.
        predictions = []
        chunk_size = 1  # Process one by one if batching fails  
        
        for i in range(0, len(instances), chunk_size):
            # OLD: batch = instances[i : i + chunk_size]
            # NEW: Wrap each string in a list to make it shape (N, 1)
            raw_batch = instances[i : i + chunk_size]
            batch = [[text] for text in raw_batch] 
            
            response = endpoint.predict(instances=batch)
            predictions.extend(response.predictions)
            
        return predictions

    except Exception as e:
        logger.error(f"Vertex AI Prediction failed: {e}")
        sys.exit(1)

def calculate_metrics(y_true, y_pred, threshold):
    """Calculates metrics and checks against threshold."""
    
    mlb = MultiLabelBinarizer()
    
    # Fit on Ground Truth
    y_true_bin = mlb.fit_transform(y_true)
    
    # Transform Predictions
    # Note: If model returns labels not seen in y_true, we handle it by refitting
    try:
        y_pred_bin = mlb.transform(y_pred)
    except ValueError:
        logger.warning("⚠️ New labels found in predictions not present in test batch. Refitting Binarizer.")
        mlb = MultiLabelBinarizer()
        all_labels = list(y_true) + list(y_pred)
        mlb.fit(all_labels)
        y_true_bin = mlb.transform(y_true)
        y_pred_bin = mlb.transform(y_pred)

    f1 = f1_score(y_true_bin, y_pred_bin, average='micro')
    
    print("\n" + "="*40)
    print(f"✅ F1 Score (Micro):    {f1:.4f}")
    print("="*40 + "\n")

    if f1 < threshold:
        logger.error(f"❌ Alert! F1-Micro ({f1:.4f}) is below threshold ({threshold}).")
        sys.exit(1)
    else:
        logger.info(f"✅ Performance is healthy.")

def main():
    args = parse_args()
    
    # 1. Load Data
    df, label_col, instances = load_data(args.dataset, args.split, args.batch_size)
    y_true = df[label_col].tolist()
    
    # 2. Get Predictions
    y_pred = get_predictions(args.project_id, args.region, args.endpoint_id, instances)
    
    # 3. Validation
    if len(y_true) != len(y_pred):
        logger.error(f"Mismatch: {len(y_true)} true labels vs {len(y_pred)} predictions.")
        sys.exit(1)
        
    # 4. Calc Metrics
    calculate_metrics(y_true, y_pred, args.alert_threshold)

if __name__ == "__main__":
    main()