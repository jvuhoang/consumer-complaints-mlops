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
    
    # Prediction thresholding
    parser.add_argument("--prediction-threshold", type=float, default=0.5, 
                        help="Threshold for converting probabilities to labels (default: 0.5)")
    parser.add_argument("--top-k", type=int, default=None, 
                        help="If set, select top-k labels per sample instead of using threshold")
    
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
        predictions = []
        chunk_size = 1  # Process one by one if batching fails  
        
        for i in range(0, len(instances), chunk_size):
            # Wrap each string in a list to make it shape (N, 1)
            raw_batch = instances[i : i + chunk_size]
            batch = [[text] for text in raw_batch] 
            
            response = endpoint.predict(instances=batch)
            predictions.extend(response.predictions)
            
        return predictions

    except Exception as e:
        logger.error(f"Vertex AI Prediction failed: {e}")
        sys.exit(1)

def parse_predictions(raw_predictions, mlb, prediction_threshold=0.5, top_k=None):
    """
    Converts raw Vertex AI predictions (probabilities/scores) to label lists.
    
    Args:
        raw_predictions: List of prediction outputs from Vertex AI
        mlb: Fitted MultiLabelBinarizer with class names
        prediction_threshold: Threshold for converting probabilities to binary
        top_k: If set, select top-k labels instead of using threshold
    
    Returns:
        List of label lists (e.g., [['label1', 'label2'], ['label3'], ...])
    """
    parsed_predictions = []
    
    for pred in raw_predictions:
        # Handle different prediction formats
        if isinstance(pred, dict):
            # Format 1: {'scores': [...], 'labels': [...]}
            if 'scores' in pred and 'labels' in pred:
                scores = np.array(pred['scores'])
                labels = pred['labels']
            # Format 2: {label1: score1, label2: score2, ...}
            elif all(isinstance(v, (int, float)) for v in pred.values()):
                labels = list(pred.keys())
                scores = np.array(list(pred.values()))
            else:
                logger.error(f"Unknown prediction dict format: {pred}")
                sys.exit(1)
                
        elif isinstance(pred, (list, np.ndarray)):
            # Format 3: [score1, score2, ...] - need mlb.classes_ for label names
            scores = np.array(pred)
            if len(scores) == len(mlb.classes_):
                labels = mlb.classes_.tolist()
            else:
                logger.error(f"Prediction array length ({len(scores)}) doesn't match classes ({len(mlb.classes_)})")
                sys.exit(1)
        else:
            logger.error(f"Unknown prediction format: {type(pred)}")
            sys.exit(1)
        
        # Convert scores to labels
        if top_k is not None:
            # Select top-k highest scoring labels
            top_indices = np.argsort(scores)[-top_k:]
            selected_labels = [labels[i] for i in top_indices if scores[i] > 0]
        else:
            # Use threshold
            selected_labels = [labels[i] for i, score in enumerate(scores) if score >= prediction_threshold]
        
        # Ensure at least one label (select highest if none pass threshold)
        if len(selected_labels) == 0:
            max_idx = np.argmax(scores)
            selected_labels = [labels[max_idx]]
            logger.warning(f"No labels above threshold. Selected highest: {labels[max_idx]} (score: {scores[max_idx]:.4f})")
        
        parsed_predictions.append(selected_labels)
    
    return parsed_predictions

def calculate_metrics(y_true, y_pred_raw, threshold, prediction_threshold=0.5, top_k=None):
    """Calculates metrics and checks against threshold."""
    
    # First, fit MLB on ground truth to get all possible classes
    mlb = MultiLabelBinarizer()
    y_true_bin = mlb.fit_transform(y_true)
    
    logger.info(f"📊 Found {len(mlb.classes_)} unique classes in ground truth")
    logger.info(f"🏷️  Classes: {mlb.classes_[:10]}..." if len(mlb.classes_) > 10 else f"🏷️  Classes: {mlb.classes_}")
    
    # Inspect first raw prediction to determine format
    logger.info(f"🔍 Raw prediction format (first sample): {type(y_pred_raw[0])}")
    if isinstance(y_pred_raw[0], dict):
        logger.info(f"   Keys: {list(y_pred_raw[0].keys())}")
    elif isinstance(y_pred_raw[0], (list, np.ndarray)):
        logger.info(f"   Length: {len(y_pred_raw[0])}")
        logger.info(f"   Sample values: {y_pred_raw[0][:5]}")
    
    # Parse predictions from probabilities to labels
    logger.info(f"🎯 Converting predictions using threshold={prediction_threshold}, top_k={top_k}")
    y_pred = parse_predictions(y_pred_raw, mlb, prediction_threshold, top_k)
    
    # Log some examples
    logger.info("\n📋 Sample Predictions (first 3):")
    for i in range(min(3, len(y_true))):
        logger.info(f"   True: {y_true[i]}")
        logger.info(f"   Pred: {y_pred[i]}")
        logger.info("")
    
    # Transform predictions using the same MLB
    try:
        y_pred_bin = mlb.transform(y_pred)
    except ValueError as e:
        logger.warning(f"⚠️ Error transforming predictions: {e}")
        logger.warning("⚠️ New labels found in predictions. Refitting Binarizer.")
        mlb = MultiLabelBinarizer()
        all_labels = list(y_true) + list(y_pred)
        mlb.fit(all_labels)
        y_true_bin = mlb.transform(y_true)
        y_pred_bin = mlb.transform(y_pred)

    # Calculate metrics
    f1_micro = f1_score(y_true_bin, y_pred_bin, average='micro')
    f1_macro = f1_score(y_true_bin, y_pred_bin, average='macro')
    precision = precision_score(y_true_bin, y_pred_bin, average='micro')
    recall = recall_score(y_true_bin, y_pred_bin, average='micro')
    hamming = hamming_loss(y_true_bin, y_pred_bin)
    
    print("\n" + "="*50)
    print(f"📊 Multi-Label Classification Metrics")
    print("="*50)
    print(f"✅ F1 Score (Micro):     {f1_micro:.4f}")
    print(f"✅ F1 Score (Macro):     {f1_macro:.4f}")
    print(f"✅ Precision (Micro):    {precision:.4f}")
    print(f"✅ Recall (Micro):       {recall:.4f}")
    print(f"✅ Hamming Loss:         {hamming:.4f}")
    print("="*50 + "\n")

    if f1_micro < threshold:
        logger.error(f"❌ Alert! F1-Micro ({f1_micro:.4f}) is below threshold ({threshold}).")
        sys.exit(1)
    else:
        logger.info(f"✅ Performance is healthy (F1-Micro: {f1_micro:.4f} >= {threshold}).")

def main():
    args = parse_args()
    
    # 1. Load Data
    df, label_col, instances = load_data(args.dataset, args.split, args.batch_size)
    y_true = df[label_col].tolist()
    
    # 2. Get Predictions (raw probabilities/scores)
    y_pred_raw = get_predictions(args.project_id, args.region, args.endpoint_id, instances)
    
    # 3. Validation
    if len(y_true) != len(y_pred_raw):
        logger.error(f"Mismatch: {len(y_true)} true labels vs {len(y_pred_raw)} predictions.")
        sys.exit(1)
    
    # 4. Calculate Metrics (with prediction parsing)
    calculate_metrics(
        y_true, 
        y_pred_raw, 
        args.alert_threshold,
        args.prediction_threshold,
        args.top_k
    )

if __name__ == "__main__":
    main()