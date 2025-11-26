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
    
    # Model Configuration - Supports multiple formats
    parser.add_argument("--model-config", type=str, default=None,
                        help="Path to model config JSON with id_to_product mapping (preferred)")
    parser.add_argument("--model-classes", type=str, default=None,
                        help="Comma-separated list of class names in model output order")
    parser.add_argument("--model-classes-file", type=str, default=None,
                        help="Path to text file with class names (one per line)")
    parser.add_argument("--class-mapping-file", type=str, default=None,
                        help="JSON file mapping test labels to model classes")
    
    # Prediction thresholding
    parser.add_argument("--prediction-threshold", type=float, default=0.3, 
                        help="Threshold for converting probabilities to labels (default: 0.3)")
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

def load_model_classes_from_config(config_path):
    """Load model classes from a JSON config file with id_to_product mapping."""
    import json
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    if 'id_to_product' in config:
        # Sort by ID to ensure correct order
        id_to_product = config['id_to_product']
        sorted_ids = sorted(id_to_product.keys(), key=lambda x: int(x))
        classes = [id_to_product[id_str] for id_str in sorted_ids]
        logger.info(f"📋 Loaded {len(classes)} model classes from config JSON: {config_path}")
        logger.info(f"   Classes: {classes}")
        return classes
    else:
        logger.error(f"Config file {config_path} does not contain 'id_to_product' key")
        sys.exit(1)

def load_model_classes(model_config, model_classes_arg, model_classes_file):
    """Load the model's class names from various sources."""
    # Priority: 1. Config JSON, 2. Classes file, 3. Argument
    if model_config:
        return load_model_classes_from_config(model_config)
    elif model_classes_file:
        with open(model_classes_file, 'r') as f:
            classes = [line.strip() for line in f if line.strip()]
        logger.info(f"📋 Loaded {len(classes)} model classes from file: {model_classes_file}")
        logger.info(f"   Classes: {classes}")
        return classes
    elif model_classes_arg:
        classes = [c.strip() for c in model_classes_arg.split(',')]
        logger.info(f"📋 Loaded {len(classes)} model classes from argument")
        logger.info(f"   Classes: {classes}")
        return classes
    else:
        return None

def load_class_mapping(class_mapping_file):
    """Load class name mapping from JSON file."""
    if class_mapping_file:
        import json
        try:
            with open(class_mapping_file, 'r') as f:
                mapping = json.load(f)
            logger.info(f"🔄 Loaded class mapping with {len(mapping)} entries from: {class_mapping_file}")
            # Show sample mappings
            sample_items = list(mapping.items())[:3]
            for key, val in sample_items:
                logger.info(f"   '{key}' → '{val}'")
            if len(mapping) > 3:
                logger.info(f"   ... and {len(mapping) - 3} more mappings")
            return mapping
        except FileNotFoundError:
            logger.error(f"Class mapping file not found: {class_mapping_file}")
            sys.exit(1)
        except json.JSONDecodeError as e:
            logger.error(f"Invalid JSON in class mapping file: {e}")
            sys.exit(1)
    return None

def apply_class_mapping(labels, class_mapping):
    """Apply class name mapping to predicted labels."""
    if class_mapping is None:
        return labels
    return [class_mapping.get(label, label) for label in labels]

def infer_model_classes(raw_predictions, dataset_name):
    """
    Attempt to infer model classes by looking at common dataset schemas.
    This is a fallback when user doesn't provide class names.
    """
    # Check prediction format
    first_pred = raw_predictions[0]
    
    if isinstance(first_pred, dict) and 'labels' in first_pred:
        # Model returns label names directly
        return None
    
    # For consumer_complaints dataset, use top 10 most common classes
    if 'consumer_complaints' in dataset_name.lower():
        # These are the most common 10 classes in the consumer complaints dataset
        common_classes = [
            'Credit reporting, credit repair services, or other personal consumer reports',
            'Debt collection',
            'Credit card or prepaid card',
            'Checking or savings account',
            'Mortgage',
            'Credit card',
            'Bank account or service',
            'Student loan',
            'Vehicle loan or lease',
            'Money transfer, virtual currency, or money service'
        ]
        logger.warning(f"⚠️ Using inferred top-10 classes for consumer_complaints dataset")
        logger.warning(f"⚠️ For accurate results, provide --model-classes or --model-classes-file")
        return common_classes
    
    return None

def parse_predictions(raw_predictions, model_classes, prediction_threshold=0.3, top_k=None):
    """
    Converts raw Vertex AI predictions (probabilities/scores) to label lists.
    
    Args:
        raw_predictions: List of prediction outputs from Vertex AI
        model_classes: List of class names in the order model outputs them
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
            # Format 3: [score1, score2, ...] - need model_classes for label names
            scores = np.array(pred)
            if model_classes is None:
                logger.error(f"Model outputs {len(scores)} probabilities but no class names provided.")
                logger.error("Please use --model-classes or --model-classes-file to specify class names.")
                sys.exit(1)
            if len(scores) != len(model_classes):
                logger.error(f"Prediction array length ({len(scores)}) doesn't match provided classes ({len(model_classes)})")
                sys.exit(1)
            labels = model_classes
        else:
            logger.error(f"Unknown prediction format: {type(pred)}")
            sys.exit(1)
        
        # Convert scores to labels
        if top_k is not None:
            # Select top-k highest scoring labels
            top_indices = np.argsort(scores)[-top_k:][::-1]  # Reverse to get highest first
            selected_labels = [labels[i] for i in top_indices if scores[i] > 0]
        else:
            # Use threshold
            selected_labels = [labels[i] for i, score in enumerate(scores) if score >= prediction_threshold]
        
        # Ensure at least one label (select highest if none pass threshold)
        if len(selected_labels) == 0:
            max_idx = np.argmax(scores)
            selected_labels = [labels[max_idx]]
            if scores[max_idx] < prediction_threshold:
                logger.debug(f"No labels above threshold ({prediction_threshold}). Selected highest: {labels[max_idx]} (score: {scores[max_idx]:.4f})")
        
        parsed_predictions.append(selected_labels)
    
    return parsed_predictions

def calculate_metrics(y_true, y_pred_raw, model_classes, class_mapping, threshold, prediction_threshold=0.3, top_k=None):
    """Calculates metrics and checks against threshold."""
    
    logger.info(f"📊 Original ground truth has {len(set([item for sublist in y_true for item in sublist]))} unique classes")
    
    # Apply class mapping to ground truth (test data) to standardize to model classes
    if class_mapping:
        logger.info(f"🔄 Applying class name mapping to ground truth...")
        y_true_mapped = [apply_class_mapping(labels, class_mapping) for labels in y_true]
    else:
        y_true_mapped = y_true
    
    # Now fit MLB on the MAPPED ground truth
    mlb = MultiLabelBinarizer()
    y_true_bin = mlb.fit_transform(y_true_mapped)
    
    logger.info(f"🏷️  After mapping, ground truth has {len(mlb.classes_)} unique classes (should match model)")
    logger.info(f"    Classes: {list(mlb.classes_[:10])}..." if len(mlb.classes_) > 10 else f"    Classes: {list(mlb.classes_)}")
    
    # Inspect first raw prediction to determine format
    logger.info(f"🔍 Raw prediction format (first sample): {type(y_pred_raw[0])}")
    if isinstance(y_pred_raw[0], dict):
        logger.info(f"   Keys: {list(y_pred_raw[0].keys())}")
    elif isinstance(y_pred_raw[0], (list, np.ndarray)):
        logger.info(f"   Length: {len(y_pred_raw[0])}")
        logger.info(f"   Sample values: {y_pred_raw[0][:5]}")
    
    if model_classes:
        logger.info(f"🎓 Model trained on {len(model_classes)} classes:")
        logger.info(f"   {model_classes[:5]}..." if len(model_classes) > 5 else f"   {model_classes}")
    
    # Parse predictions from probabilities to labels (these are already in model format)
    logger.info(f"🎯 Converting predictions using threshold={prediction_threshold}, top_k={top_k}")
    y_pred = parse_predictions(y_pred_raw, model_classes, prediction_threshold, top_k)
    
    # Log some examples
    logger.info("\n📋 Sample Predictions (first 5):")
    for i in range(min(5, len(y_true))):
        logger.info(f"   Sample {i+1}:")
        logger.info(f"     Ground Truth (Original): {y_true[i]}")
        if class_mapping:
            logger.info(f"     Ground Truth (Mapped):   {y_true_mapped[i]}")
        logger.info(f"     Prediction:               {y_pred[i]}")
        logger.info(f"     Match: {'✅' if set(y_true_mapped[i]) == set(y_pred[i]) else '❌'}")
    
    # Transform predictions using the same MLB (fitted on mapped ground truth)
    try:
        y_pred_bin = mlb.transform(y_pred)
    except ValueError as e:
        logger.warning(f"⚠️ Error transforming predictions: {e}")
        logger.warning("⚠️ New labels found in predictions. Refitting Binarizer.")
        
        # Find which labels are missing
        all_pred_labels = set()
        for pred_list in y_pred:
            all_pred_labels.update(pred_list)
        missing_labels = all_pred_labels - set(mlb.classes_)
        if missing_labels:
            logger.warning(f"⚠️ Labels in predictions but not in ground truth: {missing_labels}")
            logger.warning(f"⚠️ This will hurt your F1 score. Check your model classes or mapping!")
        
        mlb = MultiLabelBinarizer()
        all_labels = list(y_true_mapped) + list(y_pred)
        mlb.fit(all_labels)
        y_true_bin = mlb.transform(y_true_mapped)
        y_pred_bin = mlb.transform(y_pred)

    # Calculate metrics
    f1_micro = f1_score(y_true_bin, y_pred_bin, average='micro', zero_division=0)
    f1_macro = f1_score(y_true_bin, y_pred_bin, average='macro', zero_division=0)
    precision = precision_score(y_true_bin, y_pred_bin, average='micro', zero_division=0)
    recall = recall_score(y_true_bin, y_pred_bin, average='micro', zero_division=0)
    hamming = hamming_loss(y_true_bin, y_pred_bin)
    
    # Calculate accuracy (exact match ratio)
    exact_matches = sum(1 for true, pred in zip(y_true_mapped, y_pred) if set(true) == set(pred))
    accuracy = exact_matches / len(y_true)
    
    print("\n" + "="*50)
    print(f"📊 Multi-Label Classification Metrics")
    print("="*50)
    print(f"✅ F1 Score (Micro):     {f1_micro:.4f}")
    print(f"✅ F1 Score (Macro):     {f1_macro:.4f}")
    print(f"✅ Precision (Micro):    {precision:.4f}")
    print(f"✅ Recall (Micro):       {recall:.4f}")
    print(f"✅ Exact Match Accuracy: {accuracy:.4f}")
    print(f"✅ Hamming Loss:         {hamming:.4f}")
    print("="*50 + "\n")

    if f1_micro < threshold:
        logger.error(f"❌ Alert! F1-Micro ({f1_micro:.4f}) is below threshold ({threshold}).")
        sys.exit(1)
    else:
        logger.info(f"✅ Performance is healthy (F1-Micro: {f1_micro:.4f} >= {threshold}).")

def main():
    args = parse_args()
    
    # 1. Load model classes (supports multiple formats)
    logger.info("=" * 60)
    logger.info("🚀 Starting Model Evaluation")
    logger.info("=" * 60)
    
    model_classes = load_model_classes(args.model_config, args.model_classes, args.model_classes_file)
    
    if model_classes is None:
        logger.warning("⚠️  No model classes provided via --model-config, --model-classes-file, or --model-classes")
        logger.warning("⚠️  Will attempt to infer classes, but results may be inaccurate")
    
    # 2. Load class mapping
    class_mapping = load_class_mapping(args.class_mapping_file)
    
    if class_mapping is None:
        logger.warning("⚠️  No class mapping provided via --class-mapping-file")
        logger.warning("⚠️  Assuming test labels match model classes exactly")
    
    # 3. Load Data
    df, label_col, instances = load_data(args.dataset, args.split, args.batch_size)
    y_true = df[label_col].tolist()
    
    # 4. Get Predictions (raw probabilities/scores)
    y_pred_raw = get_predictions(args.project_id, args.region, args.endpoint_id, instances)
    
    # 5. Infer model classes if not provided
    if model_classes is None:
        logger.warning("⚠️ Attempting to infer model classes from dataset...")
        model_classes = infer_model_classes(y_pred_raw, args.dataset)
        if model_classes is None:
            logger.error("❌ Could not infer model classes. Please provide --model-config or --model-classes-file")
            sys.exit(1)
    
    # 6. Validation
    if len(y_true) != len(y_pred_raw):
        logger.error(f"Mismatch: {len(y_true)} true labels vs {len(y_pred_raw)} predictions.")
        sys.exit(1)
    
    logger.info("=" * 60)
    logger.info("📊 Starting Metric Calculation")
    logger.info("=" * 60)
    
    # 7. Calculate Metrics (with prediction parsing and mapping)
    calculate_metrics(
        y_true, 
        y_pred_raw,
        model_classes,
        class_mapping,
        args.alert_threshold,
        args.prediction_threshold,
        args.top_k
    )

if __name__ == "__main__":
    main()