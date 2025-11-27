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
    parser.add_argument("--class-mapping-file", type=str, required=True,
                        help="JSON file mapping test labels to model classes (REQUIRED)")
    
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
    if not class_mapping_file:
        logger.error("❌ --class-mapping-file is REQUIRED for ground truth label conversion")
        sys.exit(1)
        
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

def convert_ground_truth_to_model_classes(y_true, class_mapping, model_classes):
    """
    Convert ground truth labels to model's 10 predefined classes using class_mapping.
    Only labels that have a mapping are converted; unmapped labels are dropped.
    
    Args:
        y_true: List of label lists from test data
        class_mapping: Dict mapping test labels to model labels
        model_classes: List of 10 model class names
        
    Returns:
        Converted label lists containing only the 10 model classes
    """
    logger.info(f"🔄 Converting ground truth labels to 10 model classes...")
    logger.info(f"   Original unique labels: {len(set([item for sublist in y_true for item in sublist]))}")
    
    y_true_converted = []
    unmapped_count = 0
    dropped_samples = 0
    
    for labels in y_true:
        converted_labels = []
        for label in labels:
            if label in class_mapping:
                mapped_label = class_mapping[label]
                # Only include if it's one of the 10 model classes
                if mapped_label in model_classes:
                    converted_labels.append(mapped_label)
            else:
                unmapped_count += 1
        
        # Keep samples even if no labels map (will be empty list)
        if len(converted_labels) == 0 and len(labels) > 0:
            dropped_samples += 1
        y_true_converted.append(converted_labels)
    
    converted_unique = set([item for sublist in y_true_converted for item in sublist])
    logger.info(f"   Converted unique labels: {len(converted_unique)}")
    logger.info(f"   Expected model classes: {len(model_classes)}")
    
    if unmapped_count > 0:
        logger.warning(f"⚠️  {unmapped_count} label instances had no mapping and were dropped")
    if dropped_samples > 0:
        logger.warning(f"⚠️  {dropped_samples} samples had all labels dropped (no valid mappings)")
    
    if len(converted_unique) != len(model_classes):
        missing = set(model_classes) - converted_unique
        if missing:
            logger.warning(f"⚠️  Model classes not found in converted ground truth: {missing}")
    
    logger.info(f"✅ Ground truth conversion complete!")
    logger.info(f"   Converted classes: {sorted(list(converted_unique))}")
    
    return y_true_converted

def parse_predictions(raw_predictions, model_classes, prediction_threshold=0.3, top_k=None):
    """
    Converts raw Vertex AI predictions (probabilities/scores) to label lists.
    Predictions are kept as-is in model format (no conversion needed).
    
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

def calculate_metrics(y_true_original, y_pred_raw, model_classes, class_mapping, threshold, prediction_threshold=0.3, top_k=None):
    """Calculates metrics after converting ground truth to 10 model classes."""
    
    logger.info(f"📊 Original ground truth has {len(set([item for sublist in y_true_original for item in sublist]))} unique classes")
    logger.info(f"   Sample original labels: {list(set([item for sublist in y_true_original[:10] for item in sublist]))[:5]}")
    
    # Convert ground truth to 10 model classes using mapping
    y_true_converted = convert_ground_truth_to_model_classes(y_true_original, class_mapping, model_classes)
    
    # Parse predictions (already in model format, no conversion needed)
    logger.info(f"🎯 Parsing model predictions (threshold={prediction_threshold}, top_k={top_k})")
    y_pred = parse_predictions(y_pred_raw, model_classes, prediction_threshold, top_k)
    
    # Fit MultiLabelBinarizer on the 10 model classes
    mlb = MultiLabelBinarizer(classes=model_classes)
    mlb.fit([model_classes])  # Fit on all 10 classes to ensure consistent encoding
    
    logger.info(f"🏷️  Using {len(mlb.classes_)} model classes for evaluation")
    logger.info(f"   Classes: {list(mlb.classes_)}")
    
    # Transform both ground truth and predictions
    y_true_bin = mlb.transform(y_true_converted)
    y_pred_bin = mlb.transform(y_pred)
    
    # Log some examples
    logger.info("\n" + "="*70)
    logger.info("📋 Sample Predictions with Label Conversion (first 5):")
    logger.info("="*70)
    for i in range(min(5, len(y_true_original))):
        logger.info(f"\n   Sample {i+1}:")
        logger.info(f"     Ground Truth (Original):  {y_true_original[i]}")
        logger.info(f"     Ground Truth (Converted): {y_true_converted[i]}")
        logger.info(f"     Model Prediction:         {y_pred[i]}")
        match = set(y_true_converted[i]) == set(y_pred[i])
        logger.info(f"     Match: {'✅ CORRECT' if match else '❌ WRONG'}")
    
    logger.info("\n" + "="*70)

    # Calculate metrics
    f1_micro = f1_score(y_true_bin, y_pred_bin, average='micro', zero_division=0)
    f1_macro = f1_score(y_true_bin, y_pred_bin, average='macro', zero_division=0)
    precision = precision_score(y_true_bin, y_pred_bin, average='micro', zero_division=0)
    recall = recall_score(y_true_bin, y_pred_bin, average='micro', zero_division=0)
    hamming = hamming_loss(y_true_bin, y_pred_bin)
    
    # Calculate accuracy (exact match ratio)
    exact_matches = sum(1 for true, pred in zip(y_true_converted, y_pred) if set(true) == set(pred))
    accuracy = exact_matches / len(y_true_converted)
    
    # Show evaluation summary
    logger.info("\n" + "="*70)
    logger.info("📈 Evaluation Summary:")
    logger.info("="*70)
    logger.info(f"   Total samples evaluated: {len(y_true_original)}")
    logger.info(f"   Original test labels: {len(set([item for sublist in y_true_original for item in sublist]))} unique classes")
    logger.info(f"   Converted to: {len(model_classes)} model classes")
    logger.info(f"   Exact matches: {exact_matches}/{len(y_true_converted)} ({accuracy:.1%})")
    logger.info("="*70 + "\n")
    
    print("\n" + "="*50)
    print(f"📊 Multi-Label Classification Metrics")
    print("="*50)
    print(f"✅ F1 Score (Micro):     {f1_micro:.4f}")
    print(f"✅ F1 Score (Macro):     {f1_macro:.4f}")
    print(f"✅ Precision (Micro):    {precision:.4f}")
    print(f"✅ Recall (Micro):       {recall:.4f}")
    print(f"✅ Exact Match Accuracy: {accuracy:.4f}")
    print(f"✅ Hamming Loss:         {hamming:.4f}")
    print("="*50)
    
    # Show interpretation
    print("\n💡 Interpretation:")
    if f1_micro >= 0.90:
        print("   🌟 Excellent performance!")
    elif f1_micro >= 0.75:
        print("   ✅ Good performance")
    elif f1_micro >= 0.60:
        print("   ⚠️  Fair performance - consider model improvements")
    else:
        print("   ❌ Poor performance - model needs retraining or better mapping")
    
    print(f"   F1={f1_micro:.1%} means the model correctly classifies ~{f1_micro:.0%} of samples")
    print("")

    if f1_micro < threshold:
        logger.error(f"❌ Alert! F1-Micro ({f1_micro:.4f}) is below threshold ({threshold}).")
        logger.error(f"   Consider:")
        logger.error(f"   1. Reviewing class_mapping.json for correctness")
        logger.error(f"   2. Lowering --alert-threshold if current performance is acceptable")
        logger.error(f"   3. Retraining model with better data or architecture")
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
        logger.error("❌ Model classes are REQUIRED for evaluation")
        logger.error("   Please provide one of: --model-config, --model-classes-file, or --model-classes")
        sys.exit(1)
    
    if len(model_classes) != 10:
        logger.warning(f"⚠️  Expected 10 model classes, but got {len(model_classes)}")
    
    # 2. Load class mapping (REQUIRED)
    class_mapping = load_class_mapping(args.class_mapping_file)
    
    # 3. Load Data
    df, label_col, instances = load_data(args.dataset, args.split, args.batch_size)
    y_true = df[label_col].tolist()
    
    # 4. Get Predictions (raw probabilities/scores)
    y_pred_raw = get_predictions(args.project_id, args.region, args.endpoint_id, instances)
    
    # 5. Validation
    if len(y_true) != len(y_pred_raw):
        logger.error(f"Mismatch: {len(y_true)} true labels vs {len(y_pred_raw)} predictions.")
        sys.exit(1)
    
    logger.info("=" * 60)
    logger.info("📊 Starting Metric Calculation")
    logger.info("=" * 60)
    
    # 6. Calculate Metrics (converts ground truth, keeps predictions as-is)
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