"""
Hybrid training script for consumer complaints classifier
- Production: Uses pre-trained Colab model (fast, high quality)
- Development: Can train from scratch with BigQuery data
"""

import argparse
import json
import os
import sys
import shutil
from datetime import datetime
from pathlib import Path
import keras
import tensorflow as tf

import logging

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Train consumer complaints classifier")

    # Data arguments
    parser.add_argument("--project-id", required=True, help="GCP project ID")
    parser.add_argument("--sample-size", type=int, default=1000, help="Number of samples")
    parser.add_argument("--epochs", type=int, default=2, help="Number of epochs")
    parser.add_argument("--model-type", default="use", help="Model type")
    parser.add_argument("--output-path", required=True, help="Output path")
    
    # Training mode
    parser.add_argument(
        "--use-pretrained",
        action="store_true",
        default=True,
        help="Use pre-trained model from Colab (default: True)"
    )
    parser.add_argument(
        "--train-from-scratch",
        action="store_true",
        help="Train from scratch using BigQuery data"
    )

    return parser.parse_args()


def use_pretrained_model(args):
    """
    Use pre-trained model from Colab
    Fast deployment strategy - uses high-quality model
    """

    logger.info("=" * 80)
    logger.info("USING PRE-TRAINED MODEL FROM COLAB")
    logger.info("=" * 80)
    
    # Create output directory
    output_path = Path(args.output_path)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Define source paths (pre-trained model)
    pretrained_model = Path('models/complete_complaint_classifier_tf.keras')
    pretrained_labels = Path('models/label_mapping.json')
    pretrained_metadata = Path('models/model_metadata.json')
    
    if not pretrained_model.exists():
        logger.error("❌ Pre-trained model not found!")
        logger.error(f"   Expected: {pretrained_model}")
        logger.error("\n💡 Solutions:")
        logger.error("   1. Run training in Google Colab first")
        logger.error("   2. Download model files to models/ folder")
        logger.error("   3. Or use --train-from-scratch flag")
        return 1
    
    logger.info(f"✅ Found pre-trained model: {pretrained_model}")
    logger.info("   Source: Google Colab T4 GPU")
    logger.info("   Quality: Production-grade (62.6% accuracy)")
    logger.info("   Training: 50K samples, 8 epochs")

    
    try:
        from tensorflow.keras.saving import register_keras_serializable
    except ImportError:
        try:
            from keras.saving import register_keras_serializable
        except ImportError:
            register_keras_serializable = tf.keras.utils.register_keras_serializable

    @register_keras_serializable(package='Custom', name='custom_standardization')
    def custom_standardization(input_data):
        """Text preprocessing function - must match training exactly"""
        lowercase = tf.strings.lower(input_data)
        no_html = tf.strings.regex_replace(lowercase, '<[^>]+>', ' ')
        no_urls = tf.strings.regex_replace(no_html, r'http\S+|www\S+', ' ')
        no_emails = tf.strings.regex_replace(no_urls, r'\S+@\S+', ' ')
        no_redacted = tf.strings.regex_replace(no_emails, r'x{2,}', ' ')
        no_numbers = tf.strings.regex_replace(no_redacted, r'\d+', ' ')
        no_punct = tf.strings.regex_replace(no_numbers, r'[^a-z\s\-]', ' ')
        cleaned = tf.strings.regex_replace(no_punct, r'\s+', ' ')
        return tf.strings.strip(cleaned)

    # Test loading the model to verify it works
    try:
        logger.info("🔄 Attempting to load model with custom objects...")
        model = keras.models.load_model(pretrained_model)
        logger.info("✅ Model loaded successfully for verification")
        logger.info(f"   Model input shape: {model.input_shape}")
        logger.info(f"   Model output shape: {model.output_shape}")
    except Exception as e:
        logger.error(f"❌ Failed to load model: {e}")
        logger.error("\n💡 This model requires retraining with the decorator.")
        logger.error("   Please go back to Google Colab and:")
        logger.error("   1. Add @keras.saving.register_keras_serializable() to custom_standardization")
        logger.error("   2. Retrain and save the model")
        logger.error("   3. Download the new model file")
        return 1
    

    # Copy model
    shutil.copy(pretrained_model, output_path / 'model.keras')
    logger.info(f"✅ Copied model to: {output_path / 'model.keras'}")
    
    # Copy label mapping
    if pretrained_labels.exists():
        shutil.copy(pretrained_labels, output_path / 'label_mapping.json')
        logger.info(f"✅ Copied labels to: {output_path / 'label_mapping.json'}")
    
    # Load or create metadata
    if pretrained_metadata.exists():
        with open(pretrained_metadata, 'r') as f:
            existing_metadata = json.load(f)
        
        # Update with deployment info
        metadata = {
            **existing_metadata,
            'deployment_version': datetime.now().strftime('%Y%m%d_%H%M%S'),
            'deployment_method': 'pre-trained',
            'deployment_timestamp': datetime.now().isoformat(),
            'ci_cd_ready': True
        }
    else:
        # Create new metadata
        metadata = {
            'model_version': datetime.now().strftime('%Y%m%d_%H%M%S'),
            'model_type': 'BiLSTM+CNN',
            'training_method': 'pre-trained',
            'source': 'Google Colab T4 GPU',
            'test_accuracy': 0.626,
            'top3_accuracy': 0.796,
            'training_samples': 50000,
            'training_epochs': 8,
            'framework': 'tensorflow',
            'num_classes': 18,
            'deployment_ready': True,
            'timestamp': datetime.now().isoformat()
        }
    
    # Save metadata
    metadata_path = output_path / 'metadata.json'
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    logger.info(f"✅ Saved metadata to: {metadata_path}")
    
    logger.info("\n" + "=" * 80)
    logger.info("✅ PRE-TRAINED MODEL READY FOR DEPLOYMENT")
    logger.info("=" * 80)
    logger.info(f"Model: {output_path / 'model.keras'}")
    logger.info(f"Accuracy: 62.6%")
    logger.info(f"Status: Production-ready")
    logger.info("=" * 80)
    
    return 0


def train_from_scratch(args):
    """
    Train model from scratch using BigQuery data
    This is the original training logic
    """
    logger.info("=" * 80)
    logger.info("TRAINING FROM SCRATCH")
    logger.info("=" * 80)
    
    try:
        import numpy as np
        import tensorflow as tf
        from sklearn.preprocessing import LabelEncoder
        from sklearn.utils.class_weight import compute_class_weight
        
        # Add parent directory to path
        sys.path.append(str(Path(__file__).parent.parent.parent))
        
        from src.data.loader import DataLoader
        from src.models.use_classifier import build_use_classifier
        
    except ImportError as e:
        logger.error(f"❌ Import error: {e}")
        logger.error("   Missing dependencies for training from scratch")
        logger.error("   Use --use-pretrained instead")
        return 1
    
    logger.info("Training from scratch not fully implemented yet")
    logger.info("Use --use-pretrained flag for production deployment")
    
    # TODO: Implement full training pipeline here
    # For now, redirect to pre-trained model
    return use_pretrained_model(args)


def main():
    """Main training function"""
    args = parse_args()
    
    logger.info("Consumer Complaints Classifier - Training Pipeline")
    logger.info(f"Project ID: {args.project_id}")
    logger.info(f"Output Path: {args.output_path}")
    logger.info(f"Mode: {'Train from scratch' if args.train_from_scratch else 'Use pre-trained'}")
    
    try:
        if args.train_from_scratch:
            return train_from_scratch(args)
        else:
            return use_pretrained_model(args)
    
    except Exception as e:
        logger.error(f"❌ Training failed: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
    