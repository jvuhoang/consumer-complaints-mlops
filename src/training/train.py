"""
Main training script for consumer complaints classifier
Can be run locally or on Vertex AI
"""

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_class_weight

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent.parent))

import logging

from src.data.loader import DataLoader
from src.models.bert_classifier import build_bert_classifier
from src.models.use_classifier import build_use_classifier

# from src.training.trainer import Trainer

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Train consumer complaints classifier")

    # Data arguments
    parser.add_argument("--project-id", required=True, help="GCP project ID")
    parser.add_argument(
        "--dataset-id", default="consumer_complaints", help="BigQuery dataset ID"
    )
    parser.add_argument("--table-id", default="complaints", help="BigQuery table ID")
    parser.add_argument("--target-column", default="Product", help="Target column name")
    parser.add_argument(
        "--sample-size", type=int, default=None, help="Number of samples to use"
    )

    # Model arguments
    parser.add_argument(
        "--model-type", default="use", choices=["bert", "use"], help="Model type"
    )
    parser.add_argument(
        "--max-seq-length", type=int, default=128, help="Max sequence length"
    )

    # Training arguments
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    parser.add_argument("--epochs", type=int, default=10, help="Number of epochs")
    parser.add_argument(
        "--learning-rate", type=float, default=0.001, help="Learning rate"
    )
    parser.add_argument(
        "--validation-split", type=float, default=0.2, help="Validation split"
    )
    parser.add_argument("--test-split", type=float, default=0.2, help="Test split")

    # Output arguments
    parser.add_argument(
        "--output-path", required=True, help="Output path (local or GCS)"
    )
    parser.add_argument(
        "--save-format", default="tf", choices=["tf", "h5"], help="Save format"
    )

    # Other arguments
    parser.add_argument(
        "--use-class-weights", action="store_true", help="Use class weights"
    )
    parser.add_argument(
        "--early-stopping", action="store_true", help="Use early stopping"
    )
    parser.add_argument(
        "--patience", type=int, default=3, help="Early stopping patience"
    )
    parser.add_argument("--verbose", type=int, default=1, help="Verbosity level")

    return parser.parse_args()


def main():
    """Main training function"""
    args = parse_args()

    logger.info("=" * 80)
    logger.info("CONSUMER COMPLAINTS CLASSIFIER - TRAINING")
    logger.info("=" * 80)
    logger.info(f"Configuration:")
    for arg, value in vars(args).items():
        logger.info(f"  {arg}: {value}")
    logger.info("=" * 80)

    # Set random seeds for reproducibility
    np.random.seed(42)
    tf.random.set_seed(42)

    # =========================================================================
    # STEP 1: Load Data
    # =========================================================================
    logger.info("\n📥 STEP 1: Loading data from BigQuery...")

    loader = DataLoader(
        project_id=args.project_id, dataset_id=args.dataset_id, table_id=args.table_id
    )

    df = loader.load_data(target_column=args.target_column, limit=args.sample_size)

    logger.info(f"✅ Data loaded: {len(df):,} samples")

    # =========================================================================
    # STEP 2: Prepare Data
    # =========================================================================
    logger.info("\n🔧 STEP 2: Preparing data for training...")

    train_df, val_df, test_df = loader.prepare_for_training(
        df, test_size=args.test_split, val_size=args.validation_split
    )

    # Encode labels
    label_encoder = LabelEncoder()
    train_df["label"] = label_encoder.fit_transform(train_df["target"])
    val_df["label"] = label_encoder.transform(val_df["target"])
    test_df["label"] = label_encoder.transform(test_df["target"])

    num_classes = len(label_encoder.classes_)
    logger.info(f"✅ Number of classes: {num_classes}")
    logger.info(
        f"   Classes: {label_encoder.classes_[:5]}..."
        if num_classes > 5
        else f"   Classes: {list(label_encoder.classes_)}"
    )

    # Compute class weights
    class_weight_dict = None
    if args.use_class_weights:
        class_weights = compute_class_weight(
            "balanced", classes=np.unique(train_df["label"]), y=train_df["label"]
        )
        class_weight_dict = dict(enumerate(class_weights))
        logger.info(
            f"✅ Class weights computed (range: {min(class_weights):.2f} - {max(class_weights):.2f})"
        )

    # =========================================================================
    # STEP 3: Build Model
    # =========================================================================
    logger.info(f"\n🏗️ STEP 3: Building {args.model_type.upper()} model...")

    if args.model_type == "bert":
        model = build_bert_classifier(
            num_classes=num_classes, max_seq_length=args.max_seq_length, trainable=True
        )
    else:
        model = build_use_classifier(num_classes=num_classes)

    logger.info("✅ Model built successfully")

    # Compile model
    if num_classes == 2:
        loss = "binary_crossentropy"
        metrics = ["accuracy", tf.keras.metrics.AUC(name="auc")]
    else:
        loss = "sparse_categorical_crossentropy"
        metrics = [
            "accuracy",
            tf.keras.metrics.SparseTopKCategoricalAccuracy(k=3, name="top_3_acc"),
        ]

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=args.learning_rate),
        loss=loss,
        metrics=metrics,
    )

    logger.info("✅ Model compiled")

    # =========================================================================
    # STEP 4: Create Datasets
    # =========================================================================
    logger.info("\n🔄 STEP 4: Creating TensorFlow datasets...")

    def create_dataset(df, batch_size, shuffle=True):
        dataset = tf.data.Dataset.from_tensor_slices(
            (df["text"].values, df["label"].values)
        )
        if shuffle:
            dataset = dataset.shuffle(buffer_size=len(df))
        dataset = dataset.batch(batch_size)
        dataset = dataset.prefetch(tf.data.AUTOTUNE)
        return dataset

    train_dataset = create_dataset(train_df, args.batch_size, shuffle=True)
    val_dataset = create_dataset(val_df, args.batch_size, shuffle=False)
    test_dataset = create_dataset(test_df, args.batch_size, shuffle=False)

    logger.info("✅ Datasets created")

    # =========================================================================
    # STEP 5: Setup Callbacks
    # =========================================================================
    logger.info("\n📞 STEP 5: Setting up callbacks...")

    callbacks = []

    if args.early_stopping:
        early_stop = tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=args.patience,
            restore_best_weights=True,
            verbose=1,
        )
        callbacks.append(early_stop)
        logger.info("  ✓ Early stopping enabled")

    # TensorBoard
    log_dir = f"{args.output_path}/logs/{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    tensorboard = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
    callbacks.append(tensorboard)
    logger.info(f"  ✓ TensorBoard logs: {log_dir}")

    # Model checkpoint
    checkpoint_path = (
        f"{args.output_path}/checkpoints/model_{{epoch:02d}}_{{val_accuracy:.4f}}.h5"
    )
    checkpoint = tf.keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_path, monitor="val_accuracy", save_best_only=True, verbose=1
    )
    callbacks.append(checkpoint)
    logger.info("  ✓ Model checkpointing enabled")

    # =========================================================================
    # STEP 6: Train Model
    # =========================================================================
    logger.info("\n🚂 STEP 6: Training model...")
    logger.info(f"  Epochs: {args.epochs}")
    logger.info(f"  Batch size: {args.batch_size}")
    logger.info(f"  Learning rate: {args.learning_rate}")
    logger.info("=" * 80)

    start_time = datetime.now()

    history = model.fit(
        train_dataset,
        validation_data=val_dataset,
        epochs=args.epochs,
        class_weight=class_weight_dict,
        callbacks=callbacks,
        verbose=args.verbose,
    )

    end_time = datetime.now()
    training_time = (end_time - start_time).total_seconds()

    logger.info("=" * 80)
    logger.info(f"✅ Training complete!")
    logger.info(f"   Time: {training_time:.2f}s ({training_time/60:.2f} min)")
    logger.info(f"   Avg per epoch: {training_time/args.epochs:.2f}s")

    # =========================================================================
    # STEP 7: Evaluate Model
    # =========================================================================
    logger.info("\n📊 STEP 7: Evaluating model on test set...")

    test_results = model.evaluate(test_dataset, verbose=0)

    logger.info("Test Results:")
    for metric_name, value in zip(model.metrics_names, test_results):
        logger.info(f"  {metric_name}: {value:.4f}")

    # =========================================================================
    # STEP 8: Save Model
    # =========================================================================
    logger.info("\n💾 STEP 8: Saving model...")

    model_path = f"{args.output_path}/model"

    if args.save_format == "tf":
        # Use .keras format for Keras 3
        model.save(f"{model_path}.keras")
        logger.info(f"✅ Model saved: {model_path}.keras")
    else:
        model.save(f"{model_path}.h5")
        logger.info(f"✅ Model saved: {model_path}.h5")

    logger.info(f"✅ Model saved: {model_path}")

    # Save label encoder
    import joblib

    encoder_path = f"{args.output_path}/label_encoder.pkl"
    joblib.dump(label_encoder, encoder_path)
    logger.info(f"✅ Label encoder saved: {encoder_path}")

    # Save training metadata
    metadata = {
        "model_type": args.model_type,
        "num_classes": num_classes,
        "classes": label_encoder.classes_.tolist(),
        "training_samples": len(train_df),
        "validation_samples": len(val_df),
        "test_samples": len(test_df),
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "training_time_seconds": training_time,
        "test_accuracy": float(test_results[1]),  # Assuming accuracy is second metric
        "timestamp": datetime.now().isoformat(),
    }

    metadata_path = f"{args.output_path}/metadata.json"
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)

    logger.info(f"✅ Metadata saved: {metadata_path}")

    # =========================================================================
    # DONE
    # =========================================================================
    logger.info("\n" + "=" * 80)
    logger.info("🎉 TRAINING PIPELINE COMPLETE!")
    logger.info("=" * 80)
    logger.info(f"Model path: {model_path}")
    logger.info(f"Test accuracy: {test_results[1]:.4f}")
    logger.info("=" * 80)

    return 0


if __name__ == "__main__":
    sys.exit(main())
