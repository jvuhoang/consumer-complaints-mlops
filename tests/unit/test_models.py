"""Unit tests for model building"""

import pytest
import tensorflow as tf
from src.models.use_classifier import build_use_classifier


def test_model_builds():
    """Test that model builds without errors"""
    model = build_bert_classifier(num_classes=18)
    assert model is not None
    assert isinstance(model, tf.keras.Model)

def test_model_input_shape():
    """Test model accepts string input"""
    model = build_bert_classifier(num_classes=18)
    
    # Test prediction
    sample_text = ["This is a test complaint"]
    prediction = model.predict(sample_text)
    
    assert prediction.shape == (1, 18)

def test_model_output_probabilities():
    """Test model outputs valid probabilities"""
    model = build_bert_classifier(num_classes=18)
    
    sample_text = ["Test"]
    prediction = model.predict(sample_text)
    
    # Check probabilities sum to 1
    assert abs(prediction.sum() - 1.0) < 0.01
    
    # Check all probabilities between 0 and 1
    assert (prediction >= 0).all()
    assert (prediction <= 1).all()