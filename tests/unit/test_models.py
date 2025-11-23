"""Unit tests for model building"""

import pytest
import tensorflow as tf
import numpy as np
from src.models.use_classifier import build_use_classifier


def test_use_model_builds():
    """Test USE model builds without errors"""
    model = build_use_classifier(num_classes=18)
    assert model is not None
    assert isinstance(model, tf.keras.Model)


def test_use_model_prediction():
    """Test USE model can make predictions"""
    model = build_use_classifier(num_classes=18)
    sample_text = np.array(["This is a test complaint"]) 
    sample_text = tf.constant(["This is a test complaint"], dtype=tf.string)
    
    prediction = model.predict(sample_text, verbose=0)
    
    assert prediction.shape == (1, 18) 
    assert np.allclose(np.sum(prediction), 1.0, atol=1e-5)