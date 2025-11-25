import sys
import types
import tensorflow as tf
import re
import string

# Ensure the name 'keras' is available for deserialization (some SavedModels reference 'keras')
# If standalone keras is installed, keep it; otherwise make 'keras' point to tf.keras
try:
    import keras  # prefer real keras if present
except Exception:
    # Provide a module alias so deserialization that imports 'keras' works
    sys.modules.setdefault('keras', tf.keras)
    keras = tf.keras

# Resolve a register function in a version-tolerant way
try:
    register_keras_serializable = tf.keras.utils.register_keras_serializable
except AttributeError:
    # fallback to standalone keras saving if available
    try:
        from keras.saving import register_keras_serializable as register_keras_serializable
    except Exception:
        # last-resort no-op (function will not be registered in keras registry)
        def register_keras_serializable(*args, **kwargs):
            def _decorator(fn):
                return fn
            return _decorator

@register_keras_serializable(package="Custom")
def custom_standardization(input_data):
    """Custom text standardization function"""
    lowercase = tf.strings.lower(input_data)
    stripped_html = tf.strings.regex_replace(lowercase, '<br />', ' ')
    return tf.strings.regex_replace(stripped_html,
                                    f'[{re.escape(string.punctuation)}]', '')

def load_complaint_model(model_path):
    """Load model with custom objects"""
    # Use tf.keras.models (or the resolved keras.models)
    models_loader = getattr(tf.keras, "models", getattr(keras, "models"))
    return models_loader.load_model(
        model_path,
        custom_objects={'custom_standardization': custom_standardization}
    )