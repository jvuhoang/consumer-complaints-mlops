import keras
import tensorflow as tf
import re
import string

@keras.saving.register_keras_serializable()
def custom_standardization(input_data):
    """Custom text standardization function"""
    lowercase = tf.strings.lower(input_data)
    stripped_html = tf.strings.regex_replace(lowercase, '<br />', ' ')
    return tf.strings.regex_replace(stripped_html,
                                    f'[{re.escape(string.punctuation)}]', '')

def load_complaint_model(model_path):
    """Load model with custom objects"""
    return keras.models.load_model(
        model_path,
        custom_objects={'custom_standardization': custom_standardization}
    )