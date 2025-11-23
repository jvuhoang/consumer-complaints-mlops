# src/models/bert_classifier.py
import tensorflow as tf
import tensorflow_hub as hub


def build_bert_classifier(num_classes: int):
    """Build BERT classifier - extracted from your Colab notebook"""
    
    preprocessor_url = "https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3"
    encoder_url = "https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-4_H-512_A-8/2"
    
    preprocessor = hub.KerasLayer(preprocessor_url, name='bert_preprocessor')
    encoder = hub.KerasLayer(encoder_url, trainable=True, name='bert_encoder')
    
    text_input = tf.keras.layers.Input(shape=(), dtype=tf.string, name='text')
    encoder_inputs = preprocessor(text_input)
    outputs = encoder(encoder_inputs)
    pooled_output = outputs['pooled_output']
    
    x = tf.keras.layers.Dropout(0.1)(pooled_output)
    x = tf.keras.layers.Dense(256, activation='relu')(x)
    x = tf.keras.layers.Dropout(0.2)(x)
    x = tf.keras.layers.Dense(128, activation='relu')(x)
    x = tf.keras.layers.Dropout(0.2)(x)
    
    output = tf.keras.layers.Dense(num_classes, activation='softmax')(x)
    
    return tf.keras.Model(inputs=text_input, outputs=output)