"""Universal Sentence Encoder classifier"""

import tensorflow as tf
import tensorflow_hub as hub
from tensorflow.keras import Model, layers


class USEEmbeddingLayer(layers.Layer):
    def __init__(self, trainable=False, **kwargs):
        super(USEEmbeddingLayer, self).__init__(**kwargs)
        self.encoder_url = "https://tfhub.dev/google/universal-sentence-encoder/4"
        self.encoder = hub.KerasLayer(
            self.encoder_url,
            trainable=trainable,
            name='universal_sentence_encoder_hub_layer'
        )

    def call(self, inputs):
        return self.encoder(inputs)

def build_use_classifier(num_classes):
    """
    Build a text classifier using Universal Sentence Encoder
    """
    # Universal Sentence Encoder from TensorFlow Hub via custom layer
    text_input = layers.Input(shape=(), dtype=tf.string, name='text')
    embedding = USEEmbeddingLayer(trainable=False)(text_input)

    # Dense layers
    x = layers.Dense(256, activation='relu')(embedding)
    x = layers.Dropout(0.3)(x)
    x = layers.Dense(128, activation='relu')(x)
    x = layers.Dropout(0.2)(x)

    # Output layer
    if num_classes == 2:
        output = layers.Dense(1, activation='sigmoid', name='output')(x)
    else:
        output = layers.Dense(num_classes, activation='softmax', name='output')(x)

    model = Model(inputs=text_input, outputs=output, name='USE_Classifier')

    return model
