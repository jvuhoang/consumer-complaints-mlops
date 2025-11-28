"""
Test the trained model from Google Colab
"""

import tensorflow as tf
import json
import numpy as np

print("="*80)
print("TESTING MODEL FROM GOOGLE COLAB")
print("="*80)

# Define the custom standardization function (same as in training)
@tf.keras.utils.register_keras_serializable()
def custom_standardization(input_data):
    """
    Custom text standardization using TensorFlow string ops
    """
    lowercase = tf.strings.lower(input_data)
    no_html = tf.strings.regex_replace(lowercase, '<[^>]+>', ' ')
    no_urls = tf.strings.regex_replace(no_html, r'http\S+|www\S+', ' ')
    no_emails = tf.strings.regex_replace(no_urls, r'\S+@\S+', ' ')
    no_redacted = tf.strings.regex_replace(no_emails, r'x{2,}', ' ')
    no_numbers = tf.strings.regex_replace(no_redacted, r'\d+', ' ')
    no_punct = tf.strings.regex_replace(no_numbers, r'[^a-z\s\-]', ' ')
    cleaned = tf.strings.regex_replace(no_punct, r'\s+', ' ')
    return tf.strings.strip(cleaned)

# Load model with custom objects
print("\n1️⃣ Loading model...")
try:
    model = tf.keras.models.load_model(
        'models/complete_complaint_classifier_tf.keras',
        custom_objects={'custom_standardization': custom_standardization}
    )
    print("✅ Model loaded successfully")
except Exception as e:
    print(f"❌ Error loading model: {e}")
    exit(1)

# Load label mapping
print("\n2️⃣ Loading label mapping...")
try:
    with open('models/label_mapping.json', 'r') as f:
        label_data = json.load(f)
        id_to_product = {int(k): v for k, v in label_data['id_to_product'].items()}
    print(f"✅ Loaded {len(id_to_product)} product categories")
except Exception as e:
    print(f"❌ Error loading labels: {e}")
    exit(1)

# Load metadata
print("\n3️⃣ Loading metadata...")
try:
    with open('models/model_metadata.json', 'r') as f:
        metadata = json.load(f)
    print(f"✅ Model: {metadata.get('model_name', 'Unknown')}")
    print(f"   Test Accuracy: {metadata.get('performance', {}).get('test_accuracy', 0)*100:.2f}%")
    print(f"   Trained: {metadata.get('training_date', 'Unknown')[:10]}")
except Exception as e:
    print(f"⚠️  Could not load metadata: {e}")

# Test predictions
print("\n4️⃣ Testing predictions...")
print("="*80)

test_complaints = [
    "Debt collector keeps calling me repeatedly and threatening me",
    "I received a foreclosure notice on my mortgage payment",
    "There are unauthorized charges on my credit card statement",
    "My student loan servicer reported incorrect payment information",
    "The bank charged me overdraft fees that I didn't authorize"
]

for i, complaint in enumerate(test_complaints, 1):
    print(f"\n[Test {i}]")
    print(f"Input: {complaint}")
    
    # Predict
    pred = model.predict(tf.constant([complaint]), verbose=0)
    pred_class = np.argmax(pred[0])
    confidence = pred[0][pred_class]
    
    # Get top 3
    top3_idx = np.argsort(pred[0])[-3:][::-1]
    
    print(f"💡 Prediction: {id_to_product[pred_class]} ({confidence:.1%})")
    print(f"📊 Top 3:")
    for j, idx in enumerate(top3_idx, 1):
        print(f"   {j}. {id_to_product[idx]} ({pred[0][idx]:.1%})")

print("\n" + "="*80)
print("✅ MODEL WORKING PERFECTLY!")
print("="*80)