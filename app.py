"""
Web demo for Consumer Complaints Classifier - Merged Version
Uses web_demo.py interface with app.py credentials and configuration
"""

import json
import os
import time
import traceback

import numpy as np
from flask import Flask, jsonify, render_template_string, request
from google.cloud import aiplatform

app = Flask(__name__)

print("🔄 Initializing...")

# --- Configuration from app.py ---
PROJECT_ID = os.environ.get("GCP_PROJECT_ID", "project-37461c1b-635d-4cf2-af5")
LOCATION = os.environ.get("GCP_LOCATION", "us-central1")
ENDPOINT_ID = os.environ.get("VERTEX_ENDPOINT_ID", "4351135847504936960")

# Category mapping from app.py
CATEGORY_MAPPING = {
    0: "Student loan",
    1: "Personal loan",
    2: "Other",
    3: "Mortgage",
    4: "Money transfer",
    5: "Debt collection",
    6: "Credit reporting",
    7: "Credit card",
    8: "Consumer Loan",
    9: "Bank account or service"
}

# Initialize Vertex AI
try:
    aiplatform.init(project=PROJECT_ID, location=LOCATION)
    endpoint_resource_name = f"projects/{PROJECT_ID}/locations/{LOCATION}/endpoints/{ENDPOINT_ID}"
    endpoint = aiplatform.Endpoint(endpoint_resource_name)
    print(f"✅ Vertex AI initialized for project {PROJECT_ID}")
    print(f"✅ Endpoint: {endpoint_resource_name}")
except Exception as e:
    print(f"❌ Error initializing Vertex AI: {e}")
    raise

# Use the category mapping as idx_to_label
idx_to_label = CATEGORY_MAPPING

print("✅ Ready!")

# --- HTML from web_demo.py ---
HTML = """
<!DOCTYPE html>
<html>
<head>
    <title>Consumer Complaints Classifier</title>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body { font-family: Arial; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); height: 100vh; display: flex; align-items: center; justify-content: center; padding: 10px; overflow: hidden; }
        .container { max-width: 750px; width: 100%; background: white; padding: 15px 20px; border-radius: 12px; box-shadow: 0 20px 60px rgba(0,0,0,0.3); }
        h1 { color: #667eea; text-align: center; font-size: 20px; margin-bottom: 3px; }
        .subtitle { text-align: center; color: #666; font-size: 12px; margin-bottom: 10px; }
        .stats { display: grid; grid-template-columns: repeat(3, 1fr); gap: 8px; margin: 10px 0; padding: 10px; background: #f8f9fa; border-radius: 8px; }
        .stat-item { text-align: center; }
        .stat-value { font-size: 18px; font-weight: bold; color: #667eea; }
        .stat-label { font-size: 11px; color: #666; margin-top: 2px; }
        .examples { margin: 10px 0; padding: 10px; background: #e3f2fd; border-radius: 8px; }
        .examples h3 { font-size: 13px; margin-bottom: 6px; }
        .example-btn { display: inline-block; margin: 3px; padding: 5px 10px; background: white; border: 2px solid #1976d2; color: #1976d2; border-radius: 12px; cursor: pointer; font-size: 11px; }
        .example-btn:hover { background: #1976d2; color: white; }
        textarea { width: 100%; padding: 10px; font-size: 13px; border: 2px solid #ddd; border-radius: 6px; margin: 10px 0; resize: none; height: 60px; }
        textarea:focus { outline: none; border-color: #667eea; }
        button { background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; padding: 10px 25px; font-size: 15px; font-weight: bold; border: none; border-radius: 6px; cursor: pointer; width: 100%; }
        button:hover { transform: translateY(-1px); }
        button:disabled { opacity: 0.6; cursor: not-allowed; transform: none; }
        .loading { display: none; text-align: center; padding: 12px; }
        .spinner { border: 3px solid #f3f3f3; border-top: 3px solid #667eea; border-radius: 50%; width: 30px; height: 30px; animation: spin 1s linear infinite; margin: 0 auto 8px; }
        @keyframes spin { 0% { transform: rotate(0deg); } 100% { transform: rotate(360deg); } }
        .result { margin-top: 15px; padding: 15px; background: linear-gradient(135deg, #e8f5e9 0%, #c8e6c9 100%); border-left: 4px solid #4caf50; border-radius: 6px; display: none; }
        .result h3 { font-size: 14px; margin-bottom: 8px; }
        .prediction { font-size: 18px; font-weight: bold; color: #1b5e20; margin: 8px 0; padding: 12px; background: white; border-radius: 6px; text-align: center; }
        .metrics { display: grid; grid-template-columns: repeat(2, 1fr); gap: 8px; margin: 10px 0; }
        .metric-box { padding: 8px; background: white; border-radius: 5px; text-align: center; }
        .metric-box div:first-child { font-size: 11px; color: #666; }
        .metric-value { font-size: 16px; font-weight: bold; color: #4caf50; }
        .pred-item { padding: 8px; margin: 6px 0; background: white; border-radius: 5px; }
        .pred-header { display: flex; justify-content: space-between; margin-bottom: 6px; font-size: 12px; }
        .pred-bar { height: 6px; background: linear-gradient(to right, #4caf50, #8bc34a); border-radius: 3px; }
        #top3 h4 { font-size: 13px; margin-bottom: 8px; }
    </style>
</head>
<body>
    <div class="container">
        <h1>🤖 Consumer Complaints Classifier</h1>
        <p style="text-align: center; color: #666;">AI-powered classification for financial complaints</p>

        <div class="stats">
            <div class="stat-item"><div class="stat-value">10</div><div class="stat-label">Categories</div></div>
            <div class="stat-item"><div class="stat-value">AI</div><div class="stat-label">Powered</div></div>
            <div class="stat-item"><div class="stat-value">Fast</div><div class="stat-label">Response</div></div>
        </div>

        <div class="examples">
            <h3>💡 Try These Examples:</h3>
            <span class="example-btn" onclick="fillExample('Debt collector keeps calling me repeatedly')">Debt Collection</span>
            <span class="example-btn" onclick="fillExample('My mortgage company charged excessive fees')">Mortgage</span>
            <span class="example-btn" onclick="fillExample('Credit card raised my interest rate')">Credit Card</span>
            <span class="example-btn" onclick="fillExample('Bank froze my account without reason')">Bank Account</span>
        </div>

        <textarea id="complaint" rows="6" placeholder="Enter a consumer complaint here..."></textarea>

        <button id="btn" onclick="classify()">🚀 Classify Complaint</button>

        <div class="loading" id="loading">
            <div class="spinner"></div>
            <p>⏳ Analyzing...</p>
        </div>

        <div class="result" id="result">
            <h3>📊 Classification Results</h3>
            <div class="prediction" id="prediction"></div>
            <div class="metrics">
                <div class="metric-box"><div style="font-size: 14px; color: #666;">Confidence</div><div class="metric-value" id="confidence"></div></div>
                <div class="metric-box"><div style="font-size: 14px; color: #666;">Response Time</div><div class="metric-value" id="latency"></div></div>
            </div>
            <div id="top3"></div>
        </div>
    </div>

    <script>
        function fillExample(text) {
            document.getElementById('complaint').value = text;
        }

        async function classify() {
            const text = document.getElementById('complaint').value.trim();

            if (!text) {
                alert('⚠️ Please enter some text');
                return;
            }

            document.getElementById('btn').disabled = true;
            document.getElementById('loading').style.display = 'block';
            document.getElementById('result').style.display = 'none';

            try {
                const res = await fetch('/predict', {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify({text: text})
                });

                if (!res.ok) {
                    throw new Error('Prediction failed with status: ' + res.status);
                }

                const data = await res.json();

                if (data.error) {
                    throw new Error(data.error);
                }

                document.getElementById('prediction').innerHTML = '🎯 ' + data.predicted_label;
                document.getElementById('confidence').textContent = (data.confidence * 100).toFixed(1) + '%';
                document.getElementById('latency').textContent = Math.round(data.latency_ms) + 'ms';

                let html = '<h4>Top 3 Predictions:</h4>';
                data.top_3.forEach((p, i) => {
                    const pct = (p.confidence * 100).toFixed(1);
                    const width = Math.round(p.confidence * 100);
                    html += `<div class="pred-item">
                        <div class="pred-header">
                            <span><strong>${i+1}. ${p.label}</strong></span>
                            <span style="color: #4caf50; font-weight: bold;">${pct}%</span>
                        </div>
                        <div class="pred-bar" style="width: ${width}%;"></div>
                    </div>`;
                });

                document.getElementById('top3').innerHTML = html;
                document.getElementById('result').style.display = 'block';

            } catch (error) {
                alert('❌ Error: ' + error.message);
                console.error('Prediction error:', error);
            } finally {
                document.getElementById('loading').style.display = 'none';
                document.getElementById('btn').disabled = false;
            }
        }

        // Allow Enter key to submit
        document.getElementById('complaint').addEventListener('keydown', function(e) {
            if (e.key === 'Enter' && e.ctrlKey) {
                classify();
            }
        });
    </script>
</body>
</html>
"""


@app.route("/")
def home():
    return render_template_string(HTML)


@app.route("/predict", methods=["POST"])
def predict():
    """Prediction endpoint using app.py logic"""
    try:
        # Get request data
        data = request.get_json()

        if not data or "text" not in data:
            return jsonify({"error": "No text provided"}), 400

        text = data["text"].strip()

        if not text:
            return jsonify({"error": "Text is empty"}), 400

        print(f"\n🔍 Received: {text[:50]}...")

        # Make prediction using app.py format
        start = time.time()
        
        # Use double-nested format for the input (from app.py)
        response = endpoint.predict(instances=[[text]])
        latency = (time.time() - start) * 1000

        # Extract results
        predictions = response.predictions
        pred = predictions[0]
        
        # Handle different prediction formats
        if isinstance(pred, list):
            probs = pred
            pred_idx = np.argmax(probs)
            confidence = float(max(probs))
        else:
            pred_idx = int(pred)
            probs = [0] * len(idx_to_label)
            probs[pred_idx] = 1.0
            confidence = 1.0

        pred_label = idx_to_label.get(pred_idx, "Unknown")

        # Get top 3
        top_3_idx = np.argsort(probs)[-3:][::-1]
        top_3 = [
            {"label": idx_to_label.get(i, "Unknown"), "confidence": float(probs[i])}
            for i in top_3_idx
        ]

        print(f"✅ Predicted: {pred_label} ({confidence*100:.1f}%) in {latency:.0f}ms")

        return jsonify(
            {
                "predicted_label": pred_label,
                "confidence": confidence,
                "latency_ms": round(latency, 2),
                "top_3": top_3,
            }
        )

    except Exception as e:
        error_msg = str(e)
        print(f"\n❌ Error: {error_msg}")
        print(traceback.format_exc())

        return jsonify({"error": f"Prediction failed: {error_msg}"}), 500


@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint from app.py"""
    return jsonify({
        "status": "healthy",
        "service": "consumer-complaints-classifier",
        "project": PROJECT_ID,
        "endpoint": ENDPOINT_ID
    }), 200


@app.route('/categories', methods=['GET'])
def get_categories():
    """Returns available categories from app.py"""
    return jsonify({
        "categories": CATEGORY_MAPPING,
        "total_categories": len(CATEGORY_MAPPING)
    })


if __name__ == "__main__":
    print("\n" + "=" * 80)
    print("🌐 WEB DEMO STARTING")
    print("=" * 80)
    print(f"\n✅ Project: {PROJECT_ID}")
    print(f"✅ Location: {LOCATION}")
    print(f"✅ Endpoint ID: {ENDPOINT_ID}")
    print(f"✅ Categories: {len(idx_to_label)}")
    print("\n👉 Access at: http://localhost:8080")
    print("👉 In Cloud Shell: Click 'Web Preview' → 'Port 8080'")
    print("\nℹ️  Press Ctrl+C to stop")
    print("=" * 80 + "\n")

    port = int(os.environ.get('PORT', 8080))
    debug_mode = os.environ.get('FLASK_DEBUG', 'False').lower() == 'true'
    app.run(host="0.0.0.0", port=port, debug=debug_mode)