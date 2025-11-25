from flask import Flask, request, jsonify, render_template_string
from google.cloud import aiplatform
import os
import json

app = Flask(__name__)

# --- Vertex AI Configuration ---
PROJECT_ID = os.environ.get("GCP_PROJECT_ID", "project-37461c1b-635d-4cf2-af5")
ENDPOINT_ID = os.environ.get("VERTEX_ENDPOINT_ID", "4351135847504936960")
LOCATION = os.environ.get("GCP_LOCATION", "us-central1")

def get_latest_endpoint():
    """Get the most recently created endpoint"""
    endpoints = aiplatform.Endpoint.list(
        filter='display_name="consumer-complaints-classifier"',  # Optional filter
        order_by="create_time desc"
    )
    if endpoints:
        return endpoints[0].name.split('/')[-1]
    return None

# Update in your code
ENDPOINT_ID = os.environ.get("VERTEX_ENDPOINT_ID") or get_latest_endpoint()


# Category mapping
CATEGORY_MAPPING = {
    0: "Bank account or service",
    1: "Checking or savings account",
    2: "Student loan",
    3: "Credit card or prepaid card",
    4: "Consumer Loan",
    5: "Credit reporting",
    6: "Credit reporting, credit repair services, or other personal consumer reports",
    7: "Mortgage",
    8: "Money transfer, virtual currency, or money service",
    9: "Money transfers",
    10: "Debt collection",
    11: "Other financial service",
    12: "Payday loan",
    13: "Payday loan, title loan, or personal loan",
    14: "Credit card",
    15: "Vehicle loan or lease",
    16: "Prepaid card",
    17: "Virtual currency"
}

# Initialize Vertex AI
try:
    aiplatform.init(project=PROJECT_ID, location=LOCATION)
    print(f"Vertex AI initialized for project {PROJECT_ID} in {LOCATION}")
except Exception as e:
    print(f"Error initializing Vertex AI client: {e}")

# HTML Template
HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Banking Support Chatbot</title>
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }
        
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
            display: flex;
            justify-content: center;
            align-items: center;
            padding: 20px;
        }
        
        .container {
            background: white;
            border-radius: 20px;
            box-shadow: 0 20px 60px rgba(0,0,0,0.3);
            max-width: 600px;
            width: 100%;
            padding: 40px;
        }
        
        .logo-container {
            text-align: center;
            margin-bottom: 30px;
        }
        
        .logo {
            width: 80px;
            height: 80px;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            border-radius: 50%;
            display: inline-flex;
            align-items: center;
            justify-content: center;
            font-size: 40px;
            color: white;
            margin-bottom: 15px;
        }
        
        h1 {
            color: #333;
            font-size: 28px;
            margin-bottom: 10px;
            text-align: center;
        }
        
        .subtitle {
            color: #666;
            text-align: center;
            font-size: 14px;
            margin-bottom: 30px;
        }
        
        .question {
            color: #444;
            font-size: 18px;
            margin-bottom: 20px;
            font-weight: 500;
        }
        
        textarea {
            width: 100%;
            padding: 15px;
            border: 2px solid #e0e0e0;
            border-radius: 10px;
            font-size: 16px;
            font-family: inherit;
            resize: vertical;
            min-height: 120px;
            transition: border-color 0.3s;
        }
        
        textarea:focus {
            outline: none;
            border-color: #667eea;
        }
        
        .button {
            width: 100%;
            padding: 15px;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            border: none;
            border-radius: 10px;
            font-size: 18px;
            font-weight: 600;
            cursor: pointer;
            margin-top: 20px;
            transition: transform 0.2s, box-shadow 0.2s;
        }
        
        .button:hover {
            transform: translateY(-2px);
            box-shadow: 0 10px 20px rgba(102, 126, 234, 0.4);
        }
        
        .button:active {
            transform: translateY(0);
        }
        
        .button:disabled {
            opacity: 0.6;
            cursor: not-allowed;
        }
        
        .result {
            margin-top: 30px;
            padding: 20px;
            background: #f8f9fa;
            border-radius: 10px;
            border-left: 4px solid #667eea;
            display: none;
        }
        
        .result.show {
            display: block;
            animation: slideIn 0.3s ease;
        }
        
        @keyframes slideIn {
            from {
                opacity: 0;
                transform: translateY(-10px);
            }
            to {
                opacity: 1;
                transform: translateY(0);
            }
        }
        
        .result-title {
            font-weight: 600;
            color: #333;
            margin-bottom: 10px;
            font-size: 18px;
        }
        
        .department {
            font-size: 24px;
            color: #667eea;
            font-weight: 700;
            margin: 10px 0;
        }
        
        .confidence {
            color: #666;
            font-size: 14px;
            margin-top: 10px;
        }
        
        .error {
            background: #fee;
            border-left-color: #f44336;
            color: #c00;
        }
        
        .loading {
            text-align: center;
            color: #667eea;
            margin-top: 20px;
            display: none;
        }
        
        .loading.show {
            display: block;
        }
        
        .spinner {
            border: 3px solid #f3f3f3;
            border-top: 3px solid #667eea;
            border-radius: 50%;
            width: 40px;
            height: 40px;
            animation: spin 1s linear infinite;
            margin: 20px auto;
        }
        
        @keyframes spin {
            0% { transform: rotate(0deg); }
            100% { transform: rotate(360deg); }
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="logo-container">
            <div class="logo">🏦</div>
            <h1>Chatbot for Banking Questions</h1>
            <p class="subtitle">We're here to help you with your banking concerns</p>
        </div>
        
        <p class="question">Is there a problem with your banking that we could help with?</p>
        
        <form id="complaintForm">
            <textarea 
                id="complaintInput" 
                placeholder="Describe your banking issue here... (e.g., 'I have unauthorized charges on my credit card')"
                required
            ></textarea>
            
            <button type="submit" class="button" id="submitBtn">
                Connect to Department
            </button>
        </form>
        
        <div class="loading" id="loading">
            <div class="spinner"></div>
            <p>Analyzing your request...</p>
        </div>
        
        <div class="result" id="result"></div>
    </div>

    <script>
        const form = document.getElementById('complaintForm');
        const input = document.getElementById('complaintInput');
        const submitBtn = document.getElementById('submitBtn');
        const loading = document.getElementById('loading');
        const result = document.getElementById('result');

        form.addEventListener('submit', async (e) => {
            e.preventDefault();
            
            const complaint = input.value.trim();
            if (!complaint) return;
            
            // Show loading
            submitBtn.disabled = true;
            loading.classList.add('show');
            result.classList.remove('show');
            
            try {
                const response = await fetch('/api/predict', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json',
                    },
                    body: JSON.stringify({ complaint: complaint })
                });
                
                const data = await response.json();
                
                loading.classList.remove('show');
                
                if (response.ok && data.status === 'success') {
                    const pred = data.prediction;
                    result.innerHTML = `
                        <div class="result-title">We've identified your concern:</div>
                        <div class="department">${pred.predicted_category}</div>
                        <p>You will be connected to the appropriate department to help resolve your issue.</p>
                        ${pred.confidence ? `<div class="confidence">Confidence: ${(pred.confidence * 100).toFixed(1)}%</div>` : ''}
                    `;
                    result.classList.remove('error');
                } else {
                    result.innerHTML = `
                        <div class="result-title">Error</div>
                        <p>${data.error || 'Unable to process your request. Please try again.'}</p>
                    `;
                    result.classList.add('error');
                }
                
                result.classList.add('show');
                
            } catch (error) {
                loading.classList.remove('show');
                result.innerHTML = `
                    <div class="result-title">Connection Error</div>
                    <p>Unable to connect to the server. Please check your connection and try again.</p>
                `;
                result.classList.add('error');
                result.classList.add('show');
            } finally {
                submitBtn.disabled = false;
            }
        });
    </script>
</body>
</html>
"""

@app.route('/', methods=['GET'])
def home():
    """Render the chatbot interface"""
    return render_template_string(HTML_TEMPLATE)

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        "status": "healthy",
        "service": "consumer-complaints-classifier",
        "project": PROJECT_ID,
        "endpoint": ENDPOINT_ID
    }), 200

@app.route('/api/predict', methods=['POST'])
def predict_complaint():
    """API endpoint for predictions"""
    try:
        data = request.json
        
        if not data or 'complaint' not in data:
            return jsonify({"error": "Missing 'complaint' in request body"}), 400
        
        complaint_text = data['complaint']
        if not isinstance(complaint_text, str) or not complaint_text.strip():
            return jsonify({"error": "Complaint text must be a non-empty string"}), 400
        
        # Call Vertex AI Endpoint
        endpoint = aiplatform.Endpoint(
            endpoint_name=f"projects/{PROJECT_ID}/locations/{LOCATION}/endpoints/{ENDPOINT_ID}"
        )
        
        #response = endpoint.predict(instances=[{"text": complaint_text}])
        # Ensure it's a plain string
        if isinstance(complaint_text, list):
            complaint_text = complaint_text[0]
        response = endpoint.predict(instances=[complaint_text])
        predictions = response.predictions
        
        # Process prediction
        pred = predictions[0]
        if isinstance(pred, list):
            predicted_class = pred.index(max(pred))
            confidence = max(pred)
        else:
            predicted_class = int(pred)
            confidence = None
        
        result = {
            "complaint": complaint_text,
            "predicted_category": CATEGORY_MAPPING.get(predicted_class, "Unknown"),
            "predicted_class": predicted_class
        }
        
        if confidence is not None:
            result["confidence"] = float(confidence)
        
        return jsonify({
            "status": "success",
            "prediction": result
        })
    
    except Exception as e:
        print(f"Prediction error: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

@app.route('/categories', methods=['GET'])
def get_categories():
    """Returns available categories"""
    return jsonify({
        "categories": CATEGORY_MAPPING,
        "total_categories": len(CATEGORY_MAPPING)
    })

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 8080))
    debug_mode = os.environ.get('FLASK_DEBUG', 'False').lower() == 'true'
    app.run(host='0.0.0.0', port=port, debug=debug_mode)