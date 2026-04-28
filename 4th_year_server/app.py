"""
Cancer Detection Server Application
Flask-based server for breast cancer detection using CNN model
"""

from flask import Flask, render_template, request, jsonify, send_from_directory   # Flask is to create server side application  web framework written in Python. to creating the web server and web applications.
import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model  #for trained the images using cnn models and its a frame work (in ml) developed by google , building and training neural networks.
from werkzeug.utils import secure_filename     # for using develop a simple server can be run in a single command like https://flask.palletsprojects.com/en/2.0.x/quickstart/
import tempfile     # to create temporary files and directories. user can share images and images will temporarily stored.
import traceback    #  means error history to debug the errors.
from pathlib import Path        # tells Python where to find the file you want to import.
from datetime import datetime

# Initialize Flask app
app = Flask(__name__, template_folder='templates', static_folder='static')

def log_info(msg):
    print(f"[SERVER] {msg}")

# Configuration
UPLOAD_FOLDER = tempfile.gettempdir()
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'bmp'}
MAX_FILE_SIZE = 50 * 1024 * 1024  # 50MB

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = MAX_FILE_SIZE

# Class mapping
CLASS_NAMES = {0: "Benign", 1: "Malignant", 2: "Normal"}

# Global model variable
cnn_model = None

def load_trained_model():
    """Load the trained CNN model from training pipeline"""
    global cnn_model
    try:
        model_path = 'model/cnn_model.keras'
        if os.path.exists(model_path):
            cnn_model = load_model(model_path)
            log_info("✓ Model loaded successfully from training pipeline!")
            return True
        else:
            log_info(f"⚠ Model not found at {model_path}")
            log_info("→ Please run: python train_model.py")
            log_info("→ Then run: python show_results.py")
            return False
    except Exception as e:
        log_info(f"✗ Error loading model: {str(e)}")
        return False

def allowed_file(filename):
    """Check if file extension is allowed"""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def prepare_image(image_path, target_size=(32, 32)):
    """Prepare image for model prediction"""
    try:
        img = cv2.imread(image_path)
        if img is None:
            return None
        
        # Resize and normalize
        img_resized = cv2.resize(img, target_size)
        img_normalized = img_resized.astype('float32') / 255.0
        img_batch = np.expand_dims(img_normalized, axis=0)
        
        return img_batch, img
    except Exception as e:
        print(f"Error preparing image: {str(e)}")
        return None

def predict_image_class(image_batch):
    """Make prediction on image"""
    try:
        if cnn_model is None:
            return None, None, None
        
        predictions = cnn_model.predict(image_batch, verbose=0)
        predicted_class = np.argmax(predictions[0])
        confidence = np.max(predictions[0]) * 100
        probabilities = predictions[0]
        
        return predicted_class, confidence, probabilities
    except Exception as e:
        print(f"Error in prediction: {str(e)}")
        traceback.print_exc()
        return None, None, None

@app.route('/')
def index():
    """Render home page"""
    return render_template('index.html')

@app.route('/api/health', methods=['GET'])
def health():
    """Health check endpoint"""
    model_status = "loaded" if cnn_model is not None else "not_loaded"
    return jsonify({
        'status': 'healthy',
        'model': model_status,
        'message': 'Cancer Detection Server is running'
    }), 200

@app.route('/api/predict', methods=['POST'])
def predict():
    """Prediction endpoint"""
    try:
        # Check if model is loaded
        if cnn_model is None:
            return jsonify({
                'success': False,
                'error': 'Model not loaded. Please ensure the model file exists at model/cnn_model.keras'
            }), 500
        
        # Check if image is in request
        if 'image' not in request.files:
            return jsonify({
                'success': False,
                'error': 'No image provided in request'
            }), 400
        
        file = request.files['image']
        
        if file.filename == '':
            return jsonify({
                'success': False,
                'error': 'No file selected'
            }), 400
        
        if not allowed_file(file.filename):
            return jsonify({
                'success': False,
                'error': f'File type not allowed. Allowed types: {", ".join(ALLOWED_EXTENSIONS)}'
            }), 400
        
        # Save temporary file
        filename = secure_filename(file.filename)
        temp_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(temp_path)
        
        try:
            # Prepare and predict
            prepared = prepare_image(temp_path)
            if prepared is None:
                return jsonify({
                    'success': False,
                    'error': 'Could not read image file'
                }), 400
            
            image_batch, original_img = prepared
            predicted_class, confidence, probabilities = predict_image_class(image_batch)
            
            if predicted_class is None:
                return jsonify({
                    'success': False,
                    'error': 'Prediction failed'
                }), 500
            
            predicted_label = CLASS_NAMES.get(int(predicted_class), 'Unknown')
            
            # Format response
            response = {
                'success': True,
                'filename': filename,
                'prediction': predicted_label,
                'confidence': float(confidence),
                'probabilities': {
                    'benign': float(probabilities[0] * 100),
                    'malignant': float(probabilities[1] * 100),
                    'normal': float(probabilities[2] * 100)
                }
            }
            
            return jsonify(response), 200
        
        finally:
            # Clean up temporary file
            try:
                if os.path.exists(temp_path):
                    os.remove(temp_path)
            except:
                pass
    
    except Exception as e:
        print(f"Error in prediction endpoint: {str(e)}")
        traceback.print_exc()
        return jsonify({
            'success': False,
            'error': f'Server error: {str(e)}'
        }), 500

@app.route('/api/info', methods=['GET'])
def info():
    """Get server information"""
    return jsonify({
        'name': 'Cancer Detection API',
        'version': '1.0.0',
        'description': 'Breast cancer detection using CNN model',
        'classes': CLASS_NAMES,
        'model_status': 'loaded' if cnn_model is not None else 'not_loaded'
    }), 200

@app.errorhandler(404)
def not_found(error):
    """Handle 404 errors"""
    return jsonify({
        'success': False,
        'error': 'Endpoint not found'
    }), 404

@app.errorhandler(500)
def server_error(error):
    """Handle 500 errors"""
    return jsonify({
        'success': False,
        'error': 'Internal server error'
    }), 500

if __name__ == '__main__':
    print("="*70)
    print("CANCER DETECTION SERVER")
    print("="*70)
    
    # Load model on startup
    print("\nLoading trained model...")
    if load_trained_model():
        print("\n✓ Server is ready!")
        print("\nAvailable endpoints:")
        print("  GET  /                 - Web interface")
        print("  GET  /api/health       - Health check")
        print("  GET  /api/info         - Server information")
        print("  POST /api/predict      - Image prediction")
        print("\n" + "="*70)
        print(f"Server running on: http://localhost:5000")
        print("="*70 + "\n")
        
        # Run the app
        app.run(debug=True, host='127.0.0.1', port=5000)
    else:
        print("\n✗ Failed to load model. Ensure the model file exists at model/cnn_model.keras")
        print("  Please train the model first using the Jupyter notebook.")
