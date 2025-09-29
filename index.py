# Flask Plant Disease Detection Web Application using XGBoost
# Complete CactusNet-XGB implementation with high accuracy - FIXED VERSION

from flask import Flask, render_template, request, jsonify, send_from_directory
import os
import cv2
import numpy as np
from PIL import Image
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder, StandardScaler
import joblib
import json
from datetime import datetime
import base64
from io import BytesIO
import pandas as pd
from werkzeug.utils import secure_filename
import seaborn as sns
import matplotlib.pyplot as plt
from skimage.feature import local_binary_pattern, graycomatrix, graycoprops
from skimage import filters, measure, segmentation
from skimage.color import rgb2gray
import warnings
warnings.filterwarnings('ignore')

# Initialize Flask app
app = Flask(__name__)
app.config['SECRET_KEY'] = 'cactusnet-xgb-secret-key'
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# Create necessary directories
for directory in ['uploads', 'models', 'static/images', 'templates', 'dataset']:
    os.makedirs(directory, exist_ok=True)

# Disease information database (based on your document)
DISEASE_INFO = {
    'healthy': {
        'name': 'Healthy',
        'severity': 'none',
        'color': '#10B981',
        'description': 'Your cactus appears healthy with no signs of disease.',
        'recommendations': [
            'Continue current care routine',
            'Monitor regularly for any changes',
            'Maintain proper drainage',
            'Ensure adequate but not excessive sunlight'
        ]
    },
    'anthracnose': {
        'name': 'Anthracnose',
        'severity': 'moderate',
        'color': '#F59E0B',
        'description': 'Fungal infection causing dark, sunken lesions on cactus surfaces.',
        'recommendations': [
            'Isolate affected plant immediately',
            'Apply copper-based fungicide',
            'Improve air circulation around plant',
            'Reduce watering frequency',
            'Remove affected parts with sterile tools'
        ]
    },
    'leaf_spot': {
        'name': 'Leaf Spot',
        'severity': 'moderate', 
        'color': '#F59E0B',
        'description': 'Bacterial or fungal infection causing circular spots on cactus pads.',
        'recommendations': [
            'Remove infected pads/segments',
            'Apply bactericide or fungicide',
            'Increase ventilation',
            'Water at soil level only'
        ]
    },
    'rust': {
        'name': 'Rust Disease',
        'severity': 'moderate',
        'color': '#F59E0B', 
        'description': 'Fungal disease causing orange or reddish-brown pustules.',
        'recommendations': [
            'Remove affected areas immediately',
            'Apply systemic fungicide',
            'Improve air circulation',
            'Avoid overhead watering'
        ]
    },
    'rot': {
        'name': 'Root/Stem Rot',
        'severity': 'high',
        'color': '#EF4444',
        'description': 'Serious fungal/bacterial condition causing tissue decay.',
        'recommendations': [
            'Stop watering immediately',
            'Cut away all affected tissue',
            'Apply fungicide to cuts',
            'Repot in fresh, sterile soil',
            'Quarantine the plant'
        ]
    },
    'cochineal': {
        'name': 'Cochineal Scale',
        'severity': 'high',
        'color': '#EF4444',
        'description': 'Parasitic insects creating white, cotton-like masses on cactus.',
        'recommendations': [
            'Remove insects with alcohol swab',
            'Apply systemic insecticide',
            'Quarantine affected plant',
            'Monitor nearby plants for spread'
        ]
    }
}

class CactusNetXGB:
    """
    CactusNet-XGB: Feature-driven framework for interpretable plant disease detection
    Implementation based on your research document
    """
    
    def __init__(self):
        self.xgb_model = None
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.class_names = list(DISEASE_INFO.keys())
        self.feature_names = []
        self.model_trained = False
        
    def extract_color_features(self, image):
        """Extract comprehensive color features from image"""
        # Convert to different color spaces
        hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
        
        features = []
        
        # RGB statistics
        for i in range(3):
            channel = image[:,:,i]
            features.extend([
                np.mean(channel),
                np.std(channel),
                np.median(channel),
                np.percentile(channel, 25),
                np.percentile(channel, 75)
            ])
        
        # HSV statistics
        for i in range(3):
            channel = hsv[:,:,i]
            features.extend([
                np.mean(channel),
                np.std(channel)
            ])
            
        # LAB statistics
        for i in range(3):
            channel = lab[:,:,i]
            features.extend([
                np.mean(channel),
                np.std(channel)
            ])
            
        return features
    
    def extract_texture_features(self, image):
        """Extract texture features using LBP and GLCM"""
        gray = rgb2gray(image)
        features = []
        
        # Local Binary Pattern
        radius = 3
        n_points = 8 * radius
        lbp = local_binary_pattern(gray, n_points, radius, method='uniform')
        lbp_hist, _ = np.histogram(lbp.ravel(), bins=n_points + 2, range=(0, n_points + 2))
        lbp_hist = lbp_hist.astype(float)
        lbp_hist /= (lbp_hist.sum() + 1e-7)  # Normalize
        features.extend(lbp_hist[:10])  # Take first 10 bins
        
        # GLCM features
        gray_int = (gray * 255).astype(np.uint8)
        glcm = graycomatrix(gray_int, [5], [0], 256, symmetric=True, normed=True)
        
        features.extend([
            graycoprops(glcm, 'dissimilarity')[0, 0],
            graycoprops(glcm, 'correlation')[0, 0],
            graycoprops(glcm, 'homogeneity')[0, 0],
            graycoprops(glcm, 'energy')[0, 0],
            graycoprops(glcm, 'contrast')[0, 0],
            graycoprops(glcm, 'ASM')[0, 0]
        ])
        
        return features
    
    def extract_shape_features(self, image):
        """Extract morphological and shape features"""
        gray = rgb2gray(image)
        
        # Threshold to get binary image
        thresh = filters.threshold_otsu(gray)
        binary = gray > thresh
        
        # Get region properties
        labeled = measure.label(binary)
        regions = measure.regionprops(labeled)
        
        features = []
        
        if regions:
            region = max(regions, key=lambda r: r.area)  # Get largest region
            features.extend([
                region.area,
                region.perimeter,
                region.eccentricity,
                region.solidity,
                region.extent,
                region.filled_area / (region.area + 1e-7),
                region.major_axis_length,
                region.minor_axis_length,
                region.orientation
            ])
        else:
            features.extend([0] * 9)
            
        # Edge density
        edges = filters.sobel(gray)
        edge_density = np.sum(edges > 0.1) / edges.size
        features.append(edge_density)
        
        return features
    
    def extract_statistical_features(self, image):
        """Extract statistical features from image"""
        gray = rgb2gray(image)
        features = []
        
        # Basic statistics
        features.extend([
            np.mean(gray),
            np.std(gray),
            np.var(gray),
            np.median(gray),
            np.min(gray),
            np.max(gray),
            np.percentile(gray, 25),
            np.percentile(gray, 75),
            np.percentile(gray, 90),
            np.percentile(gray, 10)
        ])
        
        # Moments
        try:
            from scipy import stats
            features.extend([
                stats.skew(gray.flatten()),
                stats.kurtosis(gray.flatten())
            ])
        except ImportError:
            # Fallback if scipy not available
            features.extend([0.0, 0.0])
        
        return features
    
    def extract_features(self, image):
        """Extract all features from image"""
        # Resize image to standard size
        image = cv2.resize(image, (224, 224))
        
        features = []
        features.extend(self.extract_color_features(image))
        features.extend(self.extract_texture_features(image))
        features.extend(self.extract_shape_features(image))
        features.extend(self.extract_statistical_features(image))
        
        return np.array(features)
    
    def train_model(self, X, y):
        """Train XGBoost model with optimal hyperparameters"""
        print("Training CactusNet-XGB model...")
        
        # Encode labels
        y_encoded = self.label_encoder.fit_transform(y)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
        )
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # XGBoost parameters for high accuracy
        xgb_params = {
            'objective': 'multi:softprob',
            'num_class': len(self.class_names),
            'max_depth': 8,
            'learning_rate': 0.1,
            'n_estimators': 200,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'reg_alpha': 0.1,
            'reg_lambda': 0.1,
            'random_state': 42,
            'n_jobs': -1,
            'verbosity': 0
        }
        
        # Train model
        self.xgb_model = xgb.XGBClassifier(**xgb_params)
        self.xgb_model.fit(X_train_scaled, y_train)
        
        # Evaluate
        y_pred = self.xgb_model.predict(X_test_scaled)
        accuracy = accuracy_score(y_test, y_pred)
        
        print(f"Model trained! Test Accuracy: {accuracy:.4f}")
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred, target_names=self.class_names))
        
        self.model_trained = True
        return accuracy
    
    def predict(self, image):
        """Predict disease from image with interpretability"""
        if not self.model_trained:
            return None, None, None
            
        # Extract features
        features = self.extract_features(image)
        features_scaled = self.scaler.transform(features.reshape(1, -1))
        
        # Predict
        probabilities = self.xgb_model.predict_proba(features_scaled)[0]
        predicted_class_idx = np.argmax(probabilities)
        predicted_class = self.class_names[predicted_class_idx]
        confidence = probabilities[predicted_class_idx]
        
        # Get feature importance for interpretability
        feature_importance = self.xgb_model.feature_importances_
        
        # Get top important features
        top_indices = np.argsort(feature_importance)[-10:][::-1]
        top_features = [(f"Feature_{i}", feature_importance[i]) for i in top_indices]
        
        return predicted_class, confidence, top_features
    
    def save_model(self, path):
        """Save trained model"""
        if self.model_trained:
            joblib.dump({
                'xgb_model': self.xgb_model,
                'scaler': self.scaler,
                'label_encoder': self.label_encoder,
                'class_names': self.class_names
            }, path)
            print(f"Model saved to {path}")
    
    def load_model(self, path):
        """Load trained model"""
        if os.path.exists(path):
            model_data = joblib.load(path)
            self.xgb_model = model_data['xgb_model']
            self.scaler = model_data['scaler']
            self.label_encoder = model_data['label_encoder']
            self.class_names = model_data['class_names']
            self.model_trained = True
            print(f"Model loaded from {path}")
            return True
        return False

# Initialize the model
detector = CactusNetXGB()

# Try to load existing model
model_path = 'models/cactusnet_xgb_model.pkl'
if not detector.load_model(model_path):
    print("No pre-trained model found. You need to train the model with your dataset.")

def allowed_file(filename):
    """Check if file type is allowed"""
    allowed_extensions = {'png', 'jpg', 'jpeg', 'gif', 'bmp'}
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in allowed_extensions

def preprocess_image(image_path):
    """Preprocess uploaded image"""
    image = cv2.imread(image_path)
    if image is None:
        # Try with PIL
        pil_image = Image.open(image_path).convert('RGB')
        image = np.array(pil_image)
    else:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    return image

@app.route('/')
def index():
    """Main page"""
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    """Handle image prediction"""
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file uploaded'}), 400
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        if not allowed_file(file.filename):
            return jsonify({'error': 'Invalid file type'}), 400
        
        # Save uploaded file
        filename = secure_filename(file.filename)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"{timestamp}_{filename}"
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        # Preprocess and predict
        image = preprocess_image(filepath)
        
        if not detector.model_trained:
            return jsonify({'error': 'Model not trained. Please train the model first.'}), 500
        
        predicted_class, confidence, feature_importance = detector.predict(image)
        
        # Get disease information
        disease_info = DISEASE_INFO.get(predicted_class, DISEASE_INFO['healthy'])
        
        # Prepare response
        result = {
            'predicted_class': predicted_class,
            'confidence': float(confidence),
            'disease_info': disease_info,
            'feature_importance': feature_importance,
            'timestamp': datetime.now().isoformat(),
            'filename': filename
        }
        
        # Clean up uploaded file
        try:
            os.remove(filepath)
        except:
            pass
        
        return jsonify(result)
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/train', methods=['POST'])
def train_model():
    """Train model with uploaded dataset"""
    try:
        if 'dataset' not in request.files:
            return jsonify({'error': 'No dataset uploaded'}), 400
        
        # This is a simplified training endpoint
        # In practice, you would handle dataset preprocessing here
        return jsonify({
            'message': 'Training endpoint ready. Please implement dataset loading logic.',
            'status': 'success'
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/model_status')
def model_status():
    """Get model training status"""
    return jsonify({
        'trained': detector.model_trained,
        'classes': detector.class_names if detector.model_trained else []
    })

# Create templates - FIXED VERSION
def create_templates():
    """Create HTML templates with proper encoding"""
    
    # Main template - Unicode characters replaced with HTML entities
    index_html = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>CactusNet-XGB Disease Detection</title>
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
            padding: 20px;
        }
        
        .container {
            max-width: 1200px;
            margin: 0 auto;
            background: white;
            border-radius: 15px;
            box-shadow: 0 20px 60px rgba(0,0,0,0.1);
            overflow: hidden;
        }
        
        .header {
            background: linear-gradient(135deg, #10B981, #059669);
            color: white;
            padding: 30px;
            text-align: center;
        }
        
        .header h1 {
            font-size: 2.5em;
            margin-bottom: 10px;
        }
        
        .header p {
            font-size: 1.1em;
            opacity: 0.9;
        }
        
        .main-content {
            padding: 40px;
        }
        
        .upload-section {
            text-align: center;
            margin-bottom: 40px;
        }
        
        .upload-area {
            border: 3px dashed #10B981;
            border-radius: 15px;
            padding: 60px 20px;
            background: #F0FDF4;
            cursor: pointer;
            transition: all 0.3s ease;
            margin: 20px 0;
        }
        
        .upload-area:hover {
            background: #DCFCE7;
            transform: translateY(-2px);
        }
        
        .upload-area.dragover {
            background: #DCFCE7;
            border-color: #059669;
        }
        
        .upload-icon {
            font-size: 4em;
            color: #10B981;
            margin-bottom: 20px;
        }
        
        .file-input {
            display: none;
        }
        
        .btn {
            background: #10B981;
            color: white;
            border: none;
            padding: 12px 30px;
            border-radius: 8px;
            cursor: pointer;
            font-size: 1em;
            transition: background 0.3s ease;
            margin: 10px;
        }
        
        .btn:hover {
            background: #059669;
        }
        
        .btn:disabled {
            background: #9CA3AF;
            cursor: not-allowed;
        }
        
        .results-section {
            display: none;
            margin-top: 40px;
        }
        
        .result-card {
            background: #F9FAFB;
            border-radius: 12px;
            padding: 30px;
            margin: 20px 0;
        }
        
        .disease-result {
            display: flex;
            align-items: center;
            gap: 20px;
            margin-bottom: 20px;
        }
        
        .disease-icon {
            width: 60px;
            height: 60px;
            border-radius: 50%;
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 24px;
            color: white;
        }
        
        .disease-info h3 {
            font-size: 1.5em;
            margin-bottom: 5px;
        }
        
        .confidence {
            background: #EFF6FF;
            color: #1D4ED8;
            padding: 4px 12px;
            border-radius: 20px;
            font-size: 0.9em;
            font-weight: 600;
        }
        
        .recommendations {
            margin-top: 20px;
        }
        
        .recommendations h4 {
            color: #374151;
            margin-bottom: 15px;
            display: flex;
            align-items: center;
            gap: 8px;
        }
        
        .rec-list {
            list-style: none;
            padding: 0;
        }
        
        .rec-list li {
            background: white;
            margin: 8px 0;
            padding: 12px 15px;
            border-left: 4px solid #10B981;
            border-radius: 0 8px 8px 0;
            box-shadow: 0 2px 4px rgba(0,0,0,0.05);
        }
        
        .loading {
            text-align: center;
            padding: 40px;
        }
        
        .spinner {
            border: 4px solid #f3f3f3;
            border-top: 4px solid #10B981;
            border-radius: 50%;
            width: 50px;
            height: 50px;
            animation: spin 1s linear infinite;
            margin: 0 auto 20px;
        }
        
        @keyframes spin {
            0% { transform: rotate(0deg); }
            100% { transform: rotate(360deg); }
        }
        
        .error {
            background: #FEF2F2;
            color: #DC2626;
            padding: 15px;
            border-radius: 8px;
            border-left: 4px solid #DC2626;
            margin: 20px 0;
        }
        
        .status-indicator {
            position: fixed;
            top: 20px;
            right: 20px;
            background: white;
            padding: 10px 15px;
            border-radius: 8px;
            box-shadow: 0 4px 12px rgba(0,0,0,0.1);
            font-size: 0.9em;
        }
        
        .status-trained {
            color: #10B981;
            border-left: 4px solid #10B981;
        }
        
        .status-not-trained {
            color: #DC2626;
            border-left: 4px solid #DC2626;
        }

        .feature-importance {
            margin-top: 20px;
            background: white;
            padding: 20px;
            border-radius: 8px;
        }
        
        .feature-item {
            display: flex;
            justify-content: space-between;
            align-items: center;
            padding: 8px 0;
            border-bottom: 1px solid #E5E7EB;
        }
        
        .feature-bar {
            width: 100px;
            height: 8px;
            background: #E5E7EB;
            border-radius: 4px;
            overflow: hidden;
        }
        
        .feature-bar-fill {
            height: 100%;
            background: #10B981;
            transition: width 0.3s ease;
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>&#127797; CactusNet-XGB</h1>
            <p>AI-Powered Plant Disease Detection with XGBoost Algorithm</p>
        </div>
        
        <div class="main-content">
            <div class="upload-section">
                <h2>Upload Plant Image for Analysis</h2>
                <div class="upload-area" onclick="document.getElementById('fileInput').click()">
                    <div class="upload-icon">&#128248;</div>
                    <h3>Click to upload or drag and drop</h3>
                    <p>Supports: JPG, PNG, GIF, BMP (Max 16MB)</p>
                </div>
                <input type="file" id="fileInput" class="file-input" accept="image/*">
                <button class="btn" onclick="document.getElementById('fileInput').click()">
                    Choose File
                </button>
                <button class="btn" id="analyzeBtn" onclick="analyzeImage()" disabled>
                    Analyze Image
                </button>
            </div>
            
            <div class="results-section" id="results">
                <div class="loading" id="loading">
                    <div class="spinner"></div>
                    <h3>Analyzing with CactusNet-XGB...</h3>
                    <p>Extracting features and detecting diseases...</p>
                </div>
            </div>
        </div>
    </div>
    
    <div class="status-indicator" id="modelStatus">
        Checking model status...
    </div>

    <script>
        let selectedFile = null;
        
        // Check model status on load
        window.onload = function() {
            checkModelStatus();
        };
        
        function checkModelStatus() {
            fetch('/model_status')
                .then(response => response.json())
                .then(data => {
                    const statusEl = document.getElementById('modelStatus');
                    if (data.trained) {
                        statusEl.className = 'status-indicator status-trained';
                        statusEl.innerHTML = '&#10003; Model Ready';
                    } else {
                        statusEl.className = 'status-indicator status-not-trained';
                        statusEl.innerHTML = '&#9888; Model Not Trained';
                    }
                });
        }
        
        // File input handling
        document.getElementById('fileInput').addEventListener('change', function(e) {
            selectedFile = e.target.files[0];
            if (selectedFile) {
                document.getElementById('analyzeBtn').disabled = false;
                document.querySelector('.upload-area h3').textContent = 'File selected: ' + selectedFile.name;
            }
        });
        
        // Drag and drop functionality
        const uploadArea = document.querySelector('.upload-area');
        
        uploadArea.addEventListener('dragover', function(e) {
            e.preventDefault();
            uploadArea.classList.add('dragover');
        });
        
        uploadArea.addEventListener('dragleave', function(e) {
            e.preventDefault();
            uploadArea.classList.remove('dragover');
        });
        
        uploadArea.addEventListener('drop', function(e) {
            e.preventDefault();
            uploadArea.classList.remove('dragover');
            
            const files = e.dataTransfer.files;
            if (files.length > 0) {
                selectedFile = files[0];
                document.getElementById('analyzeBtn').disabled = false;
                document.querySelector('.upload-area h3').textContent = 'File selected: ' + selectedFile.name;
            }
        });
        
        function analyzeImage() {
            if (!selectedFile) {
                alert('Please select an image first!');
                return;
            }
            
            const formData = new FormData();
            formData.append('file', selectedFile);
            
            // Show loading
            document.getElementById('results').style.display = 'block';
            document.getElementById('loading').style.display = 'block';
            
            fetch('/predict', {
                method: 'POST',
                body: formData
            })
            .then(response => response.json())
            .then(data => {
                document.getElementById('loading').style.display = 'none';
                
                if (data.error) {
                    showError(data.error);
                } else {
                    showResults(data);
                }
            })
            .catch(error => {
                document.getElementById('loading').style.display = 'none';
                showError('Error analyzing image: ' + error.message);
            });
        }
        
        function showResults(data) {
            const resultsEl = document.getElementById('results');
            const diseaseInfo = data.disease_info;
            
            resultsEl.innerHTML = `
                <div class="result-card">
                    <div class="disease-result">
                        <div class="disease-icon" style="background-color: ${diseaseInfo.color}">
                            ${diseaseInfo.name === 'Healthy' ? '&#10004;' : '&#9888;'}
                        </div>
                        <div class="disease-info">
                            <h3>${diseaseInfo.name}</h3>
                            <span class="confidence">${(data.confidence * 100).toFixed(1)}% Confidence</span>
                        </div>
                    </div>
                    
                    <p style="margin: 20px 0; color: #6B7280;">${diseaseInfo.description}</p>
                    
                    <div class="recommendations">
                        <h4>&#128295; Recommendations:</h4>
                        <ul class="rec-list">
                            ${diseaseInfo.recommendations.map(rec => `<li>${rec}</li>`).join('')}
                        </ul>
                    </div>
                    
                    <div class="feature-importance">
                        <h4>&#129504; Model Interpretability - Key Features:</h4>
                        ${data.feature_importance.map(([feature, importance]) => `
                            <div class="feature-item">
                                <span>${feature}</span>
                                <div class="feature-bar">
                                    <div class="feature-bar-fill" style="width: ${importance * 100}%"></div>
                                </div>
                                <span>${(importance * 100).toFixed(1)}%</span>
                            </div>
                        `).join('')}
                    </div>
                </div>
            `;
        }
        
        function showError(message) {
            const resultsEl = document.getElementById('results');
            resultsEl.innerHTML = `
                <div class="error">
                    <h3>&#10060; Error</h3>
                    <p>${message}</p>
                </div>
            `;
        }
    </script>
</body>
</html>"""

    # Write with proper UTF-8 encoding
    with open('templates/index.html', 'w', encoding='utf-8') as f:
        f.write(index_html)

# Dataset processing utility
def process_dataset(dataset_path):
    """
    Process dataset for training
    Expected structure:
    dataset/
    ├── healthy/
    ├── anthracnose/
    ├── leaf_spot/
    ├── rust/
    ├── rot/
    └── cochineal/
    """
    features = []
    labels = []
    
    for class_name in DISEASE_INFO.keys():
        class_path = os.path.join(dataset_path, class_name)
        if os.path.exists(class_path):
            print(f"Processing {class_name} images...")
            for image_file in os.listdir(class_path):
                if image_file.lower().endswith(('.png', '.jpg', '.jpeg')):
                    image_path = os.path.join(class_path, image_file)
                    try:
                        image = preprocess_image(image_path)
                        feature_vector = detector.extract_features(image)
                        features.append(feature_vector)
                        labels.append(class_name)
                    except Exception as e:
                        print(f"Error processing {image_path}: {e}")
    
    return np.array(features), np.array(labels)

@app.route('/train_with_dataset', methods=['POST'])
def train_with_dataset():
    """Train model with local dataset"""
    try:
        dataset_path = 'dataset'  # Assuming dataset is in dataset/ folder
        
        if not os.path.exists(dataset_path):
            return jsonify({'error': 'Dataset folder not found. Please create dataset/ folder with subfolders for each class.'}), 400
        
        # Process dataset
        print("Loading and processing dataset...")
        X, y = process_dataset(dataset_path)
        
        if len(X) == 0:
            return jsonify({'error': 'No valid images found in dataset.'}), 400
        
        print(f"Dataset loaded: {len(X)} images, {len(np.unique(y))} classes")
        
        # Train model
        accuracy = detector.train_model(X, y)
        
        # Save trained model
        detector.save_model(model_path)
        
        return jsonify({
            'status': 'success',
            'message': f'Model trained successfully with {accuracy:.4f} accuracy',
            'accuracy': accuracy,
            'samples': len(X),
            'classes': list(np.unique(y))
        })
        
    except Exception as e:
        return jsonify({'error': f'Training failed: {str(e)}'}), 500

@app.route('/upload_dataset', methods=['POST'])
def upload_dataset():
    """Handle dataset upload via web interface"""
    try:
        if 'files' not in request.files:
            return jsonify({'error': 'No files uploaded'}), 400
            
        files = request.files.getlist('files')
        labels = request.form.getlist('labels')
        
        if len(files) != len(labels):
            return jsonify({'error': 'Number of files and labels must match'}), 400
        
        features = []
        y_labels = []
        
        for file, label in zip(files, labels):
            if file and allowed_file(file.filename):
                # Save temporary file
                filename = secure_filename(file.filename)
                temp_path = os.path.join('uploads', f"temp_{filename}")
                file.save(temp_path)
                
                try:
                    # Process image
                    image = preprocess_image(temp_path)
                    feature_vector = detector.extract_features(image)
                    features.append(feature_vector)
                    y_labels.append(label)
                    
                    # Clean up
                    os.remove(temp_path)
                    
                except Exception as e:
                    print(f"Error processing uploaded file {filename}: {e}")
                    if os.path.exists(temp_path):
                        os.remove(temp_path)
        
        if len(features) == 0:
            return jsonify({'error': 'No valid images processed'}), 400
        
        # Train model
        X = np.array(features)
        y = np.array(y_labels)
        
        accuracy = detector.train_model(X, y)
        detector.save_model(model_path)
        
        return jsonify({
            'status': 'success',
            'message': f'Model trained with uploaded dataset. Accuracy: {accuracy:.4f}',
            'accuracy': accuracy,
            'samples': len(X)
        })
        
    except Exception as e:
        return jsonify({'error': f'Upload training failed: {str(e)}'}), 500

# Additional utility routes
@app.route('/download_model')
def download_model():
    """Download trained model"""
    if os.path.exists(model_path):
        return send_from_directory('models', 'cactusnet_xgb_model.pkl', as_attachment=True)
    else:
        return jsonify({'error': 'No trained model found'}), 404

@app.route('/model_info')
def model_info():
    """Get detailed model information"""
    if not detector.model_trained:
        return jsonify({'error': 'Model not trained'}), 400
    
    # Get feature importance
    feature_importance = detector.xgb_model.feature_importances_
    
    info = {
        'model_type': 'XGBoost Classifier',
        'classes': detector.class_names,
        'num_features': len(feature_importance),
        'feature_importance_stats': {
            'mean': float(np.mean(feature_importance)),
            'std': float(np.std(feature_importance)),
            'max': float(np.max(feature_importance)),
            'min': float(np.min(feature_importance))
        },
        'model_params': detector.xgb_model.get_params(),
        'training_timestamp': datetime.now().isoformat()
    }
    
    return jsonify(info)

# Health check endpoint
@app.route('/health')
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'model_trained': detector.model_trained,
        'timestamp': datetime.now().isoformat(),
        'version': '1.0.0'
    })

if __name__ == '__main__':
    # Create templates on startup
    create_templates()
    
    # Print startup information
    print("=" * 60)
    print("CactusNet-XGB Plant Disease Detection System")
    print("=" * 60)
    print("Features:")
    print("✓ XGBoost-based disease classification")
    print("✓ Feature-driven interpretable AI")
    print("✓ High-accuracy plant disease detection")
    print("✓ Web-based user interface")
    print("✓ Dataset training capabilities")
    print("✓ Model persistence and loading")
    print()
    print("Endpoints:")
    print("• GET  /              - Main web interface")
    print("• POST /predict       - Predict disease from image")
    print("• POST /train_with_dataset - Train with local dataset")
    print("• POST /upload_dataset - Train with uploaded files")
    print("• GET  /model_status  - Check model training status")
    print("• GET  /model_info    - Get detailed model information")
    print("• GET  /health        - Health check")
    print()
    print("Dataset Structure (for training):")
    print("dataset/")
    for disease in DISEASE_INFO.keys():
        print(f"├── {disease}/")
    print()
    print("Starting Flask development server...")
    print("=" * 60)

    app.run(debug=True, host='0.0.0.0', port=5000)