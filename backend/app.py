from flask import Flask, request, jsonify
from flask_cors import CORS
from flask_jwt_extended import JWTManager, create_access_token, create_refresh_token, jwt_required, get_jwt_identity, get_jwt
from models import db, User, Token, Category, DEFAULT_CATEGORIES
from config import config
from sqlalchemy import text
import os
import logging
from datetime import datetime, timezone, timedelta
import re
import traceback
import sys
import cv2
import numpy as np
import pytesseract
from PIL import Image, ImageEnhance, ImageFilter
import io
import base64
import json

# Initialize Flask application
app = Flask(__name__)
CORS(app, supports_credentials=True)

# Load configuration
config_name = os.environ.get('FLASK_ENV', 'development')
app.config.from_object(config[config_name])

# JWT Configuration
app.config['JWT_SECRET_KEY'] = os.getenv('JWT_SECRET_KEY', 'your-secret-key-change-this-in-production-immediately')
app.config['JWT_ACCESS_TOKEN_EXPIRES'] = timedelta(hours=24)
app.config['JWT_REFRESH_TOKEN_EXPIRES'] = timedelta(days=30)
app.config['JWT_ALGORITHM'] = 'HS256'

# Initialize extensions
db.init_app(app)
jwt = JWTManager(app)

# Token blacklist
blacklisted_tokens = set()

# Enable CORS with proper configuration
CORS(app, 
     origins=["http://localhost:3000", "http://127.0.0.1:3000"], 
     supports_credentials=True,
     allow_headers=["Content-Type", "Authorization"],
     methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"])

# Logger configuration
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout),
              logging.FileHandler('app.log')]
)
logger = logging.getLogger(__name__)

# JWT Token Blacklist Check
@jwt.token_in_blocklist_loader
def check_if_token_revoked(jwt_header, jwt_payload):
    jti = jwt_payload['jti']
    return jti in blacklisted_tokens

# Helper functions for validation, token creation, etc.
def validate_email(email):
    """Validate email format"""
    if not email or not isinstance(email, str):
        return False
    pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    return re.match(pattern, email.strip()) is not None

def validate_password(password):
    """Validate password strength"""
    if not password or not isinstance(password, str):
        return False
    password = password.strip()
    if len(password) < 6:
        return False
    return True

# Class for OCR and receipt processing
class EnhancedReceiptOCR:
    def __init__(self):
        self.configs = [
            r'--oem 3 --psm 6 -c tessedit_char_whitelist=0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz.,-:/()* ',
            r'--oem 3 --psm 4',
            r'--oem 3 --psm 11',
            r'--oem 3 --psm 13',
            r'--oem 1 --psm 6',
        ]
        
    def enhance_image(self, image):
        """Enhanced image preprocessing for better OCR"""
        try:
            if isinstance(image, Image.Image):
                opencv_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
            else:
                opencv_image = image
            gray = cv2.cvtColor(opencv_image, cv2.COLOR_BGR2GRAY)
            blurred = cv2.GaussianBlur(gray, (5, 5), 0)
            _, thresh2 = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            kernel = np.ones((2, 2), np.uint8)
            opened = cv2.morphologyEx(thresh2, cv2.MORPH_OPEN, kernel, iterations=1)
            closed = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, kernel, iterations=1)
            dilated = cv2.dilate(closed, kernel, iterations=1)
            kernel_sharp = np.array([[-1,-1,-1], [-1, 9,-1], [-1,-1,-1]])
            sharpened = cv2.filter2D(dilated, -1, kernel_sharp)
            return Image.fromarray(sharpened)
        except Exception as e:
            print(f"Error in image enhancement: {e}")
            return image
    
    def extract_text_multiple_configs(self, image):
        results = []
        enhanced_image = self.enhance_image(image)
        for i, config in enumerate(self.configs):
            try:
                text = pytesseract.image_to_string(enhanced_image, config=config, lang='eng+ind')
                if text.strip():
                    results.append((text.strip(), f"enhanced_config_{i+1}"))
            except Exception as e:
                continue
        if results:
            best_result = max(results, key=lambda x: len(x[0]))
            return best_result[0]
        else:
            return ""

ocr_processor = EnhancedReceiptOCR()

# Routes for Authentication
@app.route('/api/auth/register', methods=['POST'])
def register():
    # Registration logic here...
    pass

@app.route('/api/auth/login', methods=['POST'])
def login():
    # Login logic here...
    pass

@app.route('/api/auth/logout', methods=['POST'])
@jwt_required()
def logout():
    # Logout logic here...
    pass

# Routes for Receipt OCR
@app.route('/api/scan-receipt', methods=['POST'])
def scan_receipt():
    try:
        if 'image' not in request.files:
            return jsonify({'error': 'No image file provided'}), 400
        image_file = request.files['image']
        image_bytes = image_file.read()
        image = Image.open(io.BytesIO(image_bytes))
        extracted_text = ocr_processor.extract_text_multiple_configs(image)
        if not extracted_text:
            return jsonify({'error': 'No text could be extracted from the image. Please try a clearer image.'}), 400
        return jsonify({'success': True, 'data': extracted_text})
    except Exception as e:
        return jsonify({'error': f'Error processing receipt: {str(e)}'}), 500

@app.route('/api/health', methods=['GET'])
def health_check():
    return jsonify({'status': 'OK', 'message': 'API is running'})

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(debug=True, host='0.0.0.0', port=port)
