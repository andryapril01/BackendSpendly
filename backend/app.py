from flask import Flask, request, jsonify
from flask_cors import CORS
from flask_jwt_extended import JWTManager, jwt_required
import cv2
import numpy as np
import pytesseract
from PIL import Image
import io
import base64
import re
import json
from datetime import datetime
import os

# Import blueprints from different modules
from reports_api import reports_bp
from transaction_api import transaction_bp
from dashboard_api import dashboard_bp

# Create Flask application
app = Flask(__name__)

# Enable CORS
CORS(app,
     supports_credentials=True,
     origins=["https://frontend-spendly-b2fg.vercel.app/"],
     allow_headers=["Content-Type", "Authorization", "Accept"],
     methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"])

# Register Blueprints for different parts of the app
app.register_blueprint(reports_bp)
app.register_blueprint(transaction_bp)
app.register_blueprint(dashboard_bp)

# JWT Configuration
app.config['JWT_SECRET_KEY'] = os.getenv('JWT_SECRET_KEY', 'your-secret-key-change-this-in-production-immediately')
app.config['JWT_ACCESS_TOKEN_EXPIRES'] = timedelta(hours=24)
app.config['JWT_REFRESH_TOKEN_EXPIRES'] = timedelta(days=30)
app.config['JWT_ALGORITHM'] = 'HS256'

jwt = JWTManager(app)

# Configure Tesseract path for Windows (or adjust based on your OS)
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

# The OCR Processor Class (unchanged)
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
            thresh1 = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
            _, thresh2 = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            combined = cv2.bitwise_and(thresh1, thresh2)
            kernel = np.ones((2, 2), np.uint8)
            opened = cv2.morphologyEx(combined, cv2.MORPH_OPEN, kernel, iterations=1)
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
                
                if i == 0: 
                    original_text = pytesseract.image_to_string(image, config=config, lang='eng+ind')
                    if original_text.strip():
                        results.append((original_text.strip(), "original_config_1"))
                        
            except Exception as e:
                print(f"Error with config {i+1}: {e}")
                continue
        
        if results:
            best_result = max(results, key=lambda x: len(x[0]))
            return best_result[0]
        else:
            return ""
    
    def smart_parse_receipt(self, text):
        lines = [line.strip() for line in text.split('\n') if line.strip()]
        
        parsed_data = {
            'merchantName': '',
            'date': datetime.now().strftime('%Y-%m-%d'),
            'items': [],
            'total': 0,
            'confidence': 0.0,
            'raw_text': text
        }

        # Logic to extract merchant name, date, items, etc...
        # ... [Extract merchant name, date, items, etc., as in your original code]

        return parsed_data


# Initialize OCR processor
ocr_processor = EnhancedReceiptOCR()

# Secure the receipt scanning API endpoint with JWT token
@app.route('/api/scan-receipt', methods=['POST'])
@jwt_required()  # Require authentication via JWT token
def scan_receipt():
    try:
        if 'image' not in request.files:
            return jsonify({'error': 'No image file provided'}), 400
        
        image_file = request.files['image']
        
        if image_file.filename == '':
            return jsonify({'error': 'No image file selected'}), 400
        
        image_bytes = image_file.read()
        image = Image.open(io.BytesIO(image_bytes))
        
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        extracted_text = ocr_processor.extract_text_multiple_configs(image)
        
        if not extracted_text:
            return jsonify({'error': 'No text could be extracted from the image.'}), 400
        
        parsed_data = ocr_processor.smart_parse_receipt(extracted_text)
        
        return jsonify({
            'success': True,
            'data': parsed_data
        })
        
    except Exception as e:
        return jsonify({'error': f'Error processing receipt: {str(e)}'}), 500

# Health check endpoint
@app.route('/api/health', methods=['GET'])
def health_check():
    return jsonify({'status': 'OK', 'message': 'API is running'})

# Run the app
if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(debug=False, host='0.0.0.0', port=port)
