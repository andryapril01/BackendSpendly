# app.py - Enhanced OCR Version
from flask import Flask, request, jsonify
from flask_cors import CORS
import cv2
import numpy as np
import pytesseract
from PIL import Image, ImageEnhance, ImageFilter
import io
import base64
import re
import json
from datetime import datetime
import os

# Import blueprints
from reports_api import reports_bp


app = Flask(__name__)
CORS(app, supports_credentials=True)

# Register blueprints
app.register_blueprint(reports_bp)

# Configure Tesseract path for Windows
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

class EnhancedReceiptOCR:
    def __init__(self):
        # Multiple OCR configurations untuk different receipt types
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
            # Convert PIL to OpenCV
            if isinstance(image, Image.Image):
                opencv_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
            else:
                opencv_image = image
            
            # Step 1: Convert to grayscale
            gray = cv2.cvtColor(opencv_image, cv2.COLOR_BGR2GRAY)
            
            # Step 2: Increase image size (upscaling)
            height, width = gray.shape
            if height < 1000:
                scale_factor = 1000 / height
                new_width = int(width * scale_factor)
                new_height = int(height * scale_factor)
                gray = cv2.resize(gray, (new_width, new_height), interpolation=cv2.INTER_CUBIC)
            
            # Step 3: Apply Gaussian blur to reduce noise
            blurred = cv2.GaussianBlur(gray, (5, 5), 0)
            
            # Step 4: Apply multiple thresholding techniques
            # Method 1: Adaptive threshold
            thresh1 = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                          cv2.THRESH_BINARY, 11, 2)
            
            # Method 2: Otsu's threshold
            _, thresh2 = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            
            # Method 3: Simple threshold
            _, thresh3 = cv2.threshold(blurred, 127, 255, cv2.THRESH_BINARY)
            
            # Combine thresholds (take the best parts)
            combined = cv2.bitwise_and(thresh1, thresh2)
            
            # Step 5: Morphological operations
            kernel = np.ones((2, 2), np.uint8)
            
            # Opening to remove small noise
            opened = cv2.morphologyEx(combined, cv2.MORPH_OPEN, kernel, iterations=1)
            
            # Closing to fill small gaps
            closed = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, kernel, iterations=1)
            
            # Step 6: Dilation to make text thicker
            dilated = cv2.dilate(closed, kernel, iterations=1)
            
            # Step 7: Final sharpening
            kernel_sharp = np.array([[-1,-1,-1],
                                   [-1, 9,-1],
                                   [-1,-1,-1]])
            sharpened = cv2.filter2D(dilated, -1, kernel_sharp)
            
            return Image.fromarray(sharpened)
            
        except Exception as e:
            print(f"Error in image enhancement: {e}")
            return image
    
    def extract_text_multiple_configs(self, image):
        """Try multiple OCR configurations and return best result"""
        results = []
        enhanced_image = self.enhance_image(image)
        
        # Try each configuration
        for i, config in enumerate(self.configs):
            try:
                print(f"Trying OCR config {i+1}/{len(self.configs)}: {config}")
                
                # Try with enhanced image
                text = pytesseract.image_to_string(enhanced_image, config=config, lang='eng+ind')
                if text.strip():
                    results.append((text.strip(), f"enhanced_config_{i+1}"))
                
                # Also try with original image
                if i == 0:  # Only for first config to save time
                    original_text = pytesseract.image_to_string(image, config=config, lang='eng+ind')
                    if original_text.strip():
                        results.append((original_text.strip(), "original_config_1"))
                        
            except Exception as e:
                print(f"Error with config {i+1}: {e}")
                continue
        
        # Return the longest result as it's likely more complete
        if results:
            best_result = max(results, key=lambda x: len(x[0]))
            print(f"Best OCR result from {best_result[1]}: {len(best_result[0])} characters")
            return best_result[0]
        else:
            return ""
    
    def smart_parse_receipt(self, text):
        """Enhanced parsing with better pattern recognition"""
        print("Raw OCR Text:")
        print("=" * 50)
        print(text)
        print("=" * 50)
        
        lines = [line.strip() for line in text.split('\n') if line.strip()]
        
        parsed_data = {
            'merchantName': '',
            'date': datetime.now().strftime('%Y-%m-%d'),
            'items': [],
            'total': 0,
            'confidence': 0.0,
            'raw_text': text
        }
        
        # Extract merchant name (first few clean lines)
        for i, line in enumerate(lines[:5]):
            # Skip lines with lots of numbers or special chars
            if not re.search(r'\d{3,}', line) and len(line) > 3 and len(line) < 50:
                # Clean up the merchant name
                cleaned = re.sub(r'[^\w\s&.-]', ' ', line)
                cleaned = ' '.join(cleaned.split())
                if len(cleaned) > 3:
                    parsed_data['merchantName'] = cleaned
                    break
        
        # Enhanced date extraction
        date_patterns = [
            r'(\d{1,2})[\/\-\.](\d{1,2})[\/\-\.](\d{2,4})',
            r'(\d{2,4})[\/\-\.](\d{1,2})[\/\-\.](\d{1,2})',
            r'(\d{1,2})\s+(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\s+(\d{2,4})',
        ]
        
        for line in lines:
            for pattern in date_patterns:
                match = re.search(pattern, line, re.IGNORECASE)
                if match:
                    groups = match.groups()
                    try:
                        if pattern == date_patterns[0]:  # DD/MM/YYYY
                            day, month, year = groups
                            if len(year) == 2:
                                year = f"20{year}"
                            parsed_data['date'] = f"{year}-{month.zfill(2)}-{day.zfill(2)}"
                        elif pattern == date_patterns[1]:  # YYYY/MM/DD
                            year, month, day = groups
                            parsed_data['date'] = f"{year}-{month.zfill(2)}-{day.zfill(2)}"
                        break
                    except:
                        continue
        
        # Enhanced total extraction
        total_patterns = [
            r'(?:TOTAL|GRAND\s*TOTAL|SUB\s*TOTAL|HARGA\s*JUAL)\s*:?\s*(?:RP\.?|IDR|Rp)?\s*([\d\.,]+)',
            r'(?:RP\.?|IDR|Rp)\s*([\d\.,]+)(?:\s*(?:TOTAL|JUMLAH))?',
            r'TOTAL.*?([\d\.,]{4,})',
            r'([\d\.,]{4,})\s*(?:TOTAL|JUMLAH)',
        ]
        
        for line in lines:
            for pattern in total_patterns:
                match = re.search(pattern, line, re.IGNORECASE)
                if match:
                    amount_str = match.group(1)
                    # Extract only numbers
                    amount_str = re.sub(r'[^\d]', '', amount_str)
                    if amount_str and len(amount_str) >= 3:  # Minimum 3 digits
                        parsed_data['total'] = int(amount_str)
                        break
            if parsed_data['total'] > 0:
                break
        
        # Enhanced item extraction
        items = self.extract_items_smart(lines)
        parsed_data['items'] = items
        
        # If no items found, add one empty item
        if not parsed_data['items']:
            parsed_data['items'] = [{'name': '', 'quantity': 1, 'price': 0}]
        
        # Calculate confidence
        parsed_data['confidence'] = self.calculate_confidence(parsed_data, text)
        
        return parsed_data
    
    def extract_items_smart(self, lines):
        """Smart item extraction with multiple patterns"""
        items = []
        
        # Enhanced item patterns
        patterns = [
            # Pattern 1: Item Qty Price Total
            r'^(.+?)\s+(\d+)\s+(?:RP\.?|IDR|Rp)?\s*([\d\.,]+)\s+(?:RP\.?|IDR|Rp)?\s*([\d\.,]+)$',
            # Pattern 2: Item Qty x Price
            r'^(.+?)\s+(\d+)\s*[xX*]\s*(?:RP\.?|IDR|Rp)?\s*([\d\.,]+)',
            # Pattern 3: Item Total
            r'^(.+?)\s+(?:RP\.?|IDR|Rp)?\s*([\d\.,]{4,})$',
            # Pattern 4: Item with quantity in parentheses
            r'^(.+?)\s*\((\d+)\)\s*(?:RP\.?|IDR|Rp)?\s*([\d\.,]+)',
        ]
        
        for line in lines:
            # Skip header and footer lines
            if re.search(r'(?:NAMA|ITEM|QTY|HARGA|TOTAL|GRAND|SUB|KASIR|TANGGAL|NO\.)', line, re.IGNORECASE):
                continue
            
            for pattern in patterns:
                match = re.search(pattern, line, re.IGNORECASE)
                if match:
                    groups = match.groups()
                    
                    try:
                        if len(groups) == 4:  # Item Qty Price Total
                            name = groups[0].strip()
                            quantity = int(groups[1])
                            unit_price = int(re.sub(r'[^\d]', '', groups[2]))
                            total_price = int(re.sub(r'[^\d]', '', groups[3]))
                            
                            # Validate the calculation
                            if unit_price > 0 and abs(quantity * unit_price - total_price) <= total_price * 0.15:
                                items.append({
                                    'name': name,
                                    'quantity': quantity,
                                    'price': unit_price
                                })
                        
                        elif len(groups) == 3:  # Item Qty x Price or Item (Qty) Price
                            name = groups[0].strip()
                            quantity = int(groups[1])
                            price = int(re.sub(r'[^\d]', '', groups[2]))
                            
                            if price > 0 and quantity > 0:
                                items.append({
                                    'name': name,
                                    'quantity': quantity,
                                    'price': price
                                })
                        
                        elif len(groups) == 2:  # Item Total
                            name = groups[0].strip()
                            price = int(re.sub(r'[^\d]', '', groups[1]))
                            
                            # Only add if price seems reasonable (> 1000 IDR)
                            if price > 1000 and len(name) > 2:
                                items.append({
                                    'name': name,
                                    'quantity': 1,
                                    'price': price
                                })
                        
                        break
                    except:
                        continue
        
        # Remove duplicates and filter out invalid items
        seen = set()
        filtered_items = []
        for item in items:
            if item['name'] and item['price'] > 0:
                # Create a key for duplicate detection
                key = (item['name'].lower().strip(), item['price'])
                if key not in seen:
                    seen.add(key)
                    filtered_items.append(item)
        
        return filtered_items
    
    def calculate_confidence(self, parsed_data, raw_text):
        """Calculate confidence score"""
        score = 0.0
        
        # Merchant name (25%)
        if parsed_data['merchantName']:
            score += 0.25
        
        # Total amount (35%)
        if parsed_data['total'] > 0:
            score += 0.35
        
        # Items (25%)
        if parsed_data['items'] and any(item['name'] for item in parsed_data['items']):
            score += 0.25
        
        # Text quality (15%)
        if len(raw_text) > 50:
            score += 0.15
        
        return min(score, 1.0)

# Initialize OCR processor
ocr_processor = EnhancedReceiptOCR()

@app.route('/api/scan-receipt', methods=['POST'])
def scan_receipt():
    try:
        # Get image from request
        if 'image' not in request.files:
            return jsonify({'error': 'No image file provided'}), 400
        
        image_file = request.files['image']
        
        if image_file.filename == '':
            return jsonify({'error': 'No image file selected'}), 400
        
        print(f"Processing image: {image_file.filename}")
        
        # Read and process image
        image_bytes = image_file.read()
        image = Image.open(io.BytesIO(image_bytes))
        
        # Convert to RGB if necessary
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        print(f"Image size: {image.size}")
        
        # Extract text from image using enhanced method
        extracted_text = ocr_processor.extract_text_multiple_configs(image)
        
        if not extracted_text:
            return jsonify({'error': 'No text could be extracted from the image. Please try a clearer image.'}), 400
        
        # Parse the extracted text
        parsed_data = ocr_processor.smart_parse_receipt(extracted_text)
        
        print(f"Parsed data: {json.dumps(parsed_data, indent=2)}")
        
        return jsonify({
            'success': True,
            'data': parsed_data
        })
        
    except Exception as e:
        print(f"Error processing receipt: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': f'Error processing receipt: {str(e)}'}), 500

@app.route('/api/health', methods=['GET'])
def health_check():
    return jsonify({'status': 'OK', 'message': 'Enhanced Receipt OCR API is running'})

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(debug=False, host='0.0.0.0', port=port)
