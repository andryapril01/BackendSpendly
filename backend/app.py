# app.py - Combined Authentication & OCR Server
from flask import Flask, request, jsonify
from flask_cors import CORS
from flask_jwt_extended import JWTManager, create_access_token, create_refresh_token, jwt_required, get_jwt_identity, get_jwt
from models import db, User, Token, Category, DEFAULT_CATEGORIES
from config import config
from sqlalchemy import text
import cv2
import numpy as np
import pytesseract
from PIL import Image, ImageEnhance, ImageFilter
import io
import base64
import re
import json
from datetime import datetime, timezone, timedelta
import os
import logging
import traceback
import sys

# Create Flask application
app = Flask(__name__)

# Load configuration
config_name = os.environ.get('FLASK_ENV', 'production')
app.config.from_object(config[config_name])

# JWT Configuration - Extended token expiry for development
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
     origins=["http://localhost:3000", "https://frontend-spendly-b2fg.vercel.app"], 
     supports_credentials=True,
     allow_headers=["Content-Type", "Authorization"],
     methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"])

# Import and register blueprints AFTER app initialization
try:
    from reports_api import reports_bp
    from dashboard_api import dashboard_bp
    from transaction_api import transaction_bp
    app.register_blueprint(reports_bp)
    app.register_blueprint(dashboard_bp)
    app.register_blueprint(transaction_bp)
    print("✅ Blueprints registered successfully")
except ImportError as e:
    print(f"⚠️ Warning: Could not import blueprints: {e}")

if os.name == 'nt':
    pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
else:
    pytesseract.pytesseract.tesseract_cmd = '/usr/bin/tesseract'

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('auth.log')
    ]
)
logger = logging.getLogger(__name__)

# JWT Token Blacklist Check
@jwt.token_in_blocklist_loader
def check_if_token_revoked(jwt_header, jwt_payload):
    jti = jwt_payload['jti']
    return jti in blacklisted_tokens

# JWT Error Handlers
@jwt.expired_token_loader
def expired_token_callback(jwt_header, jwt_payload):
    logger.warning(f"Expired token accessed: {jwt_payload.get('sub', 'unknown')}")
    return jsonify({
        'error': 'Token has expired',
        'message': 'Your session has expired. Please login again.',
        'code': 'TOKEN_EXPIRED'
    }), 401

@jwt.invalid_token_loader
def invalid_token_callback(error):
    logger.warning(f"Invalid token: {str(error)}")
    return jsonify({
        'error': 'Invalid token',
        'message': 'The provided token is invalid. Please login again.',
        'code': 'INVALID_TOKEN'
    }), 401

@jwt.unauthorized_loader
def missing_token_callback(error):
    logger.warning(f"Missing token: {str(error)}")
    return jsonify({
        'error': 'Authorization token required',
        'message': 'Please provide a valid access token.',
        'code': 'MISSING_TOKEN'
    }), 401

@jwt.revoked_token_loader
def revoked_token_callback(jwt_header, jwt_payload):
    logger.warning(f"Revoked token accessed: {jwt_payload.get('sub', 'unknown')}")
    return jsonify({
        'error': 'Token has been revoked',
        'message': 'Your session has been terminated. Please login again.',
        'code': 'TOKEN_REVOKED'
    }), 401

# Global error handler
@app.errorhandler(Exception)
def handle_exception(e):
    logger.error(f"Unhandled exception: {str(e)}")
    logger.error(f"Traceback: {traceback.format_exc()}")
    
    return jsonify({
        'error': 'Internal server error',
        'message': str(e) if app.config.get('DEBUG') else 'An unexpected error occurred',
        'code': 'INTERNAL_ERROR'
    }), 500

# Enhanced Receipt OCR Class
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

# Helper functions for authentication
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
    
    if len(password) < 6:  # Relaxed for development
        return False
    return True

def create_default_categories(user_id):
    """Create default categories for a new user"""
    categories = []
    for cat_data in DEFAULT_CATEGORIES:
        category = Category(
            user_id=user_id,
            name=cat_data['name'],
            icon=cat_data['icon'],
            color=cat_data['color'],
            is_default=True,
            budget=1000000
        )
        categories.append(category)
    return categories

def create_tokens_for_user(user_id):
    """Create access and refresh tokens for a user"""
    user_id_str = str(user_id)
    
    access_token = create_access_token(
        identity=user_id_str,
        expires_delta=app.config['JWT_ACCESS_TOKEN_EXPIRES']
    )
    refresh_token = create_refresh_token(
        identity=user_id_str,
        expires_delta=app.config['JWT_REFRESH_TOKEN_EXPIRES']
    )
    return access_token, refresh_token

# Database connection test
def test_database_connection():
    """Test database connection"""
    try:
        with app.app_context():
            db.session.execute(text('SELECT 1'))
            return True
    except Exception as e:
        logger.error(f"Database connection failed: {str(e)}")
        return False

# ==================== ROUTES ====================

# Health check endpoint
@app.route('/api/health', methods=['GET'])
def health_check():
    try:
        db_status = 'connected' if test_database_connection() else 'disconnected'
    except Exception as e:
        logger.error(f"Database health check failed: {str(e)}")
        db_status = 'disconnected'
    
    return jsonify({
        'status': 'OK', 
        'message': 'Combined Auth & OCR API is running',
        'timestamp': datetime.now().isoformat(),
        'version': '2.0.3',
        'database': db_status,
        'jwt_config': {
            'access_token_expires': str(app.config['JWT_ACCESS_TOKEN_EXPIRES']),
            'refresh_token_expires': str(app.config['JWT_REFRESH_TOKEN_EXPIRES'])
        }
    }), 200

# ==================== AUTHENTICATION ROUTES ====================

@app.route('/api/auth/register', methods=['POST'])
def register():
    try:
        logger.info("=== REGISTRATION REQUEST ===")
        
        if not request.is_json:
            return jsonify({
                'error': 'Content-Type must be application/json',
                'code': 'INVALID_CONTENT_TYPE'
            }), 400
        
        data = request.get_json()
        if not data:
            return jsonify({
                'error': 'Request body is empty or invalid JSON',
                'code': 'EMPTY_REQUEST_BODY'
            }), 400
        
        logger.info(f"Registration data received for: {data.get('email', 'no email')}")
        
        # Validate required fields
        required_fields = ['firstName', 'lastName', 'email', 'password']
        missing_fields = []
        
        for field in required_fields:
            if field not in data or not str(data[field]).strip():
                missing_fields.append(field)
        
        if missing_fields:
            logger.warning(f"Missing required fields: {missing_fields}")
            return jsonify({
                'error': f'Missing required fields: {", ".join(missing_fields)}',
                'code': 'MISSING_REQUIRED_FIELDS',
                'missing_fields': missing_fields
            }), 400
        
        # Clean and validate data
        email = str(data['email']).strip().lower()
        password = str(data['password']).strip()
        first_name = str(data['firstName']).strip()
        last_name = str(data['lastName']).strip()
        phone = str(data.get('phone', '')).strip()
        
        # Validate email format
        if not validate_email(email):
            logger.warning(f"Invalid email format: {email}")
            return jsonify({
                'error': 'Invalid email format',
                'code': 'INVALID_EMAIL_FORMAT'
            }), 400
        
        # Validate password strength
        if not validate_password(password):
            logger.warning("Password validation failed")
            return jsonify({
                'error': 'Password must be at least 6 characters',
                'code': 'WEAK_PASSWORD'
            }), 400
        
        # Check if user already exists
        existing_user = User.query.filter_by(email=email).first()
        if existing_user:
            logger.warning(f"Email already exists: {email}")
            return jsonify({
                'error': 'An account with this email already exists',
                'code': 'EMAIL_ALREADY_EXISTS'
            }), 409
        
        # Create new user
        new_user = User(
            email=email,
            first_name=first_name,
            last_name=last_name,
            phone=phone,
            password=password,
            is_verified=True
        )
        
        db.session.add(new_user)
        db.session.flush()
        
        logger.info(f"User created with ID: {new_user.id}")
        
        # Create default categories for this user
        categories = create_default_categories(new_user.id)
        for category in categories:
            db.session.add(category)
        
        logger.info(f"Created {len(categories)} default categories")
        
        db.session.commit()
        
        # Generate JWT tokens
        access_token, refresh_token = create_tokens_for_user(new_user.id)
        
        logger.info(f"✅ Registration successful for user: {new_user.email}")
        
        return jsonify({
            'message': 'Registration successful',
            'access_token': access_token,
            'refresh_token': refresh_token,
            'user': {
                'id': new_user.id,
                'email': new_user.email,
                'firstName': new_user.first_name,
                'lastName': new_user.last_name,
                'phone': new_user.phone
            }
        }), 201
        
    except Exception as e:
        db.session.rollback()
        logger.error(f"❌ Registration error: {str(e)}")
        logger.error(f"❌ Traceback: {traceback.format_exc()}")
        return jsonify({
            'error': 'Registration failed due to server error',
            'message': str(e),
            'code': 'REGISTRATION_ERROR'
        }), 500

@app.route('/api/auth/login', methods=['POST'])
def login():
    try:
        logger.info("=== LOGIN REQUEST START ===")
        
        # Test database connection first
        if not test_database_connection():
            logger.error("❌ Database connection failed")
            return jsonify({
                'error': 'Database connection failed',
                'message': 'Cannot connect to database. Please try again later.',
                'code': 'DATABASE_ERROR'
            }), 503
        
        if not request.is_json:
            return jsonify({
                'error': 'Content-Type must be application/json',
                'code': 'INVALID_CONTENT_TYPE'
            }), 400
        
        data = request.get_json()
        if not data:
            return jsonify({
                'error': 'Request body is empty or invalid JSON',
                'code': 'EMPTY_REQUEST_BODY'
            }), 400
        
        logger.info(f"Login attempt for email: {data.get('email')}")
        
        # Validate required fields
        if not data.get('email') or not data.get('password'):
            logger.warning("Missing email or password")
            return jsonify({
                'error': 'Email and password are required',
                'code': 'MISSING_CREDENTIALS'
            }), 400
        
        email = str(data['email']).strip().lower()
        password = str(data['password']).strip()
        
        logger.info(f"�� Looking up user: {email}")
        
        # Find user by email with explicit error handling
        try:
            user = User.query.filter_by(email=email).first()
            logger.info(f"�� User lookup result: {'Found' if user else 'Not found'}")
        except Exception as db_error:
            logger.error(f"❌ Database query error: {str(db_error)}")
            logger.error(f"❌ DB Traceback: {traceback.format_exc()}")
            return jsonify({
                'error': 'Database query failed',
                'message': 'Error accessing user data. Please try again.',
                'code': 'DATABASE_QUERY_ERROR'
            }), 500
        
        if not user:
            logger.warning(f"❌ User not found: {email}")
            return jsonify({
                'error': 'Invalid email or password',
                'code': 'INVALID_CREDENTIALS'
            }), 401
        
        logger.info(f"✅ User found: {user.email}, ID: {user.id}")
        
        # Check password with explicit error handling
        try:
            password_valid = user.check_password(password)
            logger.info(f"🔐 Password check result: {'Valid' if password_valid else 'Invalid'}")
        except Exception as pwd_error:
            logger.error(f"❌ Password check error: {str(pwd_error)}")
            logger.error(f"❌ PWD Traceback: {traceback.format_exc()}")
            return jsonify({
                'error': 'Password verification failed',
                'message': 'Error checking password. Please try again.',
                'code': 'PASSWORD_CHECK_ERROR'
            }), 500
        
        if not password_valid:
            logger.warning(f"❌ Invalid password for user: {email}")
            return jsonify({
                'error': 'Invalid email or password',
                'code': 'INVALID_CREDENTIALS'
            }), 401
        
        # Check if user is active
        if not user.is_active:
            logger.warning(f"❌ Inactive user attempted login: {user.email}")
            return jsonify({
                'error': 'This account has been deactivated',
                'code': 'ACCOUNT_DEACTIVATED'
            }), 403
        
        logger.info(f"✅ Authentication successful for: {user.email}")
        
        # Create tokens with explicit error handling
        try:
            access_token, refresh_token = create_tokens_for_user(user.id)
            logger.info(f"✅ Tokens created successfully")
        except Exception as token_error:
            logger.error(f"❌ Token creation error: {str(token_error)}")
            logger.error(f"❌ Token Traceback: {traceback.format_exc()}")
            return jsonify({
                'error': 'Token creation failed',
                'message': 'Error creating authentication tokens. Please try again.',
                'code': 'TOKEN_CREATION_ERROR'
            }), 500
        
        # Update last login timestamp
        try:
            user.last_login = datetime.now(timezone.utc)
            db.session.commit()
            logger.info(f"✅ Last login updated")
        except Exception as update_error:
            logger.warning(f"⚠️ Failed to update last login: {str(update_error)}")
            # This is not critical, continue with login
        
        # Prepare user data
        user_data = {
            'id': user.id,
            'email': user.email,
            'firstName': user.first_name,
            'lastName': user.last_name,
            'phone': user.phone if user.phone else ''
        }
        
        logger.info(f"🎉 LOGIN SUCCESSFUL for user: {user.email}")
        
        # Return success response
        return jsonify({
            'message': 'Login successful',
            'access_token': access_token,
            'refresh_token': refresh_token,
            'user': user_data
        }), 200
            
    except Exception as e:
        logger.error(f"❌ CRITICAL LOGIN ERROR: {str(e)}")
        logger.error(f"❌ CRITICAL Traceback: {traceback.format_exc()}")
        return jsonify({
            'error': 'Login failed due to server error',
            'message': 'An unexpected error occurred. Please try again later.',
            'code': 'LOGIN_ERROR'
        }), 500

@app.route('/api/auth/refresh', methods=['POST'])
@jwt_required(refresh=True)
def refresh():
    try:
        logger.info("=== TOKEN REFRESH REQUEST ===")
        
        current_user_id = get_jwt_identity()
        logger.info(f"Refreshing token for user ID: {current_user_id}")
        
        try:
            user_id_int = int(current_user_id)
        except (ValueError, TypeError):
            logger.error(f"Invalid user ID format: {current_user_id}")
            return jsonify({
                'error': 'Invalid token format',
                'code': 'INVALID_TOKEN_FORMAT'
            }), 401
        
        user = User.query.get(user_id_int)
        if not user:
            logger.warning(f"Token refresh denied - user not found: {user_id_int}")
            return jsonify({
                'error': 'User not found',
                'code': 'USER_NOT_FOUND'
            }), 404
        
        if not user.is_active:
            logger.warning(f"Token refresh denied - user inactive: {user_id_int}")
            return jsonify({
                'error': 'Account has been deactivated',
                'code': 'ACCOUNT_DEACTIVATED'
            }), 403
        
        new_access_token = create_access_token(
            identity=str(user.id),
            expires_delta=app.config['JWT_ACCESS_TOKEN_EXPIRES']
        )
        
        logger.info(f"✅ Token refreshed successfully for user: {user.email}")
        
        return jsonify({
            'access_token': new_access_token,
            'message': 'Token refreshed successfully'
        }), 200
        
    except Exception as e:
        logger.error(f"❌ Token refresh error: {str(e)}")
        logger.error(f"❌ Traceback: {traceback.format_exc()}")
        return jsonify({
            'error': 'Token refresh failed',
            'message': 'Please login again.',
            'code': 'TOKEN_REFRESH_ERROR'
        }), 500

@app.route('/api/auth/profile', methods=['GET'])
@jwt_required()
def get_profile():
    try:
        logger.info("=== GET PROFILE REQUEST ===")
        
        current_user_id = get_jwt_identity()
        logger.info(f"Profile request for user ID: {current_user_id}")
        
        try:
            user_id_int = int(current_user_id)
        except (ValueError, TypeError):
            logger.error(f"Invalid user ID format: {current_user_id}")
            return jsonify({
                'error': 'Invalid token format',
                'code': 'INVALID_TOKEN_FORMAT'
            }), 401
        
        user = User.query.get(user_id_int)
        
        if not user:
            logger.warning(f"Profile not found for user ID: {user_id_int}")
            return jsonify({
                'error': 'User not found',
                'code': 'USER_NOT_FOUND'
            }), 404
        
        if not user.is_active:
            logger.warning(f"Profile request for inactive user: {user_id_int}")
            return jsonify({
                'error': 'Account has been deactivated',
                'code': 'ACCOUNT_DEACTIVATED'
            }), 403
        
        user_data = {
            'id': user.id,
            'email': user.email,
            'firstName': user.first_name,
            'lastName': user.last_name,
            'phone': user.phone if user.phone else '',
            'isVerified': user.is_verified,
            'createdAt': user.created_at.isoformat(),
            'lastLogin': user.last_login.isoformat() if user.last_login else None
        }
        
        logger.info(f"✅ Profile retrieved for user: {user.email}")
        
        return jsonify(user_data), 200
        
    except Exception as e:
        logger.error(f"❌ Get profile error: {str(e)}")
        logger.error(f"❌ Traceback: {traceback.format_exc()}")
        return jsonify({
            'error': 'Failed to retrieve profile',
            'code': 'PROFILE_ERROR'
        }), 500

@app.route('/api/auth/profile', methods=['PUT'])
@jwt_required()
def update_profile():
    try:
        logger.info("=== UPDATE PROFILE REQUEST ===")
        
        current_user_id = get_jwt_identity()
        
        try:
            user_id_int = int(current_user_id)
        except (ValueError, TypeError):
            return jsonify({
                'error': 'Invalid token format',
                'code': 'INVALID_TOKEN_FORMAT'
            }), 401
        
        user = User.query.get(user_id_int)
        
        if not user:
            logger.warning(f"Profile update failed - user not found: {user_id_int}")
            return jsonify({
                'error': 'User not found',
                'code': 'USER_NOT_FOUND'
            }), 404
        
        if not user.is_active:
            logger.warning(f"Profile update for inactive user: {user_id_int}")
            return jsonify({
                'error': 'Account has been deactivated',
                'code': 'ACCOUNT_DEACTIVATED'
            }), 403
        
        if not request.is_json:
            return jsonify({
                'error': 'Content-Type must be application/json',
                'code': 'INVALID_CONTENT_TYPE'
            }), 400
        
        data = request.get_json()
        if not data:
            return jsonify({
                'error': 'Request body is empty',
                'code': 'EMPTY_REQUEST_BODY'
            }), 400
        
        logger.info(f"Profile update for user: {user.email}")
        
        # Update user fields
        updated_fields = []
        
        if 'firstName' in data and data['firstName']:
            user.first_name = str(data['firstName']).strip()
            updated_fields.append('firstName')
            
        if 'lastName' in data and data['lastName']:
            user.last_name = str(data['lastName']).strip()
            updated_fields.append('lastName')
            
        if 'phone' in data:
            user.phone = str(data['phone']).strip() if data['phone'] else ''
            updated_fields.append('phone')
        
        if not updated_fields:
            return jsonify({
                'error': 'No valid fields to update',
                'code': 'NO_FIELDS_TO_UPDATE'
            }), 400
        
        db.session.commit()
        
        user_data = {
            'id': user.id,
            'email': user.email,
            'firstName': user.first_name,
            'lastName': user.last_name,
            'phone': user.phone if user.phone else ''
        }
        
        logger.info(f"✅ Profile updated successfully for user: {user.email}")
        logger.info(f"Updated fields: {updated_fields}")
        
        return jsonify({
            'message': f'Profile updated successfully. Updated: {", ".join(updated_fields)}',
            'user': user_data
        }), 200
        
    except Exception as e:
        db.session.rollback()
        logger.error(f"❌ Update profile error: {str(e)}")
        logger.error(f"❌ Traceback: {traceback.format_exc()}")
        return jsonify({
            'error': 'Failed to update profile',
            'code': 'PROFILE_UPDATE_ERROR'
        }), 500

@app.route('/api/auth/logout', methods=['POST'])
@jwt_required()
def logout():
    try:
        logger.info("=== LOGOUT REQUEST ===")
        
        current_user_id = get_jwt_identity()
        jti = get_jwt()['jti']
        
        try:
            user_id_int = int(current_user_id)
            user = User.query.get(user_id_int)
        except (ValueError, TypeError):
            user = None
        
        if user:
            logger.info(f"Logout for user: {user.email}")
        
        blacklisted_tokens.add(jti)
        
        logger.info("✅ Logout successful")
        
        return jsonify({
            'message': 'Logged out successfully'
        }), 200
        
    except Exception as e:
        logger.error(f"❌ Logout error: {str(e)}")
        logger.error(f"❌ Traceback: {traceback.format_exc()}")
        return jsonify({
            'error': 'Logout failed',
            'code': 'LOGOUT_ERROR'
        }), 500

@app.route('/api/auth/test', methods=['GET'])
@jwt_required()
def test_auth():
    try:
        current_user_id = get_jwt_identity()
        
        try:
            user_id_int = int(current_user_id)
            user = User.query.get(user_id_int)
        except (ValueError, TypeError):
            user = None
        
        return jsonify({
            'message': 'Authentication test successful',
            'user_id': current_user_id,
            'user_email': user.email if user else 'Unknown',
            'timestamp': datetime.now().isoformat(),
            'token_claims': get_jwt()
        }), 200
        
    except Exception as e:
        logger.error(f"❌ Auth test error: {str(e)}")
        return jsonify({
            'error': 'Authentication test failed',
            'code': 'AUTH_TEST_ERROR'
        }), 500

@app.route('/api/auth/debug', methods=['GET'])
def debug_auth():
    try:
        # Test database connection
        db_connected = test_database_connection()
        
        # Get user count
        user_count = 0
        users_info = []
        
        if db_connected:
            try:
                user_count = User.query.count()
                users = User.query.limit(5).all()
                users_info = [
                    {
                        'id': u.id,
                        'email': u.email,
                        'is_active': u.is_active,
                        'is_verified': u.is_verified,
                        'created_at': u.created_at.isoformat()
                    } for u in users
                ]
            except Exception as e:
                logger.error(f"Error getting user info: {e}")
        
        return jsonify({
            'status': 'OK',
            'timestamp': datetime.now().isoformat(),
            'database_connected': db_connected,
            'user_count': user_count,
            'sample_users': users_info,
            'config': {
                'database_url': app.config.get('SQLALCHEMY_DATABASE_URI', 'Not set')[:50] + '...',
                'jwt_secret_set': bool(app.config.get('JWT_SECRET_KEY')),
                'debug_mode': app.config.get('DEBUG', False)
            }
        }), 200
        
    except Exception as e:
        logger.error(f"❌ Debug endpoint error: {str(e)}")
        return jsonify({
            'error': 'Debug endpoint failed',
            'message': str(e)
        }), 500

# ==================== OCR ROUTES ====================

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

# ==================== MAIN APPLICATION ====================

if __name__ == '__main__':
    # Create database tables if they don't exist
    with app.app_context():
        try:
            db.create_all()
            logger.info("✅ Database tables created/verified")
            
            # Check if default admin user exists
            admin_exists = User.query.filter_by(email='admin@spendly.com').first()
            if not admin_exists:
                logger.info("Creating default admin user...")
                admin_user = User(
                    email='admin@spendly.com',
                    first_name='Admin',
                    last_name='User',
                    password='Admin123!',
                    is_active=True,
                    is_verified=True
                )
                db.session.add(admin_user)
                db.session.commit()
                logger.info("✅ Default admin user created")
            else:
                logger.info(f"✅ Admin user exists with ID: {admin_exists.id}")
                
        except Exception as e:
            logger.error(f"❌ Database setup error: {str(e)}")
    
    # Get port from environment variable
    port = int(os.environ.get('PORT', 5000))
    
    logger.info(f"🚀 Starting Combined Auth & OCR server on port {port}...")
    logger.info(f"🔑 JWT Access Token Expiry: {app.config['JWT_ACCESS_TOKEN_EXPIRES']}")
    logger.info(f"🔄 JWT Refresh Token Expiry: {app.config['JWT_REFRESH_TOKEN_EXPIRES']}")
    logger.info(f"🌍 CORS Enabled for: http://localhost:3000")
    logger.info(f"�� OCR Endpoint: http://localhost:{port}/api/scan-receipt")
    logger.info(f"🔐 Auth Endpoints: http://localhost:{port}/api/auth/*")
    logger.info(f"🐛 Debug endpoint available at: http://localhost:{port}/api/auth/debug")
    
    app.run(debug=True, host='0.0.0.0', port=port)
