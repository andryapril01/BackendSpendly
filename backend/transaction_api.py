# transaction_api.py - Fixed Database Issue

from flask import Blueprint, request, jsonify
from flask_jwt_extended import jwt_required, get_jwt_identity
from models import db, Transaction, TransactionItem, Category, User
from sqlalchemy import text  # Import text for SQL expressions
from datetime import datetime
import logging
import traceback
import re

transaction_bp = Blueprint('transaction', __name__)

# Configure logging
logger = logging.getLogger(__name__)

# Helper functions
def validate_date_format(date_string):
    """Validate date format YYYY-MM-DD"""
    try:
        datetime.strptime(date_string, '%Y-%m-%d')
        return True
    except ValueError:
        return False

def validate_transaction_data(data):
    """Validate transaction data with detailed error messages"""
    errors = []
    
    # Required fields validation
    if not data.get('merchantName') or not str(data['merchantName']).strip():
        errors.append('Merchant name is required and cannot be empty')
    
    if not data.get('total') or not isinstance(data['total'], (int, float)) or data['total'] <= 0:
        errors.append('Total amount must be a positive number')
    
    if not data.get('date') or not validate_date_format(str(data['date'])):
        errors.append('Date must be in YYYY-MM-DD format')
    
    # Items validation
    items = data.get('items', [])
    if not items or not isinstance(items, list):
        errors.append('At least one item is required')
    else:
        for i, item in enumerate(items):
            if not isinstance(item, dict):
                errors.append(f'Item {i+1} must be an object')
                continue
                
            if not item.get('name') or not str(item['name']).strip():
                errors.append(f'Item {i+1} name is required')
            
            if not isinstance(item.get('quantity'), (int, float)) or item.get('quantity', 0) <= 0:
                errors.append(f'Item {i+1} quantity must be a positive number')
            
            if not isinstance(item.get('price'), (int, float)) or item.get('price', 0) < 0:
                errors.append(f'Item {i+1} price must be a non-negative number')
    
    return errors

@transaction_bp.route('/api/transactions', methods=['POST'])
@jwt_required()
def create_transaction():
    try:
        current_user_id_str = get_jwt_identity()
        
        # Convert user ID to integer
        try:
            current_user_id = int(current_user_id_str)
        except (ValueError, TypeError):
            logger.error(f"❌ Invalid user ID format: {current_user_id_str}")
            return jsonify({
                'success': False,
                'error': 'Invalid user ID format',
                'code': 'INVALID_USER_ID'
            }), 400
        
        data = request.get_json()
        
        logger.info(f"=== CREATING TRANSACTION ===")
        logger.info(f"User ID: {current_user_id}")
        logger.info(f"Request data: {data}")
        
        # Validate user exists and is active
        user = User.query.get(current_user_id)
        if not user:
            logger.error(f"❌ User not found: {current_user_id}")
            return jsonify({
                'success': False,
                'error': 'User not found',
                'code': 'USER_NOT_FOUND'
            }), 404
        
        if not user.is_active:
            logger.error(f"❌ User not active: {current_user_id}")
            return jsonify({
                'success': False,
                'error': 'User account is not active',
                'code': 'USER_INACTIVE'
            }), 403
        
        logger.info(f"✅ User found: {user.email}")
        
        # Validate JSON data exists
        if not data:
            logger.error("❌ No JSON data received")
            return jsonify({
                'success': False,
                'error': 'No data provided',
                'code': 'NO_DATA'
            }), 400
        
        # Validate transaction data
        validation_errors = validate_transaction_data(data)
        if validation_errors:
            logger.error(f"❌ Validation errors: {validation_errors}")
            return jsonify({
                'success': False,
                'error': 'Validation failed',
                'details': validation_errors,
                'code': 'VALIDATION_ERROR'
            }), 422
        
        # Parse and clean data
        try:
            merchant_name = str(data['merchantName']).strip()
            date_str = str(data['date'])
            transaction_date = datetime.strptime(date_str, '%Y-%m-%d')
            amount = float(data['total'])
            category_name = str(data.get('category', 'Lain-lain')).strip()
            description = str(data.get('description', '')).strip()
            is_scanned = bool(data.get('isScanned', False))
            trx_type = str(data.get('type', 'expense')).strip().lower()  # income/expense
            if trx_type not in ['income', 'expense']:
                trx_type = 'expense'
        except (ValueError, TypeError) as ve:
            logger.error(f"❌ Data parsing error: {ve}")
            return jsonify({
                'success': False,
                'error': 'Invalid data format',
                'details': str(ve),
                'code': 'PARSE_ERROR'
            }), 422
        
        logger.info(f"✅ Data validated: {merchant_name}, {amount}, {transaction_date}, {trx_type}")
        
        # Find or create category
        category = None
        if category_name:
            logger.info(f"Looking for category: {category_name}")
            
            category = Category.query.filter_by(
                user_id=current_user_id,
                name=category_name
            ).first()
            
            if not category:
                logger.info(f"Creating new category: {category_name}")
                category = Category(
                    user_id=current_user_id,
                    name=category_name,
                    icon='shopping-bag',
                    color='#6366f1',
                    budget=0,
                    is_default=False
                )
                db.session.add(category)
                db.session.flush()
                logger.info(f"✅ Category created with ID: {category.id}")
            else:
                logger.info(f"✅ Category found: {category.name} (ID: {category.id})")
        
        # Create main transaction
        transaction = Transaction(
            user_id=current_user_id,
            category_id=category.id if category else None,
            merchant_name=merchant_name,
            amount=amount,
            transaction_date=transaction_date,
            description=description or f"Payment via {data.get('paymentMethod', 'Unknown')}",
            is_scanned=is_scanned,
            type=trx_type
        )
        
        db.session.add(transaction)
        db.session.flush()
        logger.info(f"✅ Transaction created with ID: {transaction.id}")
        
        # Add transaction items
        items_added = 0
        if data.get('items') and isinstance(data['items'], list):
            for i, item_data in enumerate(data['items']):
                if not isinstance(item_data, dict):
                    logger.warning(f"⚠️ Skipping invalid item {i}: not a dict")
                    continue
                    
                item_name = str(item_data.get('name', '')).strip()
                if not item_name:
                    logger.warning(f"⚠️ Skipping item {i}: no name")
                    continue
                
                try:
                    quantity = int(item_data.get('quantity', 1))
                    price = float(item_data.get('price', 0))
                    
                    if quantity <= 0:
                        quantity = 1
                    if price < 0:
                        price = 0
                    
                    item = TransactionItem(
                        transaction_id=transaction.id,
                        name=item_name,
                        quantity=quantity,
                        price=price
                    )
                    db.session.add(item)
                    items_added += 1
                    logger.info(f"✅ Item added: {item_name} x{quantity} @{price}")
                    
                except (ValueError, TypeError) as e:
                    logger.warning(f"⚠️ Skipping invalid item {i}: {item_data} - {e}")
                    continue
        
        # If no items were added, create a default item
        if items_added == 0:
            default_item = TransactionItem(
                transaction_id=transaction.id,
                name=f"Purchase at {merchant_name}",
                quantity=1,
                price=amount
            )
            db.session.add(default_item)
            items_added = 1
            logger.info(f"✅ Default item created")
        
        # Commit all changes
        db.session.commit()
        logger.info(f"✅ Transaction committed successfully!")
        
        # Return success response
        response_data = {
            'success': True,
            'message': 'Transaction created successfully',
            'transaction_id': transaction.id,
            'data': {
                'id': transaction.id,
                'merchant_name': transaction.merchant_name,
                'amount': transaction.amount,
                'date': transaction.transaction_date.strftime('%Y-%m-%d'),
                'category': category.name if category else None,
                'items_count': items_added,
                'user_id': current_user_id,
                'created_at': transaction.created_at.isoformat()
            }
        }
        
        logger.info(f"✅ Returning response: {response_data}")
        return jsonify(response_data), 201
        
    except Exception as e:
        db.session.rollback()
        error_msg = f"Error creating transaction: {str(e)}"
        logger.error(f"❌ {error_msg}")
        logger.error(f"❌ Traceback: {traceback.format_exc()}")
        return jsonify({
            'success': False,
            'error': 'Failed to create transaction',
            'message': 'An unexpected error occurred while creating the transaction',
            'details': str(e),
            'code': 'CREATION_ERROR'
        }), 500

# Health check for transaction API
@transaction_bp.route('/api/transactions/health', methods=['GET'])
def transaction_health():
    try:
        # FIXED: Test database connection with proper text() wrapper
        db.session.execute(text('SELECT 1'))
        
        return jsonify({
            'status': 'OK',
            'message': 'Transaction API is running',
            'timestamp': datetime.now().isoformat(),
            'version': '2.0.1',
            'database': 'connected'
        }), 200
    except Exception as e:
        logger.error(f"❌ Transaction API health check failed: {str(e)}")
        return jsonify({
            'status': 'ERROR',
            'message': 'Transaction API has issues',
            'error': str(e)
        }), 500

# Debug endpoint for checking user and transactions
@transaction_bp.route('/api/transactions/debug', methods=['GET'])
@jwt_required()
def debug_transactions():
    try:
        current_user_id_str = get_jwt_identity()
        
        # Convert user ID to integer
        try:
            current_user_id = int(current_user_id_str)
        except (ValueError, TypeError):
            logger.error(f"❌ Invalid user ID format: {current_user_id_str}")
            return jsonify({
                'error': 'Invalid user ID format',
                'code': 'INVALID_USER_ID'
            }), 400
        
        logger.info(f"=== DEBUG TRANSACTIONS FOR USER {current_user_id} ===")
        
        # Get user info
        user = User.query.get(current_user_id)
        if not user:
            return jsonify({
                'error': 'User not found',
                'code': 'USER_NOT_FOUND'
            }), 404
        
        # Get transactions
        transactions = Transaction.query.filter_by(user_id=current_user_id).order_by(
            Transaction.transaction_date.desc()
        ).limit(10).all()
        
        # Get categories
        categories = Category.query.filter_by(user_id=current_user_id).all()
        
        # Format transactions
        transaction_list = []
        for transaction in transactions:
            transaction_list.append({
                'id': transaction.id,
                'merchant': transaction.merchant_name,
                'amount': float(transaction.amount),
                'date': transaction.transaction_date.strftime('%Y-%m-%d'),
                'category': transaction.category.name if transaction.category else 'Uncategorized',
                'created_at': transaction.created_at.isoformat()
            })
        
        # Format categories
        category_list = []
        for category in categories:
            category_list.append({
                'id': category.id,
                'name': category.name,
                'budget': float(category.budget),
                'is_default': category.is_default
            })
        
        debug_data = {
            'user': {
                'id': user.id,
                'email': user.email,
                'firstName': user.first_name,
                'lastName': user.last_name,
                'is_active': user.is_active
            },
            'transactions': transaction_list,
            'transactions_count': len(transaction_list),
            'categories': category_list,
            'categories_count': len(category_list),
            'jwt_user_id': current_user_id_str,
            'converted_user_id': current_user_id
        }
        
        logger.info(f"✅ Debug data retrieved for user: {user.email}")
        return jsonify(debug_data), 200
        
    except Exception as e:
        logger.error(f"❌ Debug endpoint error: {str(e)}")
        logger.error(f"❌ Traceback: {traceback.format_exc()}")
        return jsonify({
            'error': 'Debug endpoint failed',
            'details': str(e),
            'code': 'DEBUG_ERROR'
        }), 500

@transaction_bp.route('/api/transactions', methods=['GET'])
@jwt_required()
def get_transactions():
    try:
        current_user_id_str = get_jwt_identity()
        try:
            current_user_id = int(current_user_id_str)
        except (ValueError, TypeError):
            logger.error(f"❌ Invalid user ID format: {current_user_id_str}")
            return jsonify({
                'success': False,
                'error': 'Invalid user ID format',
                'code': 'INVALID_USER_ID'
            }), 400

        logger.info(f"=== GET TRANSACTIONS FOR USER {current_user_id} ===")
        user = User.query.get(current_user_id)
        if not user:
            return jsonify({
                'success': False,
                'error': 'User not found',
                'code': 'USER_NOT_FOUND'
            }), 404

        transactions = Transaction.query.filter_by(user_id=current_user_id).order_by(Transaction.transaction_date.desc()).all()
        transaction_list = [trx.to_dict(include_items=True) for trx in transactions]

        return jsonify({
            'success': True,
            'transactions': transaction_list,
            'count': len(transaction_list)
        }), 200
    except Exception as e:
        logger.error(f"❌ Error fetching transactions: {str(e)}")
        logger.error(f"❌ Traceback: {traceback.format_exc()}")
        return jsonify({
            'success': False,
            'error': 'Failed to fetch transactions',
            'details': str(e),
            'code': 'FETCH_ERROR'
        }), 500

# Endpoint: Edit transaction
@transaction_bp.route('/api/transactions/<int:transaction_id>', methods=['PATCH'])
@jwt_required()
def update_transaction(transaction_id):
    try:
        current_user_id_str = get_jwt_identity()
        current_user_id = int(current_user_id_str)
        user = User.query.get(current_user_id)
        if not user:
            return jsonify({'success': False, 'error': 'User not found', 'code': 'USER_NOT_FOUND'}), 404
        trx = Transaction.query.filter_by(id=transaction_id, user_id=user.id).first()
        if not trx:
            return jsonify({'success': False, 'error': 'Transaction not found', 'code': 'TRANSACTION_NOT_FOUND'}), 404
        data = request.get_json() or {}
        # Update only provided fields
        if 'merchantName' in data:
            trx.merchant_name = data['merchantName']
        if 'amount' in data:
            trx.amount = float(data['amount'])
        if 'date' in data:
            trx.transaction_date = datetime.strptime(data['date'], '%Y-%m-%d')
        if 'description' in data:
            trx.description = data['description']
        if 'category' in data:
            # Find or create category
            category = Category.query.filter_by(user_id=user.id, name=data['category']).first()
            if not category:
                category = Category(user_id=user.id, name=data['category'], icon='shopping-bag', color='#6366f1', budget=0, is_default=False)
                db.session.add(category)
                db.session.flush()
            trx.category_id = category.id
        if 'type' in data:
            trx.type = data['type']
        db.session.commit()
        return jsonify({'success': True, 'message': 'Transaction updated'}), 200
    except Exception as e:
        db.session.rollback()
        return jsonify({'success': False, 'error': str(e), 'code': 'UPDATE_ERROR'}), 500

# Endpoint: Delete transaction
@transaction_bp.route('/api/transactions/<int:transaction_id>', methods=['DELETE'])
@jwt_required()
def delete_transaction(transaction_id):
    try:
        current_user_id_str = get_jwt_identity()
        current_user_id = int(current_user_id_str)
        user = User.query.get(current_user_id)
        if not user:
            return jsonify({'success': False, 'error': 'User not found', 'code': 'USER_NOT_FOUND'}), 404
        trx = Transaction.query.filter_by(id=transaction_id, user_id=user.id).first()
        if not trx:
            return jsonify({'success': False, 'error': 'Transaction not found', 'code': 'TRANSACTION_NOT_FOUND'}), 404
        db.session.delete(trx)
        db.session.commit()
        return jsonify({'success': True, 'message': 'Transaction deleted'}), 200
    except Exception as e:
        db.session.rollback()
        return jsonify({'success': False, 'error': str(e), 'code': 'DELETE_ERROR'}), 500