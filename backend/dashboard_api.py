# dashboard_api.py - FIXED Dashboard API

from flask import Blueprint, request, jsonify
from flask_jwt_extended import jwt_required, get_jwt_identity
from models import db, Transaction, Category, User
from sqlalchemy import text, func
from datetime import datetime, timedelta
import logging
import traceback

dashboard_bp = Blueprint('dashboard', __name__)

# Configure logging
logger = logging.getLogger(__name__)

# Helper function to get current user
def get_current_user():
    """Get current user from JWT token"""
    try:
        current_user_id_str = get_jwt_identity()
        current_user_id = int(current_user_id_str)
        user = User.query.get(current_user_id)
        
        if not user or not user.is_active:
            return None
            
        return user
    except (ValueError, TypeError) as e:
        logger.error(f"Error getting current user: {e}")
        return None

# Dashboard summary endpoint
@dashboard_bp.route('/api/dashboard/summary', methods=['GET'])
@jwt_required()
def get_dashboard_summary():
    try:
        user = get_current_user()
        if not user:
            return jsonify({
                'error': 'User not found or inactive',
                'code': 'USER_NOT_FOUND'
            }), 404
        
        logger.info(f"Getting dashboard summary for user: {user.email}")
        
        now = datetime.now()
        start_of_month = now.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
        
        # Hitung pemasukan dan pengeluaran bulan ini
        monthly_income_query = db.session.query(func.sum(Transaction.amount)).filter(
            Transaction.user_id == user.id,
            Transaction.transaction_date >= start_of_month,
            Transaction.type == 'income'
        )
        monthly_expenses_query = db.session.query(func.sum(Transaction.amount)).filter(
            Transaction.user_id == user.id,
            Transaction.transaction_date >= start_of_month,
            Transaction.type == 'expense'
        )
        monthly_income = float(monthly_income_query.scalar() or 0)
        monthly_expenses = float(monthly_expenses_query.scalar() or 0)
        balance = monthly_income - monthly_expenses
        
        summary = {
            'monthlyIncome': monthly_income,
            'monthlyExpenses': monthly_expenses,
            'balance': balance,
            'savingsGoal': 0,  # 5M IDR target
            'expenseChange': 0  # Change from last month
        }

        logger.info(f"✅ Dashboard summary calculated for user: {user.email}")
        return jsonify(summary), 200
        
    except Exception as e:
        logger.error(f"❌ Dashboard summary error: {str(e)}")
        logger.error(f"❌ Traceback: {traceback.format_exc()}")
        return jsonify({
            'error': 'Failed to get dashboard summary',
            'message': str(e),
            'code': 'DASHBOARD_SUMMARY_ERROR'
        }), 500

# Recent transactions endpoint
@dashboard_bp.route('/api/dashboard/recent-transactions', methods=['GET'])
@jwt_required()
def get_recent_transactions():
    try:
        user = get_current_user()
        if not user:
            return jsonify({
                'error': 'User not found or inactive',
                'code': 'USER_NOT_FOUND'
            }), 404
        
        logger.info(f"Getting recent transactions for user: {user.email}")
        
        # Get recent transactions
        recent_transactions = Transaction.query.filter_by(user_id=user.id)\
            .order_by(Transaction.transaction_date.desc())\
            .limit(10).all()
        
        transactions_data = []
        for transaction in recent_transactions:
            transactions_data.append({
                'id': transaction.id,
                'merchant': transaction.merchant_name,
                'merchantName': transaction.merchant_name,
                'amount': float(transaction.amount),
                'date': transaction.transaction_date.strftime('%Y-%m-%d'),
                'category': transaction.category.name if transaction.category else 'Uncategorized',
                'description': transaction.description
            })
        
        logger.info(f"✅ Retrieved {len(transactions_data)} recent transactions")
        return jsonify(transactions_data), 200
        
    except Exception as e:
        logger.error(f"❌ Recent transactions error: {str(e)}")
        logger.error(f"❌ Traceback: {traceback.format_exc()}")
        return jsonify({
            'error': 'Failed to get recent transactions',
            'message': str(e),
            'code': 'RECENT_TRANSACTIONS_ERROR'
        }), 500

# Expense distribution endpoint
@dashboard_bp.route('/api/dashboard/expense-distribution', methods=['GET'])
@jwt_required()
def get_expense_distribution():
    try:
        user = get_current_user()
        if not user:
            return jsonify({
                'error': 'User not found or inactive',
                'code': 'USER_NOT_FOUND'
            }), 404
        
        logger.info(f"Getting expense distribution for user: {user.email}")
        
        # Calculate date range (current month)
        now = datetime.now()
        start_of_month = now.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
        
        # Get expenses by category for current month
        expense_by_category = db.session.query(
            Category.name,
            Category.color,
            func.sum(Transaction.amount).label('total')
        ).join(Transaction, Category.id == Transaction.category_id)\
         .filter(Transaction.user_id == user.id)\
         .filter(Transaction.transaction_date >= start_of_month)\
         .group_by(Category.id, Category.name, Category.color)\
         .all()
        
        # Calculate total expenses
        total_expenses = sum([float(item.total) for item in expense_by_category])
        
        # Prepare distribution data
        distribution_data = []
        for item in expense_by_category:
            amount = float(item.total)
            percentage = (amount / total_expenses * 100) if total_expenses > 0 else 0
            
            distribution_data.append({
                'category': item.name,
                'amount': amount,
                'percentage': round(percentage, 1),
                'color': item.color
            })
        
        logger.info(f"✅ Retrieved expense distribution with {len(distribution_data)} categories")
        return jsonify(distribution_data), 200
        
    except Exception as e:
        logger.error(f"❌ Expense distribution error: {str(e)}")
        logger.error(f"❌ Traceback: {traceback.format_exc()}")
        return jsonify({
            'error': 'Failed to get expense distribution',
            'message': str(e),
            'code': 'EXPENSE_DISTRIBUTION_ERROR'
        }), 500

# Budget status endpoint
@dashboard_bp.route('/api/dashboard/budget-status', methods=['GET'])
@jwt_required()
def get_budget_status():
    try:
        user = get_current_user()
        if not user:
            return jsonify({
                'error': 'User not found or inactive',
                'code': 'USER_NOT_FOUND'
            }), 404
        
        logger.info(f"Getting budget status for user: {user.email}")
        
        # Get user's categories with budgets
        categories = Category.query.filter_by(user_id=user.id).all()
        
        # Calculate date range (current month)
        now = datetime.now()
        start_of_month = now.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
        
        budget_data = []
        for category in categories:
            if category.budget > 0:  # Only include categories with budgets
                # Get spent amount for this category this month
                spent_query = db.session.query(func.sum(Transaction.amount)).filter(
                    Transaction.user_id == user.id,
                    Transaction.category_id == category.id,
                    Transaction.transaction_date >= start_of_month
                )
                spent = float(spent_query.scalar() or 0)
                
                # Calculate percentage and status
                percentage = (spent / category.budget * 100) if category.budget > 0 else 0
                
                if percentage >= 100:
                    status = 'over'
                elif percentage >= 90:
                    status = 'warning'
                elif percentage >= 75:
                    status = 'caution'
                else:
                    status = 'normal'
                
                budget_data.append({
                    'category': category.name,
                    'spent': spent,
                    'budget': float(category.budget),
                    'percentage': round(percentage, 1),
                    'status': status,
                    'color': category.color
                })
        
        logger.info(f"✅ Retrieved budget status for {len(budget_data)} categories")
        return jsonify(budget_data), 200
        
    except Exception as e:
        logger.error(f"❌ Budget status error: {str(e)}")
        logger.error(f"❌ Traceback: {traceback.format_exc()}")
        return jsonify({
            'error': 'Failed to get budget status',
            'message': str(e),
            'code': 'BUDGET_STATUS_ERROR'
        }), 500

# Upcoming payments endpoint
@dashboard_bp.route('/api/dashboard/upcoming-payments', methods=['GET'])
@jwt_required()
def get_upcoming_payments():
    try:
        user = get_current_user()
        if not user:
            return jsonify({
                'error': 'User not found or inactive',
                'code': 'USER_NOT_FOUND'
            }), 404
        
        logger.info(f"Getting upcoming payments for user: {user.email}")
        
        # For now, return sample data
        # In the future, this could be connected to a recurring payments table
        upcoming_payments = [
            {
                'id': 1,
                'title': 'Listrik PLN',
                'amount': 500000,
                'dueDate': (datetime.now() + timedelta(days=5)).strftime('%Y-%m-%d'),
                'icon': 'zap'
            },
            {
                'id': 2,
                'title': 'Internet Indihome',
                'amount': 300000,
                'dueDate': (datetime.now() + timedelta(days=10)).strftime('%Y-%m-%d'),
                'icon': 'wifi'
            }
        ]
        
        logger.info(f"✅ Retrieved {len(upcoming_payments)} upcoming payments")
        return jsonify(upcoming_payments), 200
        
    except Exception as e:
        logger.error(f"❌ Upcoming payments error: {str(e)}")
        logger.error(f"❌ Traceback: {traceback.format_exc()}")
        return jsonify({
            'error': 'Failed to get upcoming payments',
            'message': str(e),
            'code': 'UPCOMING_PAYMENTS_ERROR'
        }), 500

# Dashboard health check
@dashboard_bp.route('/api/dashboard/health', methods=['GET'])
def dashboard_health():
    try:
        # Test database connection
        db.session.execute(text('SELECT 1'))
        
        return jsonify({
            'status': 'OK',
            'message': 'Dashboard API is running',
            'timestamp': datetime.now().isoformat(),
            'version': '2.0.1',
            'database': 'connected'
        }), 200
        
    except Exception as e:
        logger.error(f"❌ Dashboard API health check failed: {str(e)}")
        return jsonify({
            'status': 'ERROR',
            'message': 'Dashboard API has issues',
            'error': str(e)
        }), 500

# Add new budget endpoint
@dashboard_bp.route('/api/budgets', methods=['POST'])
@jwt_required()
def create_budget():
    try:
        user = get_current_user()
        if not user:
            return jsonify({'error': 'User not found or inactive', 'code': 'USER_NOT_FOUND'}), 404

        data = request.get_json()
        if not data:
            return jsonify({'error': 'No data provided', 'code': 'NO_DATA'}), 400

        # Required fields
        name = data.get('name')
        amount = data.get('amount')
        period = data.get('period', 'monthly')
        start_date = data.get('start_date')
        category_id = data.get('category_id')
        end_date = data.get('end_date')
        alert_threshold = data.get('alert_threshold', 80.0)

        # Validate required fields
        if not name or amount is None or not start_date:
            return jsonify({'error': 'Missing required fields', 'code': 'MISSING_FIELDS'}), 400

        try:
            amount = float(amount)
            alert_threshold = float(alert_threshold)
        except ValueError:
            return jsonify({'error': 'Invalid amount or alert_threshold', 'code': 'INVALID_AMOUNT'}), 400

        try:
            start_date_obj = datetime.strptime(start_date, '%Y-%m-%d')
            end_date_obj = None
            if end_date:
                end_date_obj = datetime.strptime(end_date, '%Y-%m-%d')
        except Exception as e:
            return jsonify({'error': 'Invalid date format', 'details': str(e), 'code': 'INVALID_DATE'}), 400

        # Validate category (optional)
        category = None
        if category_id:
            category = Category.query.filter_by(id=category_id, user_id=user.id).first()
            if not category:
                return jsonify({'error': 'Category not found', 'code': 'CATEGORY_NOT_FOUND'}), 404

        # Create budget
        from models import Budget
        budget = Budget(
            user_id=user.id,
            category_id=category.id if category else None,
            name=name,
            amount=amount,
            period=period,
            start_date=start_date_obj,
            end_date=end_date_obj,
            is_active=True,
            alert_threshold=alert_threshold
        )
        db.session.add(budget)
        db.session.commit()

        return jsonify({'success': True, 'message': 'Budget created successfully', 'budget_id': budget.id}), 201
    except Exception as e:
        db.session.rollback()
        logger.error(f"❌ Error creating budget: {str(e)}")
        logger.error(f"❌ Traceback: {traceback.format_exc()}")
        return jsonify({'error': 'Failed to create budget', 'details': str(e), 'code': 'CREATE_BUDGET_ERROR'}), 500

# Endpoint: Get all categories for current user
@dashboard_bp.route('/api/categories', methods=['GET'])
@jwt_required()
def get_categories():
    try:
        user = get_current_user()
        if not user:
            return jsonify({'error': 'User not found or inactive', 'code': 'USER_NOT_FOUND'}), 404
        categories = Category.query.filter_by(user_id=user.id).all()
        data = [
            {'id': c.id, 'name': c.name, 'icon': c.icon, 'color': c.color}
            for c in categories
        ]
        return jsonify({'categories': data}), 200
    except Exception as e:
        logger.error(f"❌ Error fetching categories: {str(e)}")
        logger.error(f"❌ Traceback: {traceback.format_exc()}")
        return jsonify({'error': 'Failed to fetch categories', 'details': str(e), 'code': 'FETCH_CATEGORIES_ERROR'}), 500

# Endpoint: Get all budgets for current user
@dashboard_bp.route('/api/budgets', methods=['GET'])
@jwt_required()
def get_budgets():
    try:
        user = get_current_user()
        if not user:
            return jsonify({'error': 'User not found or inactive', 'code': 'USER_NOT_FOUND'}), 404
        from models import Budget, Category, Transaction
        budgets = Budget.query.filter_by(user_id=user.id).order_by(Budget.start_date.desc()).all()
        data = []
        now = datetime.now()
        for b in budgets:
            # Tentukan range periode budget
            if b.period == 'weekly':
                # Mulai dari start_date, cari minggu berjalan
                start = b.start_date
                while start + timedelta(days=7) <= now:
                    start += timedelta(days=7)
                end = start + timedelta(days=6)
            elif b.period == 'monthly':
                start = b.start_date.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
                while (start.month < now.month or start.year < now.year):
                    # Naik ke bulan berikutnya
                    if start.month == 12:
                        start = start.replace(year=start.year+1, month=1)
                    else:
                        start = start.replace(month=start.month+1)
                end = (start.replace(day=28) + timedelta(days=4)).replace(day=1) - timedelta(days=1)
            elif b.period == 'yearly':
                start = b.start_date.replace(month=1, day=1, hour=0, minute=0, second=0, microsecond=0)
                while start.year < now.year:
                    start = start.replace(year=start.year+1)
                end = start.replace(month=12, day=31)
            else:
                # fallback: pakai start_date sampai end_date
                start = b.start_date
                end = b.end_date or now
            # Hitung total pengeluaran pada kategori & periode
            spent_query = Transaction.query.filter(
                Transaction.user_id == user.id,
                Transaction.category_id == b.category_id,
                Transaction.type == 'expense',
                Transaction.transaction_date >= start,
                Transaction.transaction_date <= end
            )
            spent = float(spent_query.with_entities(db.func.sum(Transaction.amount)).scalar() or 0)
            remaining = b.amount - spent
            percentage = (spent / b.amount * 100) if b.amount > 0 else 0
            if percentage >= 100:
                status = 'over'
            elif percentage >= 90:
                status = 'warning'
            elif percentage >= 75:
                status = 'caution'
            else:
                status = 'normal'
            category = None
            if b.category_id:
                cat = Category.query.get(b.category_id)
                if cat:
                    category = {'id': cat.id, 'name': cat.name, 'icon': cat.icon, 'color': cat.color}
            data.append({
                'id': b.id,
                'name': b.name,
                'amount': b.amount,
                'period': b.period,
                'start_date': b.start_date.strftime('%Y-%m-%d'),
                'end_date': b.end_date.strftime('%Y-%m-%d') if b.end_date else None,
                'is_active': b.is_active,
                'alert_threshold': b.alert_threshold,
                'category': category,
                'created_at': b.created_at.isoformat(),
                'updated_at': b.updated_at.isoformat(),
                'spent': spent,
                'remaining': remaining,
                'percentage': round(percentage, 1),
                'status': status
            })
        return jsonify({'budgets': data}), 200
    except Exception as e:
        logger.error(f"❌ Error fetching budgets: {str(e)}")
        logger.error(f"❌ Traceback: {traceback.format_exc()}")
        return jsonify({'error': 'Failed to fetch budgets', 'details': str(e), 'code': 'FETCH_BUDGETS_ERROR'}), 500

# Endpoint: Edit budget
@dashboard_bp.route('/api/budgets/<int:budget_id>', methods=['PATCH'])
@jwt_required()
def update_budget(budget_id):
    try:
        user = get_current_user()
        if not user:
            return jsonify({'error': 'User not found or inactive', 'code': 'USER_NOT_FOUND'}), 404
        from models import Budget, Category
        budget = Budget.query.filter_by(id=budget_id, user_id=user.id).first()
        if not budget:
            return jsonify({'error': 'Budget not found', 'code': 'BUDGET_NOT_FOUND'}), 404
        data = request.get_json() or {}
        # Update only provided fields
        if 'name' in data:
            budget.name = data['name']
        if 'amount' in data:
            try:
                budget.amount = float(data['amount'])
            except ValueError:
                return jsonify({'error': 'Invalid amount', 'code': 'INVALID_AMOUNT'}), 400
        if 'period' in data:
            budget.period = data['period']
        if 'start_date' in data:
            try:
                budget.start_date = datetime.strptime(data['start_date'], '%Y-%m-%d')
            except Exception as e:
                return jsonify({'error': 'Invalid start_date', 'details': str(e), 'code': 'INVALID_DATE'}), 400
        if 'end_date' in data:
            try:
                budget.end_date = datetime.strptime(data['end_date'], '%Y-%m-%d') if data['end_date'] else None
            except Exception as e:
                return jsonify({'error': 'Invalid end_date', 'details': str(e), 'code': 'INVALID_DATE'}), 400
        if 'alert_threshold' in data:
            try:
                budget.alert_threshold = float(data['alert_threshold'])
            except ValueError:
                return jsonify({'error': 'Invalid alert_threshold', 'code': 'INVALID_ALERT_THRESHOLD'}), 400
        if 'category_id' in data:
            if data['category_id']:
                category = Category.query.filter_by(id=data['category_id'], user_id=user.id).first()
                if not category:
                    return jsonify({'error': 'Category not found', 'code': 'CATEGORY_NOT_FOUND'}), 404
                budget.category_id = category.id
            else:
                budget.category_id = None
        db.session.commit()
        return jsonify({'success': True, 'message': 'Budget updated successfully'}), 200
    except Exception as e:
        db.session.rollback()
        logger.error(f"❌ Error updating budget: {str(e)}")
        logger.error(f"❌ Traceback: {traceback.format_exc()}")
        return jsonify({'error': 'Failed to update budget', 'details': str(e), 'code': 'UPDATE_BUDGET_ERROR'}), 500

# Endpoint: Delete budget
@dashboard_bp.route('/api/budgets/<int:budget_id>', methods=['DELETE'])
@jwt_required()
def delete_budget(budget_id):
    try:
        user = get_current_user()
        if not user:
            return jsonify({'error': 'User not found or inactive', 'code': 'USER_NOT_FOUND'}), 404
        from models import Budget
        budget = Budget.query.filter_by(id=budget_id, user_id=user.id).first()
        if not budget:
            return jsonify({'error': 'Budget not found', 'code': 'BUDGET_NOT_FOUND'}), 404
        db.session.delete(budget)
        db.session.commit()
        return jsonify({'success': True, 'message': 'Budget deleted successfully'}), 200
    except Exception as e:
        db.session.rollback()
        logger.error(f"❌ Error deleting budget: {str(e)}")
        logger.error(f"❌ Traceback: {traceback.format_exc()}")
        return jsonify({'error': 'Failed to delete budget', 'details': str(e), 'code': 'DELETE_BUDGET_ERROR'}), 500