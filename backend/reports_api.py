from flask import Blueprint, jsonify, request
from flask_jwt_extended import jwt_required, get_jwt_identity
from models import db, User, Transaction, Category
from datetime import datetime, timedelta
from sqlalchemy import func, extract
import calendar

reports_bp = Blueprint('reports', __name__)

@reports_bp.route('/api/reports', methods=['GET'])
@jwt_required()
def get_reports():
    try:
        user_id = get_jwt_identity()
        period = request.args.get('period', 'this_month')
        
        # Get date range based on period
        now = datetime.now()
        if period == 'this_month':
            start_date = now.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
            end_date = now.replace(hour=23, minute=59, second=59, microsecond=999999)
        elif period == 'last_month':
            last_month = now.replace(day=1) - timedelta(days=1)
            start_date = last_month.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
            end_date = now.replace(day=1, hour=0, minute=0, second=0, microsecond=0) - timedelta(microseconds=1)
        elif period == 'this_year':
            start_date = now.replace(month=1, day=1, hour=0, minute=0, second=0, microsecond=0)
            end_date = now.replace(hour=23, minute=59, second=59, microsecond=999999)
        elif period == 'last_year':
            start_date = now.replace(year=now.year-1, month=1, day=1, hour=0, minute=0, second=0, microsecond=0)
            end_date = now.replace(year=now.year-1, month=12, day=31, hour=23, minute=59, second=59, microsecond=999999)
        else:
            return jsonify({'error': 'Invalid period'}), 400

        # Get total income and expense
        income_query = Transaction.query.filter(
            Transaction.user_id == user_id,
            Transaction.type == 'income',
            Transaction.transaction_date >= start_date,
            Transaction.transaction_date <= end_date
        )
        total_income = income_query.with_entities(func.sum(Transaction.amount)).scalar() or 0

        expense_query = Transaction.query.filter(
            Transaction.user_id == user_id,
            Transaction.type == 'expense',
            Transaction.transaction_date >= start_date,
            Transaction.transaction_date <= end_date
        )
        total_expense = expense_query.with_entities(func.sum(Transaction.amount)).scalar() or 0

        balance = total_income - total_expense

        # Get top categories by expense
        top_categories = db.session.query(
            Category.name,
            Category.icon,
            Category.color,
            func.sum(Transaction.amount).label('total')
        ).join(Transaction).filter(
            Transaction.user_id == user_id,
            Transaction.type == 'expense',
            Transaction.transaction_date >= start_date,
            Transaction.transaction_date <= end_date
        ).group_by(Category.id, Category.name, Category.icon, Category.color).order_by(
            func.sum(Transaction.amount).desc()
        ).limit(5).all()

        # Calculate percentages for top categories
        total_expense_for_percentage = total_expense if total_expense > 0 else 1
        top_categories_data = []
        for cat in top_categories:
            percentage = (cat.total / total_expense_for_percentage) * 100
            top_categories_data.append({
                'id': cat.name,  # Using name as ID for simplicity
                'name': cat.name,
                'icon': cat.icon or '📊',
                'color': cat.color or '#6366f1',
                'total': float(cat.total),
                'percentage': percentage
            })

        # Get monthly data for the last 6 months
        monthly_data = []
        for i in range(6):
            if period in ['this_year', 'last_year']:
                # For yearly reports, show monthly data
                month_date = start_date + timedelta(days=32*i)
                month_start = month_date.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
                if month_start > end_date:
                    break
                month_end = (month_start.replace(day=28) + timedelta(days=4)).replace(day=1) - timedelta(microseconds=1)
                if month_end > end_date:
                    month_end = end_date
            else:
                # For monthly reports, show weekly data
                week_start = start_date + timedelta(weeks=i)
                week_end = week_start + timedelta(days=6, hours=23, minutes=59, seconds=59, microseconds=999999)
                if week_start > end_date:
                    break
                if week_end > end_date:
                    week_end = end_date
                month_start = week_start
                month_end = week_end

            month_income = Transaction.query.filter(
                Transaction.user_id == user_id,
                Transaction.type == 'income',
                Transaction.transaction_date >= month_start,
                Transaction.transaction_date <= month_end
            ).with_entities(func.sum(Transaction.amount)).scalar() or 0

            month_expense = Transaction.query.filter(
                Transaction.user_id == user_id,
                Transaction.type == 'expense',
                Transaction.transaction_date >= month_start,
                Transaction.transaction_date <= month_end
            ).with_entities(func.sum(Transaction.amount)).scalar() or 0

            if period in ['this_year', 'last_year']:
                month_label = month_start.strftime('%b')
            else:
                month_label = f"Week {i+1}"

            monthly_data.append({
                'month': month_label,
                'income': float(month_income),
                'expense': float(month_expense)
            })

        return jsonify({
            'totalIncome': float(total_income),
            'totalExpense': float(total_expense),
            'balance': float(balance),
            'topCategories': top_categories_data,
            'monthlyData': monthly_data,
            'period': period,
            'startDate': start_date.isoformat(),
            'endDate': end_date.isoformat()
        }), 200

    except Exception as e:
        print(f"Error in reports API: {str(e)}")
        return jsonify({'error': 'Internal server error'}), 500

@reports_bp.route('/api/reports/categories', methods=['GET'])
@jwt_required()
def get_category_reports():
    try:
        user_id = get_jwt_identity()
        period = request.args.get('period', 'this_month')
        
        # Get date range
        now = datetime.now()
        if period == 'this_month':
            start_date = now.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
            end_date = now.replace(hour=23, minute=59, second=59, microsecond=999999)
        elif period == 'last_month':
            last_month = now.replace(day=1) - timedelta(days=1)
            start_date = last_month.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
            end_date = now.replace(day=1, hour=0, minute=0, second=0, microsecond=0) - timedelta(microseconds=1)
        else:
            return jsonify({'error': 'Invalid period'}), 400

        # Get all categories with their totals
        categories = db.session.query(
            Category.id,
            Category.name,
            Category.icon,
            Category.color,
            func.sum(Transaction.amount).label('total')
        ).outerjoin(Transaction).filter(
            Category.user_id == user_id,
            (Transaction.user_id == user_id) | (Transaction.user_id.is_(None)),
            (Transaction.type == 'expense') | (Transaction.type.is_(None)),
            (Transaction.transaction_date >= start_date) | (Transaction.transaction_date.is_(None)),
            (Transaction.transaction_date <= end_date) | (Transaction.transaction_date.is_(None))
        ).group_by(Category.id, Category.name, Category.icon, Category.color).all()

        total_expense = sum(cat.total or 0 for cat in categories)
        total_expense_for_percentage = total_expense if total_expense > 0 else 1

        categories_data = []
        for cat in categories:
            total = cat.total or 0
            percentage = (total / total_expense_for_percentage) * 100 if total > 0 else 0
            categories_data.append({
                'id': cat.id,
                'name': cat.name,
                'icon': cat.icon or '📊',
                'color': cat.color or '#6366f1',
                'total': float(total),
                'percentage': percentage
            })

        return jsonify({
            'categories': categories_data,
            'totalExpense': float(total_expense)
        }), 200

    except Exception as e:
        print(f"Error in category reports API: {str(e)}")
        return jsonify({'error': 'Internal server error'}), 500 