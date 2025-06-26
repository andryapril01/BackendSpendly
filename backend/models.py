# models.py - Enhanced Database Models with better relationships
from flask_sqlalchemy import SQLAlchemy
from flask_bcrypt import Bcrypt
from datetime import datetime, timedelta, timezone
from sqlalchemy.ext.hybrid import hybrid_property
import uuid

db = SQLAlchemy()
bcrypt = Bcrypt()

class User(db.Model):
    __tablename__ = 'users'
    
    id = db.Column(db.Integer, primary_key=True)
    email = db.Column(db.String(120), unique=True, nullable=False, index=True)
    first_name = db.Column(db.String(100), nullable=False)
    last_name = db.Column(db.String(100), nullable=False)
    phone = db.Column(db.String(20), nullable=True)
    _password = db.Column('password', db.String(128), nullable=False)
    is_active = db.Column(db.Boolean, default=True)
    is_verified = db.Column(db.Boolean, default=False, nullable=False)
    last_login = db.Column(db.DateTime, nullable=True)
    created_at = db.Column(db.DateTime, default=lambda: datetime.now(timezone.utc), nullable=False)
    updated_at = db.Column(db.DateTime, default=lambda: datetime.now(timezone.utc), onupdate=lambda: datetime.now(timezone.utc), nullable=False)
    
    # Relationships
    categories = db.relationship('Category', backref='user', lazy=True, cascade='all, delete-orphan')
    transactions = db.relationship('Transaction', backref='user', lazy=True, cascade='all, delete-orphan')
    tokens = db.relationship('Token', backref='user', lazy=True, cascade='all, delete-orphan')
    
    @hybrid_property
    def password(self):
        return self._password
    
    @password.setter
    def password(self, plain_text_password):
        self._password = bcrypt.generate_password_hash(plain_text_password).decode('utf-8')
    
    def check_password(self, plain_text_password):
        return bcrypt.check_password_hash(self._password, plain_text_password)
    
    @property
    def full_name(self):
        return f"{self.first_name} {self.last_name}"
    
    def to_dict(self):
        return {
            'id': self.id,
            'email': self.email,
            'firstName': self.first_name,
            'lastName': self.last_name,
            'phone': self.phone,
            'isVerified': self.is_verified,
            'isActive': self.is_active,
            'lastLogin': self.last_login.isoformat() if self.last_login else None,
            'createdAt': self.created_at.isoformat(),
            'updatedAt': self.updated_at.isoformat()
        }
    
    def __repr__(self):
        return f'<User {self.email}>'


class Category(db.Model):
    __tablename__ = 'categories'
    
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=False, index=True)
    name = db.Column(db.String(50), nullable=False)
    icon = db.Column(db.String(50), nullable=True, default='shopping-bag')
    color = db.Column(db.String(20), nullable=True, default='#6366f1')
    is_default = db.Column(db.Boolean, default=False, nullable=False)
    budget = db.Column(db.Float, default=0.0, nullable=False)
    is_active = db.Column(db.Boolean, default=True, nullable=False)
    created_at = db.Column(db.DateTime, default=lambda: datetime.now(timezone.utc), nullable=False)
    updated_at = db.Column(db.DateTime, default=lambda: datetime.now(timezone.utc), onupdate=lambda: datetime.now(timezone.utc), nullable=False)
    
    # Relationships
    transactions = db.relationship('Transaction', backref='category', lazy=True)
    
    # Unique constraint: user can't have duplicate category names
    __table_args__ = (db.UniqueConstraint('user_id', 'name', name='unique_user_category'),)
    
    def get_spent_amount(self, start_date=None, end_date=None):
        """Get total spent amount for this category in date range"""
        query = Transaction.query.filter_by(category_id=self.id)
        
        if start_date:
            query = query.filter(Transaction.transaction_date >= start_date)
        if end_date:
            query = query.filter(Transaction.transaction_date <= end_date)
        
        total = db.session.query(db.func.sum(Transaction.amount)).filter(
            Transaction.category_id == self.id
        )
        if start_date:
            total = total.filter(Transaction.transaction_date >= start_date)
        if end_date:
            total = total.filter(Transaction.transaction_date <= end_date)
        
        return total.scalar() or 0.0
    
    def get_budget_status(self, start_date=None, end_date=None):
        """Get budget status for this category"""
        spent = self.get_spent_amount(start_date, end_date)
        percentage = (spent / self.budget * 100) if self.budget > 0 else 0
        
        if percentage >= 100:
            status = 'over'
        elif percentage >= 90:
            status = 'warning'
        elif percentage >= 75:
            status = 'caution'
        else:
            status = 'normal'
        
        return {
            'spent': spent,
            'budget': self.budget,
            'percentage': round(percentage, 1),
            'status': status,
            'remaining': max(0, self.budget - spent)
        }
    
    def to_dict(self):
        return {
            'id': self.id,
            'name': self.name,
            'icon': self.icon,
            'color': self.color,
            'budget': self.budget,
            'isDefault': self.is_default,
            'isActive': self.is_active,
            'createdAt': self.created_at.isoformat(),
            'updatedAt': self.updated_at.isoformat()
        }
    
    def __repr__(self):
        return f'<Category {self.name}>'


class Transaction(db.Model):
    __tablename__ = 'transactions'
    
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=False, index=True)
    category_id = db.Column(db.Integer, db.ForeignKey('categories.id'), nullable=True, index=True)
    merchant_name = db.Column(db.String(100), nullable=True)
    amount = db.Column(db.Float, nullable=False)
    transaction_date = db.Column(db.DateTime, nullable=False, index=True)
    description = db.Column(db.String(200), nullable=True)
    receipt_image = db.Column(db.String(255), nullable=True)  # Path to receipt image if uploaded
    is_scanned = db.Column(db.Boolean, default=False, nullable=False)  # Indicates if this transaction came from OCR scan
    payment_method = db.Column(db.String(50), nullable=True, default='Cash')
    reference_number = db.Column(db.String(100), nullable=True)  # For tracking/reference
    notes = db.Column(db.Text, nullable=True)  # Additional notes
    is_recurring = db.Column(db.Boolean, default=False, nullable=False)
    type = db.Column(db.String(10), default='expense')  # 'income' atau 'expense'
    created_at = db.Column(db.DateTime, default=lambda: datetime.now(timezone.utc), nullable=False)
    updated_at = db.Column(db.DateTime, default=lambda: datetime.now(timezone.utc), onupdate=lambda: datetime.now(timezone.utc), nullable=False)
    
    # Relationships
    items = db.relationship('TransactionItem', backref='transaction', lazy=True, cascade='all, delete-orphan')
    
    @property
    def total_items_value(self):
        """Calculate total value from transaction items"""
        return sum(item.quantity * item.price for item in self.items)
    
    def to_dict(self, include_items=True):
        data = {
            'id': self.id,
            'merchantName': self.merchant_name,
            'amount': self.amount,
            'date': self.transaction_date.strftime('%Y-%m-%d'),
            'description': self.description,
            'paymentMethod': self.payment_method,
            'referenceNumber': self.reference_number,
            'notes': self.notes,
            'isScanned': self.is_scanned,
            'isRecurring': self.is_recurring,
            'category': self.category.name if self.category else None,
            'categoryId': self.category_id,
            'createdAt': self.created_at.isoformat(),
            'updatedAt': self.updated_at.isoformat()
        }
        
        if include_items:
            data['items'] = [item.to_dict() for item in self.items]
        
        return data
    
    def __repr__(self):
        return f'<Transaction {self.id} ${self.amount:.2f}>'


class TransactionItem(db.Model):
    __tablename__ = 'transaction_items'
    
    id = db.Column(db.Integer, primary_key=True)
    transaction_id = db.Column(db.Integer, db.ForeignKey('transactions.id'), nullable=False, index=True)
    name = db.Column(db.String(100), nullable=False)
    quantity = db.Column(db.Integer, default=1, nullable=False)
    price = db.Column(db.Float, nullable=False)
    unit = db.Column(db.String(20), nullable=True)  # e.g., 'pcs', 'kg', 'liter'
    category = db.Column(db.String(50), nullable=True)  # Item-level category (different from transaction category)
    created_at = db.Column(db.DateTime, default=lambda: datetime.now(timezone.utc), nullable=False)
    
    @property
    def subtotal(self):
        """Calculate subtotal for this item"""
        return self.quantity * self.price
    
    def to_dict(self):
        return {
            'id': self.id,
            'name': self.name,
            'quantity': self.quantity,
            'price': self.price,
            'unit': self.unit,
            'category': self.category,
            'subtotal': self.subtotal,
            'createdAt': self.created_at.isoformat()
        }
    
    def __repr__(self):
        return f'<TransactionItem {self.name} ${self.price:.2f}>'


class Token(db.Model):
    __tablename__ = 'tokens'
    
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=False, index=True)
    token = db.Column(db.String(100), unique=True, nullable=False, default=lambda: str(uuid.uuid4()), index=True)
    token_type = db.Column(db.String(20), nullable=False)  # 'verify_email', 'reset_password', etc.
    expires_at = db.Column(db.DateTime, nullable=False, 
                           default=lambda: datetime.now(timezone.utc) + timedelta(hours=24))
    is_used = db.Column(db.Boolean, default=False, nullable=False)
    used_at = db.Column(db.DateTime, nullable=True)
    created_at = db.Column(db.DateTime, default=lambda: datetime.now(timezone.utc), nullable=False)
    
    @property
    def is_expired(self):
        return datetime.now(timezone.utc) > self.expires_at
    
    @property
    def is_valid(self):
        return not self.is_used and not self.is_expired
    
    def mark_as_used(self):
        """Mark token as used"""
        self.is_used = True
        self.used_at = datetime.now(timezone.utc)
        db.session.commit()
    
    def to_dict(self):
        return {
            'id': self.id,
            'token': self.token,
            'tokenType': self.token_type,
            'expiresAt': self.expires_at.isoformat(),
            'isUsed': self.is_used,
            'isExpired': self.is_expired,
            'isValid': self.is_valid,
            'createdAt': self.created_at.isoformat()
        }
    
    def __repr__(self):
        return f'<Token {self.token_type} for user {self.user_id}>'


# Budget model for more advanced budget tracking
class Budget(db.Model):
    __tablename__ = 'budgets'
    
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=False, index=True)
    category_id = db.Column(db.Integer, db.ForeignKey('categories.id'), nullable=True, index=True)
    name = db.Column(db.String(100), nullable=False)
    amount = db.Column(db.Float, nullable=False)
    period = db.Column(db.String(20), default='monthly', nullable=False)  # 'weekly', 'monthly', 'yearly'
    start_date = db.Column(db.DateTime, nullable=False)
    end_date = db.Column(db.DateTime, nullable=True)
    is_active = db.Column(db.Boolean, default=True, nullable=False)
    alert_threshold = db.Column(db.Float, default=80.0, nullable=False)  # Alert when % of budget is reached
    created_at = db.Column(db.DateTime, default=lambda: datetime.now(timezone.utc), nullable=False)
    updated_at = db.Column(db.DateTime, default=lambda: datetime.now(timezone.utc), onupdate=lambda: datetime.now(timezone.utc), nullable=False)
    
    # Relationships
    user = db.relationship('User', backref='budgets')
    category = db.relationship('Category', backref='budgets')
    
    def get_spent_amount(self):
        """Get amount spent in current budget period"""
        # Implementation depends on budget period logic
        pass
    
    def __repr__(self):
        return f'<Budget {self.name} ${self.amount:.2f}>'


# Default categories that will be created for each new user
DEFAULT_CATEGORIES = [
    {"name": "Makanan & Minuman", "icon": "utensils", "color": "#FF5733"},
    {"name": "Transportasi", "icon": "car", "color": "#33A8FF"},
    {"name": "Belanja", "icon": "shopping-bag", "color": "#33FF57"},
    {"name": "Hiburan", "icon": "film", "color": "#B533FF"},
    {"name": "Tagihan", "icon": "file-invoice", "color": "#FF33A8"},
    {"name": "Kesehatan", "icon": "medkit", "color": "#33FFF5"},
    {"name": "Pendidikan", "icon": "graduation-cap", "color": "#FFD433"},
    {"name": "Lain-lain", "icon": "ellipsis-h", "color": "#BFBFBF"},
]

# Helper function to create database tables
def create_tables(app):
    """Create all database tables"""
    with app.app_context():
        db.create_all()
        print("✅ Database tables created successfully")

# Helper function to reset database
def reset_database(app):
    """Drop and recreate all tables"""
    with app.app_context():
        db.drop_all()
        db.create_all()
        print("✅ Database reset successfully")