# db_check.py - Untuk memeriksa status database

import os
import sys
from dotenv import load_dotenv
from flask import Flask
from models import db, User, Category
from config import config

# Load environment variables
load_dotenv()

def create_app(config_name='development'):
    """Create Flask application with database setup"""
    app = Flask(__name__)
    app.config.from_object(config[config_name])
    
    # Initialize extensions
    db.init_app(app)
    
    return app

def check_database_connection():
    """Check if we can connect to the database"""
    app = create_app()
    
    with app.app_context():
        try:
            # Try to execute a simple query
            users_count = User.query.count()
            print(f"✅ Database connection successful")
            print(f"✅ Users in database: {users_count}")
            
            # Check if admin user exists
            admin = User.query.filter_by(email='admin@spendly.com').first()
            if admin:
                print(f"✅ Admin user exists: {admin.email}")
                
                # Check if admin can be authenticated
                if admin.check_password('Admin123!'):
                    print(f"✅ Admin password verification working")
                else:
                    print(f"❌ Admin password verification failed")
            else:
                print(f"❌ Admin user does not exist")
                
            # Check categories
            categories = Category.query.all()
            print(f"✅ Categories in database: {len(categories)}")
            
            return True
        except Exception as e:
            print(f"❌ Database connection error: {str(e)}")
            return False

if __name__ == '__main__':
    # Check database connection
    connection_ok = check_database_connection()
    
    if not connection_ok:
        print("\nTroubleshooting steps:")
        print("1. Check if PostgreSQL server is running")
        print("2. Verify database credentials in .env file")
        print("3. Make sure the database exists")
        print("4. Run setup_database.py to initialize the database")
        sys.exit(1)