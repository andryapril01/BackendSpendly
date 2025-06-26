# setup_database.py - FIXED Database Setup Script
from flask import Flask
from flask_migrate import Migrate
from models import db, User, Category, DEFAULT_CATEGORIES
from config import config
import os
import sys
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def create_app(config_name='development'):
    """Create Flask application with database setup"""
    app = Flask(__name__)
    app.config.from_object(config[config_name])
    
    # Initialize extensions
    db.init_app(app)
    migrate = Migrate(app, db)
    
    return app

def test_database_connection(app):
    """Test database connection"""
    try:
        with app.app_context():
            # Test basic connection
            db.session.execute(db.text('SELECT 1'))
            logger.info("✅ Database connection successful")
            return True
    except Exception as e:
        logger.error(f"❌ Database connection failed: {str(e)}")
        return False

def setup_database(config_name='development'):
    """Create database and run initial setup"""
    app = create_app(config_name)
    
    logger.info(f"🚀 Setting up database for environment: {config_name}")
    logger.info(f"📊 Database URL: {app.config['SQLALCHEMY_DATABASE_URI']}")
    
    with app.app_context():
        try:
            # Test connection first
            if not test_database_connection(app):
                logger.error("❌ Cannot connect to database. Please check your database configuration.")
                return False
            
            # Create all tables
            logger.info("🔨 Creating database tables...")
            db.create_all()
            logger.info("✅ Database tables created successfully!")
            
            # Check if default admin user exists
            admin_exists = User.query.filter_by(email='admin@spendly.com').first()
            if not admin_exists:
                logger.info("👤 Creating default admin user...")
                
                # Create admin user with a secure default password
                admin_user = User(
                    email='admin@spendly.com',
                    first_name='Admin',
                    last_name='User',
                    password='Admin123!',  # This will be hashed automatically
                    is_active=True,
                    is_verified=True
                )
                
                db.session.add(admin_user)
                db.session.flush()  # Get the user ID
                
                logger.info(f"✅ Admin user created with ID: {admin_user.id}")
                
                # Create default categories for admin user
                logger.info("📂 Creating default categories for admin user...")
                categories = create_default_categories(admin_user.id)
                for category in categories:
                    db.session.add(category)
                
                # Commit all changes
                db.session.commit()
                
                logger.info(f"✅ Created {len(categories)} default categories")
                logger.info("✅ Default admin user setup completed!")
                logger.info("📧 Admin email: admin@spendly.com")
                logger.info("🔑 Admin password: Admin123!")
                
            else:
                logger.info(f"✅ Admin user already exists with email: {admin_exists.email}")
                
                # Check if admin has categories
                admin_categories = Category.query.filter_by(user_id=admin_exists.id).count()
                if admin_categories == 0:
                    logger.info("📂 Adding default categories to existing admin user...")
                    categories = create_default_categories(admin_exists.id)
                    for category in categories:
                        db.session.add(category)
                    db.session.commit()
                    logger.info(f"✅ Added {len(categories)} default categories to admin user")
                else:
                    logger.info(f"✅ Admin user has {admin_categories} categories")
            
            # Display summary
            total_users = User.query.count()
            total_categories = Category.query.count()
            logger.info("📊 Database Setup Summary:")
            logger.info(f"   👥 Total Users: {total_users}")
            logger.info(f"   📂 Total Categories: {total_categories}")
            
            logger.info("🎉 Database setup completed successfully!")
            return True
            
        except Exception as e:
            logger.error(f"❌ Error setting up database: {str(e)}")
            logger.error(f"❌ Traceback: {str(e)}")
            db.session.rollback()
            return False

def reset_database(config_name='development'):
    """
    Drop all tables and recreate them
    WARNING: This will delete all data!
    """
    app = create_app(config_name)
    
    logger.warning("⚠️ RESETTING DATABASE - ALL DATA WILL BE LOST!")
    
    with app.app_context():
        try:
            # Drop all tables
            logger.info("🗑️ Dropping all tables...")
            db.drop_all()
            logger.warning("⚠️ All tables dropped!")
            
            # Recreate tables
            logger.info("🔨 Recreating tables...")
            db.create_all()
            logger.info("✅ Database tables recreated successfully!")
            
            # Create default admin user
            logger.info("👤 Creating default admin user...")
            admin_user = User(
                email='admin@spendly.com',
                first_name='Admin',
                last_name='User',
                password='Admin123!',
                is_active=True,
                is_verified=True
            )
            
            db.session.add(admin_user)
            db.session.flush()
            
            # Create default categories
            categories = create_default_categories(admin_user.id)
            for category in categories:
                db.session.add(category)
            
            db.session.commit()
            
            logger.info("✅ Database reset completed!")
            logger.info("📧 Admin email: admin@spendly.com")
            logger.info("🔑 Admin password: Admin123!")
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Error resetting database: {str(e)}")
            db.session.rollback()
            return False

def create_default_categories(user_id):
    """Create default categories for a user"""
    categories = []
    for cat_data in DEFAULT_CATEGORIES:
        category = Category(
            user_id=user_id,
            name=cat_data['name'],
            icon=cat_data['icon'],
            color=cat_data['color'],
            is_default=True,
            budget=1000000  # Default budget 1M IDR
        )
        categories.append(category)
    return categories

def check_database_status(config_name='development'):
    """Check current database status"""
    app = create_app(config_name)
    
    logger.info(f"🔍 Checking database status for environment: {config_name}")
    
    with app.app_context():
        try:
            # Test connection
            if not test_database_connection(app):
                return False
            
            # Get statistics
            user_count = User.query.count()
            category_count = Category.query.count()
            
            logger.info("📊 Database Status:")
            logger.info(f"   👥 Users: {user_count}")
            logger.info(f"   📂 Categories: {category_count}")
            
            # List users
            if user_count > 0:
                logger.info("👥 Users in database:")
                users = User.query.limit(10).all()
                for user in users:
                    logger.info(f"   - {user.email} (ID: {user.id}, Active: {user.is_active})")
            
            # Check admin user
            admin_user = User.query.filter_by(email='admin@spendly.com').first()
            if admin_user:
                logger.info(f"✅ Admin user exists: {admin_user.email} (ID: {admin_user.id})")
            else:
                logger.warning("⚠️ No admin user found")
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Error checking database status: {str(e)}")
            return False

if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Database setup script for Spendly')
    parser.add_argument('--reset', action='store_true', help='Reset database (WARNING: Deletes all data)')
    parser.add_argument('--config', default='development', help='Configuration to use (development/testing/production)')
    parser.add_argument('--status', action='store_true', help='Check database status')
    parser.add_argument('--test-connection', action='store_true', help='Test database connection only')
    
    args = parser.parse_args()
    
    logger.info(f"🔧 Using configuration: {args.config}")
    
    if args.test_connection:
        app = create_app(args.config)
        if test_database_connection(app):
            logger.info("✅ Database connection test passed")
            sys.exit(0)
        else:
            logger.error("❌ Database connection test failed")
            sys.exit(1)
            
    elif args.status:
        if check_database_status(args.config):
            sys.exit(0)
        else:
            sys.exit(1)
            
    elif args.reset:
        confirm = input("⚠️  This will delete ALL data. Are you sure? (type 'yes' to confirm): ")
        if confirm.lower() == 'yes':
            if reset_database(args.config):
                logger.info("✅ Database reset completed successfully")
                sys.exit(0)
            else:
                logger.error("❌ Database reset failed")
                sys.exit(1)
        else:
            logger.info("❌ Operation cancelled")
            sys.exit(0)
    else:
        # Default: setup database
        if setup_database(args.config):
            logger.info("✅ Database setup completed successfully")
            sys.exit(0)
        else:
            logger.error("❌ Database setup failed")
            sys.exit(1)