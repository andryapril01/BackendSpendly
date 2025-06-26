# fix_database.py - Database Troubleshooting and Fix Tool

import os
import sys
import subprocess
import time
from pathlib import Path
from dotenv import load_dotenv
import psutil

# Load environment variables
load_dotenv()

def check_postgresql_service():
    """Check if PostgreSQL service is running"""
    print("🔍 Checking PostgreSQL service...")
    
    # Check if PostgreSQL process is running
    postgres_running = False
    for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
        try:
            if 'postgres' in proc.info['name'].lower():
                postgres_running = True
                print(f"✅ Found PostgreSQL process: {proc.info['name']} (PID: {proc.info['pid']})")
                break
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue
    
    if not postgres_running:
        print("❌ PostgreSQL service is not running")
        return False
    
    print("✅ PostgreSQL service is running")
    return True

def start_postgresql_windows():
    """Try to start PostgreSQL on Windows"""
    try:
        print("🚀 Attempting to start PostgreSQL service on Windows...")
        
        # Try to start PostgreSQL service
        result = subprocess.run([
            'sc', 'start', 'postgresql-x64-14'
        ], capture_output=True, text=True, shell=True)
        
        if result.returncode == 0:
            print("✅ PostgreSQL service started successfully")
            time.sleep(3)  # Wait for service to fully start
            return True
        else:
            print(f"❌ Failed to start PostgreSQL service: {result.stderr}")
            
            # Try alternative service names
            for service_name in ['postgresql-14', 'postgresql', 'PostgreSQL']:
                print(f"🔄 Trying service name: {service_name}")
                result = subprocess.run([
                    'sc', 'start', service_name
                ], capture_output=True, text=True, shell=True)
                
                if result.returncode == 0:
                    print(f"✅ PostgreSQL service started with name: {service_name}")
                    time.sleep(3)
                    return True
            
            return False
            
    except Exception as e:
        print(f"❌ Error starting PostgreSQL: {e}")
        return False

def install_postgresql_windows():
    """Guide user to install PostgreSQL on Windows"""
    print("📦 PostgreSQL Installation Guide for Windows:")
    print("1. Download PostgreSQL from: https://www.postgresql.org/download/windows/")
    print("2. Run the installer and follow the setup wizard")
    print("3. Remember your superuser password")
    print("4. Default port should be 5432")
    print("5. After installation, restart this script")
    
    choice = input("\n❓ Have you installed PostgreSQL? (y/n): ").lower()
    return choice == 'y'

def create_postgresql_user():
    """Create PostgreSQL user and database"""
    try:
        print("👤 Creating PostgreSQL user and database...")
        
        DB_USER = os.getenv('DB_USER', 'spendly_user')
        DB_PASSWORD = os.getenv('DB_PASSWORD', 'spendly_pass')
        DB_NAME = os.getenv('DB_NAME', 'spendly_dev')
        
        # Ask for PostgreSQL superuser password
        postgres_password = input("🔑 Enter PostgreSQL superuser (postgres) password: ")
        
        # Create user
        user_sql = f"""
        DO $$
        BEGIN
            IF NOT EXISTS (SELECT FROM pg_catalog.pg_user WHERE usename = '{DB_USER}') THEN
                CREATE USER {DB_USER} WITH PASSWORD '{DB_PASSWORD}';
                ALTER USER {DB_USER} CREATEDB;
                GRANT ALL PRIVILEGES ON DATABASE postgres TO {DB_USER};
            END IF;
        END
        $$;
        """
        
        # Create database
        db_sql = f"""
        SELECT 'CREATE DATABASE {DB_NAME} OWNER {DB_USER}'
        WHERE NOT EXISTS (SELECT FROM pg_database WHERE datname = '{DB_NAME}')\\gexec
        """
        
        # Execute SQL commands
        print("🔧 Creating user...")
        user_result = subprocess.run([
            'psql', '-U', 'postgres', '-h', 'localhost', '-c', user_sql
        ], input=postgres_password, text=True, capture_output=True)
        
        if user_result.returncode == 0:
            print("✅ User created successfully")
        else:
            print(f"⚠️ User creation: {user_result.stderr}")
        
        print("🔧 Creating database...")
        db_result = subprocess.run([
            'psql', '-U', 'postgres', '-h', 'localhost', '-c', db_sql
        ], input=postgres_password, text=True, capture_output=True)
        
        if db_result.returncode == 0:
            print("✅ Database setup completed")
            return True
        else:
            print(f"⚠️ Database creation: {db_result.stderr}")
            return False
            
    except Exception as e:
        print(f"❌ Error creating PostgreSQL user/database: {e}")
        return False

def setup_sqlite_fallback():
    """Setup SQLite as fallback database"""
    try:
        print("💾 Setting up SQLite fallback database...")
        
        # Create SQLite database file
        db_file = Path('spendly_dev.db')
        
        if not db_file.exists():
            db_file.touch()
            print("✅ SQLite database file created")
        else:
            print("✅ SQLite database file already exists")
        
        # Update .env file to use SQLite
        env_file = Path('.env')
        
        if env_file.exists():
            with open(env_file, 'r') as f:
                content = f.read()
            
            # Comment out PostgreSQL DATABASE_URL and add SQLite
            lines = content.split('\n')
            new_lines = []
            
            for line in lines:
                if line.startswith('DATABASE_URL=postgresql'):
                    new_lines.append(f'# {line}')
                elif line.startswith('# DATABASE_URL=sqlite'):
                    new_lines.append('DATABASE_URL=sqlite:///spendly_dev.db')
                else:
                    new_lines.append(line)
            
            # Add SQLite URL if not present
            if 'DATABASE_URL=sqlite:///spendly_dev.db' not in new_lines:
                new_lines.append('DATABASE_URL=sqlite:///spendly_dev.db')
            
            with open(env_file, 'w') as f:
                f.write('\n'.join(new_lines))
            
            print("✅ .env file updated to use SQLite")
        
        return True
        
    except Exception as e:
        print(f"❌ Error setting up SQLite: {e}")
        return False

def test_database_connection():
    """Test database connection"""
    try:
        print("🔍 Testing database connection...")
        
        # Import and test
        from sqlalchemy import create_engine, text
        
        db_url = os.getenv('DATABASE_URL')
        if not db_url:
            print("❌ No DATABASE_URL found in environment")
            return False
        
        print(f"🔗 Testing connection to: {db_url[:50]}...")
        
        engine = create_engine(db_url)
        with engine.connect() as conn:
            result = conn.execute(text('SELECT 1'))
            result.fetchone()
        
        print("✅ Database connection successful!")
        return True
        
    except Exception as e:
        print(f"❌ Database connection failed: {e}")
        return False

def run_database_setup():
    """Run the database setup script"""
    try:
        print("🚀 Running database setup...")
        
        result = subprocess.run([
            sys.executable, 'setup_database.py'
        ], capture_output=True, text=True)
        
        if result.returncode == 0:
            print("✅ Database setup completed successfully")
            print(result.stdout)
            return True
        else:
            print(f"❌ Database setup failed:")
            print(result.stderr)
            return False
            
    except Exception as e:
        print(f"❌ Error running database setup: {e}")
        return False

def main():
    """Main troubleshooting function"""
    print("🔧 === SPENDLY DATABASE TROUBLESHOOTING ===")
    print("This tool will help diagnose and fix database issues\n")
    
    # Step 1: Check if .env file exists
    if not Path('.env').exists():
        print("❌ .env file not found!")
        print("📋 Creating .env file with default configuration...")
        
        # Create basic .env file
        with open('.env', 'w') as f:
            f.write("""# Spendly Environment Configuration
DB_HOST=localhost
DB_PORT=5432
DB_NAME=spendly_dev
DB_USER=spendly_user
DB_PASSWORD=spendly_pass
DATABASE_URL=sqlite:///spendly_dev.db
FLASK_ENV=development
FLASK_DEBUG=1
JWT_SECRET_KEY=dev-secret-key-change-in-production
SECRET_KEY=dev-secret-key-change-in-production
AUTH_PORT=5001
OCR_PORT=5000
""")
        print("✅ .env file created with SQLite fallback")
    
    # Step 2: Check PostgreSQL
    postgres_available = check_postgresql_service()
    
    if not postgres_available:
        print("\n🔄 PostgreSQL is not running. Attempting to start...")
        
        if sys.platform == 'win32':
            postgres_started = start_postgresql_windows()
            
            if not postgres_started:
                print("\n❓ PostgreSQL is not installed or not starting.")
                install_choice = input("Do you want to install PostgreSQL? (y/n): ").lower()
                
                if install_choice == 'y':
                    install_postgresql_windows()
                    return
                else:
                    print("🔄 Using SQLite fallback instead...")
                    setup_sqlite_fallback()
        else:
            print("🐧 For Linux/Mac, please start PostgreSQL manually:")
            print("   sudo systemctl start postgresql  # Linux")
            print("   brew services start postgresql   # Mac")
            print("\n🔄 Using SQLite fallback instead...")
            setup_sqlite_fallback()
    else:
        # PostgreSQL is running, try to create user/database
        print("\n🔧 PostgreSQL is running. Setting up user and database...")
        pg_setup = create_postgresql_user()
        
        if not pg_setup:
            print("⚠️ PostgreSQL setup failed. Using SQLite fallback...")
            setup_sqlite_fallback()
    
    # Step 3: Test database connection
    print("\n🔍 Testing database connection...")
    
    # Reload environment variables
    load_dotenv(override=True)
    
    connection_ok = test_database_connection()
    
    if not connection_ok:
        print("🔄 Connection failed. Ensuring SQLite fallback...")
        setup_sqlite_fallback()
        load_dotenv(override=True)
        connection_ok = test_database_connection()
    
    # Step 4: Run database setup
    if connection_ok:
        print("\n📋 Running database table setup...")
        setup_ok = run_database_setup()
        
        if setup_ok:
            print("\n🎉 === DATABASE TROUBLESHOOTING COMPLETED ===")
            print("✅ Database is ready!")
            print("🏃 You can now run: python run_servers.py")
            print("\n📧 Default login credentials:")
            print("   Email: admin@spendly.com")
            print("   Password: admin123")
        else:
            print("❌ Database table setup failed")
    else:
        print("❌ Could not establish database connection")
        print("💡 Try manually installing PostgreSQL or check SQLite permissions")

if __name__ == '__main__':
    main()