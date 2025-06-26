# check_auth_server.py
import os
import sys
import subprocess
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def check_auth_server():
    """Check auth server configuration and startup issues"""
    print("=== Auth Server Configuration Check ===")
    
    # Check environment variables
    auth_port = os.getenv('AUTH_PORT', '5001')
    print(f"- AUTH_PORT in environment: {auth_port}")
    
    # Check if port is available
    try:
        import socket
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        result = s.connect_ex(('localhost', int(auth_port)))
        if result == 0:
            print(f"⚠️ Warning: Port {auth_port} is already in use!")
        else:
            print(f"✓ Port {auth_port} is available")
        s.close()
    except Exception as e:
        print(f"Error checking port: {e}")
    
    # Try to run the auth server with output capture
    print("\nAttempting to start auth server...")
    try:
        env = os.environ.copy()
        env['FLASK_APP'] = 'auth.py'
        env['FLASK_ENV'] = 'development'
        env['FLASK_DEBUG'] = '1'
        
        result = subprocess.run(
            [sys.executable, 'auth.py'],
            env=env,
            capture_output=True,
            text=True,
            timeout=5  # Time out after 5 seconds
        )
        
        print(f"\nExit code: {result.returncode}")
        
        if result.stdout:
            print("\n=== STDOUT ===")
            print(result.stdout)
            
        if result.stderr:
            print("\n=== STDERR ===")
            print(result.stderr)
            
    except subprocess.TimeoutExpired:
        print("Server started and is running (timeout reached)")
    except Exception as e:
        print(f"Error running server: {e}")

if __name__ == '__main__':
    check_auth_server()