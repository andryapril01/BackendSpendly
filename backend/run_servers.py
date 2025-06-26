# run_servers.py
import subprocess
import time
import os
import sys
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Get port settings
AUTH_PORT = os.getenv('AUTH_PORT', '5001')
OCR_PORT = os.getenv('OCR_PORT', '5000')

def start_server(server_file, port, name):
    """Start a Flask server as a subprocess"""
    print(f"Starting {name} server on port {port}...")
    
    # Prepare environment variables
    env = os.environ.copy()
    env['FLASK_APP'] = server_file
    env['FLASK_ENV'] = 'development'
    env['FLASK_DEBUG'] = '1'
    
    # Set the port in environment to ensure it's used by the server
    if name == 'Auth':
        env['AUTH_PORT'] = port
    elif name == 'OCR':
        env['OCR_PORT'] = port
    
    # Command to start the server
    command = [sys.executable, server_file]
    
    # Start the server as a subprocess with proper output redirection
    process = subprocess.Popen(
        command,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        universal_newlines=True,
        bufsize=1  # Line buffered
    )
    
    print(f"{name} server process started with PID: {process.pid}")
    return process

def monitor_process_output(process, name, timeout=5):
    """Monitor the initial output of a process to detect startup issues"""
    start_time = time.time()
    output_lines = []
    error_lines = []
    
    # Check for initial output for a few seconds to detect startup issues
    while time.time() - start_time < timeout:
        # Check stdout
        if process.stdout.readable():
            line = process.stdout.readline()
            if line:
                output_lines.append(line.strip())
                print(f"[{name}] {line.strip()}")
        
        # Check stderr
        if process.stderr.readable():
            line = process.stderr.readline()
            if line:
                error_lines.append(line.strip())
                print(f"[{name} ERROR] {line.strip()}")
        
        # If process ended, break
        if process.poll() is not None:
            break
            
        time.sleep(0.1)
    
    return output_lines, error_lines

if __name__ == '__main__':
    try:
        # Ensure database is set up
        try:
            subprocess.run([sys.executable, 'setup_database.py'], check=True)
            print("Database setup complete.")
        except subprocess.CalledProcessError as e:
            print(f"Error setting up database: {e}")
            sys.exit(1)
        
        # Start auth server
        auth_process = start_server('auth.py', AUTH_PORT, 'Auth')
        
        # Start OCR server
        ocr_process = start_server('app.py', OCR_PORT, 'OCR')
        
        # Monitor initial output
        auth_output, auth_errors = monitor_process_output(auth_process, 'Auth')
        ocr_output, ocr_errors = monitor_process_output(ocr_process, 'OCR')
        
        # Check if servers are running
        if auth_process.poll() is not None:
            print(f"Auth server failed to start. Exit code: {auth_process.returncode}")
            if auth_errors:
                print("Auth server errors:")
                for line in auth_errors:
                    print(f"  {line}")
            sys.exit(1)
        else:
            print(f"Auth server running on http://localhost:{AUTH_PORT}")
        
        if ocr_process.poll() is not None:
            print(f"OCR server failed to start. Exit code: {ocr_process.returncode}")
            if ocr_errors:
                print("OCR server errors:")
                for line in ocr_errors:
                    print(f"  {line}")
            sys.exit(1)
        else:
            print(f"OCR server running on http://localhost:{OCR_PORT}")
        
        # Print success message
        print("\n✅ All servers are running")
        print(f"- Auth API: http://localhost:{AUTH_PORT}/api/auth")
        print(f"- OCR API: http://localhost:{OCR_PORT}/api/scan-receipt")
        print("\nPress Ctrl+C to stop all servers")
        
        # Monitor servers continuously
        while auth_process.poll() is None and ocr_process.poll() is None:
            # Check output from auth server
            if auth_process.stdout.readable():
                line = auth_process.stdout.readline()
                if line:
                    print(f"[Auth] {line.strip()}")
            
            # Check stderr from auth server
            if auth_process.stderr.readable():
                line = auth_process.stderr.readline()
                if line:
                    print(f"[Auth ERROR] {line.strip()}")
            
            # Check output from OCR server
            if ocr_process.stdout.readable():
                line = ocr_process.stdout.readline()
                if line:
                    print(f"[OCR] {line.strip()}")
            
            # Check stderr from OCR server
            if ocr_process.stderr.readable():
                line = ocr_process.stderr.readline()
                if line:
                    print(f"[OCR ERROR] {line.strip()}")
            
            # Sleep to avoid high CPU usage
            time.sleep(0.1)
        
    except KeyboardInterrupt:
        print("\nShutting down servers...")
        
        # Terminate processes
        if 'auth_process' in locals() and auth_process.poll() is None:
            auth_process.terminate()
            print("Auth server terminated.")
        
        if 'ocr_process' in locals() and ocr_process.poll() is None:
            ocr_process.terminate()
            print("OCR server terminated.")
        
        print("All servers stopped.")
    
    except Exception as e:
        print(f"Error: {e}")
    
    finally:
        # Make sure all processes are terminated
        if 'auth_process' in locals() and auth_process.poll() is None:
            auth_process.terminate()
        
        if 'ocr_process' in locals() and ocr_process.poll() is None:
            ocr_process.terminate()