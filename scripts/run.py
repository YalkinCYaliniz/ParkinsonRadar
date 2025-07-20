#!/usr/bin/env python3
"""
Parkinson Voice Analysis System - Startup Script
Starts the Flask web application with proper configuration
"""

import os
import sys
import subprocess
from pathlib import Path

def check_requirements():
    """Check if all required files and dependencies exist"""
    required_files = [
        'parkinsons.data',
        'app.py',
        'requirements.txt'
    ]
    
    missing_files = []
    for file in required_files:
        if not Path(file).exists():
            missing_files.append(file)
    
    if missing_files:
        print("❌ Missing required files:")
        for file in missing_files:
            print(f"   - {file}")
        return False
    
    return True

def install_dependencies():
    """Install Python dependencies if needed"""
    try:
        import flask
        import pandas
        import numpy
        import sklearn
        print("✅ Dependencies already installed")
        return True
    except ImportError:
        print("📦 Installing dependencies...")
        try:
            subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-r', 'requirements.txt'])
            print("✅ Dependencies installed successfully")
            return True
        except subprocess.CalledProcessError:
            print("❌ Failed to install dependencies")
            return False

def create_directories():
    """Create necessary directories"""
    directories = ['uploads', 'static/css', 'static/js', 'templates']
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
    
    print("✅ Directories created")

def main():
    """Main startup function"""
    print("🎤 Parkinson Voice Analysis System")
    print("=" * 50)
    
    # Check requirements
    if not check_requirements():
        print("\n❌ Setup incomplete. Please ensure all required files are present.")
        return
    
    # Install dependencies
    if not install_dependencies():
        print("\n❌ Failed to install dependencies. Please install manually.")
        return
    
    # Create directories
    create_directories()
    
    # Set environment variables
    os.environ['FLASK_ENV'] = 'development'
    os.environ['FLASK_DEBUG'] = '1'
    
    print("\n🚀 Starting Parkinson Voice Analysis System...")
    print("📍 URL: http://localhost:5001")
    print("🛑 Press Ctrl+C to stop the server")
    print("-" * 50)
    
    try:
        # Import and run the Flask app
        from app import app
        app.run(host='0.0.0.0', port=5000, debug=True)
    except KeyboardInterrupt:
        print("\n\n👋 Server stopped. Thank you for using Parkinson Voice Analysis!")
    except Exception as e:
        print(f"\n❌ Error starting server: {e}")
        print("Please check the error message and try again.")

if __name__ == '__main__':
    main() 