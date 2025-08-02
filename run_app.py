#!/usr/bin/env python3
"""
Parkinson Voice Analysis Application Launcher
Hierarchical Project Structure
"""

import sys
import os

# Add project root to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def main():
    """Launch the Parkinson Voice Analysis application"""
    try:
        from app.main import app, initialize_models
        
        print("="*60)
        print("PARKINSON VOICE ANALYSIS APPLICATION")
        print("="*60)
        print("Initializing models and starting server...")
        
        # Initialize models
        initialize_models()
        
        # Start Flask application
        app.run(debug=True, host='0.0.0.0', port=5002)
        
    except ImportError as e:
        print(f"Import Error: {e}")
        print("Please ensure all dependencies are installed:")
        print("pip install -r requirements.txt")
        sys.exit(1)
    except Exception as e:
        print(f"Error starting application: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main() 