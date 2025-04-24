#!/usr/bin/env python
"""
Simple script to run the Mental Health Predictor API.
"""
import sys
import os

# Add the project root to the Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import and run the application
from backend.run import main

if __name__ == "__main__":
    main()