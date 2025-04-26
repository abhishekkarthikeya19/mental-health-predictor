#!/bin/bash

# Exit on error
set -e

echo "Setting up the Mental Health Predictor API server..."

# Make sure we're in the backend directory
cd "$(dirname "$0")"

# Check if virtual environment exists, create if not
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment
echo "Activating virtual environment..."
source venv/bin/activate

# Install dependencies
echo "Installing dependencies..."
pip install -r requirements.txt

echo "Setup complete! You can now run the server with:"
echo "source venv/bin/activate"
echo "uvicorn main:app --host 0.0.0.0 --port 8000"