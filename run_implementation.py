"""
Script to run the complete implementation of the Mental Health Predictor.

This script:
1. Prepares the datasets
2. Trains the model
3. Tests the model
4. Starts the backend API server
"""

import os
import sys
import subprocess
import time
import logging
import argparse

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("implementation_log.txt")
    ]
)
logger = logging.getLogger(__name__)

def run_command(command, cwd=None):
    """Run a shell command and log the output."""
    logger.info(f"Running command: {command}")
    process = subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        shell=True,
        universal_newlines=True,
        cwd=cwd
    )
    
    # Stream the output
    for line in process.stdout:
        logger.info(line.strip())
    
    # Get the return code
    process.wait()
    return process.returncode

def prepare_environment():
    """Install required dependencies."""
    logger.info("Installing required dependencies...")
    return run_command("pip install -r requirements.txt")

def prepare_datasets():
    """Prepare the datasets for training."""
    logger.info("Preparing datasets...")
    return run_command("python app/data_preparation.py")

def train_model():
    """Train the model using the prepared datasets."""
    logger.info("Training the model...")
    return run_command("python app/train_model.py")

def test_model():
    """Test the trained model."""
    logger.info("Testing the model...")
    return run_command("python app/test_model.py")

def start_backend():
    """Start the backend API server."""
    logger.info("Starting the backend API server...")
    return run_command("python backend/run.py")

def start_frontend():
    """Start the frontend development server."""
    logger.info("Starting the frontend development server...")
    return run_command("npm start", cwd="frontend")

def main():
    """Run the complete implementation."""
    parser = argparse.ArgumentParser(description="Run the Mental Health Predictor implementation")
    parser.add_argument("--skip-training", action="store_true", help="Skip dataset preparation and model training")
    parser.add_argument("--backend-only", action="store_true", help="Start only the backend API server")
    parser.add_argument("--frontend-only", action="store_true", help="Start only the frontend development server")
    args = parser.parse_args()
    
    start_time = time.time()
    logger.info("Starting Mental Health Predictor implementation")
    
    # Prepare environment
    if not args.frontend_only:
        env_result = prepare_environment()
        if env_result != 0:
            logger.error("Failed to install dependencies")
            return False
    
    # Skip training if requested
    if not args.skip_training and not args.frontend_only:
        # Prepare datasets
        data_prep_result = prepare_datasets()
        if data_prep_result != 0:
            logger.error("Dataset preparation failed")
            return False
        
        # Train model
        train_result = train_model()
        if train_result != 0:
            logger.error("Model training failed")
            return False
        
        # Test model
        test_result = test_model()
        if test_result != 0:
            logger.error("Model testing failed")
            return False
    
    # Start backend if not frontend-only
    if not args.frontend_only:
        if args.backend_only:
            # Start backend and wait for it to complete
            backend_result = start_backend()
            if backend_result != 0:
                logger.error("Backend API server failed to start")
                return False
        else:
            # Start backend in a separate process
            backend_process = subprocess.Popen(
                "python backend/run.py",
                shell=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                universal_newlines=True
            )
            logger.info("Backend API server started in the background")
    
    # Start frontend if not backend-only
    if not args.backend_only:
        frontend_result = start_frontend()
        if frontend_result != 0:
            logger.error("Frontend development server failed to start")
            return False
    
    # Calculate total time
    total_time = time.time() - start_time
    logger.info(f"Implementation completed in {total_time:.2f} seconds ({total_time/60:.2f} minutes)")
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)