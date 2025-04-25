"""
Script to run the complete training pipeline for the Mental Health Predictor.

This script:
1. Prepares the datasets
2. Trains the model
3. Evaluates the model
4. Saves the model and related artifacts
"""

import os
import sys
import logging
import subprocess
import time

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("app/training_log.txt")
    ]
)
logger = logging.getLogger(__name__)

def run_command(command):
    """Run a shell command and log the output."""
    logger.info(f"Running command: {command}")
    process = subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        shell=True,
        universal_newlines=True
    )
    
    # Stream the output
    for line in process.stdout:
        logger.info(line.strip())
    
    # Get the return code
    process.wait()
    return process.returncode

def main():
    """Run the complete training pipeline."""
    start_time = time.time()
    logger.info("Starting Mental Health Predictor training pipeline")
    
    # Step 1: Prepare the datasets
    logger.info("Step 1: Preparing datasets")
    data_prep_result = run_command("python app/data_preparation.py")
    if data_prep_result != 0:
        logger.error("Dataset preparation failed")
        return False
    
    # Step 2: Train the model
    logger.info("Step 2: Training the model")
    train_result = run_command("python app/train_model.py")
    if train_result != 0:
        logger.error("Model training failed")
        return False
    
    # Calculate total time
    total_time = time.time() - start_time
    logger.info(f"Training pipeline completed in {total_time:.2f} seconds ({total_time/60:.2f} minutes)")
    
    # Check if model files exist
    model_file = "app/model/mental_health_model.pkl"
    transformer_dir = "app/model/transformer_model"
    
    if os.path.exists(model_file) and os.path.exists(transformer_dir):
        logger.info("Model files successfully created")
        logger.info(f"Model file: {model_file}")
        logger.info(f"Transformer model directory: {transformer_dir}")
        return True
    else:
        logger.error("Model files not found after training")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)