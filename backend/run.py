# run.py
"""
Script to run the FastAPI application with uvicorn.
"""
import sys
import os
import logging
import argparse

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Run the Mental Health Predictor API")
    parser.add_argument(
        "--host", 
        type=str, 
        default="0.0.0.0", 
        help="Host to bind the server to"
    )
    parser.add_argument(
        "--port", 
        type=int, 
        default=8000, 
        help="Port to bind the server to"
    )
    parser.add_argument(
        "--reload", 
        action="store_true", 
        help="Enable auto-reload for development"
    )
    parser.add_argument(
        "--env", 
        type=str, 
        choices=["development", "production"], 
        default="development",
        help="Environment to run in"
    )
    return parser.parse_args()

def main():
    """Run the application."""
    args = parse_arguments()
    
    # Set environment variable
    os.environ["ENVIRONMENT"] = args.env
    
    # Import here to ensure patches are applied first in main.py
    import uvicorn
    
    logger.info(f"Starting server in {args.env} mode")
    logger.info(f"Server will be available at http://{args.host}:{args.port}")
    
    # Determine the correct import path based on the current directory
    import pathlib
    current_dir = pathlib.Path().absolute()
    
    if current_dir.name == "backend":
        # Running from inside the backend directory
        app_import_path = "main:app"
    else:
        # Running from the project root
        app_import_path = "backend.main:app"
    
    logger.info(f"Using import path: {app_import_path}")
    
    # Run the uvicorn server
    uvicorn.run(
        app_import_path, 
        host=args.host, 
        port=args.port, 
        reload=args.reload if args.env == "development" else False,
        log_level="info",
        workers=1
    )

if __name__ == "__main__":
    main()