# utils/model_loader.py
"""
Utility module for loading and validating the ML model.
"""
import os
import logging
import joblib
from typing import Any, List, Tuple

logger = logging.getLogger(__name__)

class ModelNotFoundError(Exception):
    """Exception raised when the model file cannot be found."""
    pass

class InvalidModelError(Exception):
    """Exception raised when the loaded model doesn't have the required methods."""
    pass

def validate_model(model: Any) -> bool:
    """
    Validate that the model has the required methods.
    
    Args:
        model: The loaded model to validate
        
    Returns:
        bool: True if the model is valid
        
    Raises:
        InvalidModelError: If the model doesn't have the required methods
    """
    required_methods = ['predict', 'predict_proba']
    
    for method in required_methods:
        if not hasattr(model, method) or not callable(getattr(model, method)):
            raise InvalidModelError(f"Model is missing required method: {method}")
    
    # Test that the model can make predictions on a simple input
    try:
        test_input = ["This is a test input"]
        prediction = model.predict(test_input)
        probabilities = model.predict_proba(test_input)
        logger.debug(f"Model validation successful. Test prediction: {prediction}, Probabilities: {probabilities}")
        return True
    except Exception as e:
        raise InvalidModelError(f"Model failed validation with error: {str(e)}")

def load_model(model_paths: List[str] = None) -> Any:
    """
    Load the ML model from one of the provided paths.
    
    Args:
        model_paths: List of possible paths to the model file
        
    Returns:
        The loaded model
        
    Raises:
        ModelNotFoundError: If the model cannot be found in any of the provided paths
    """
    if model_paths is None:
        model_paths = [
            "../app/model/mental_health_model.pkl",
            "app/model/mental_health_model.pkl",
            "./app/model/mental_health_model.pkl"
        ]
    
    for path in model_paths:
        try:
            logger.info(f"Attempting to load model from: {path}")
            model = joblib.load(path)
            logger.info(f"Successfully loaded model from: {path}")
            
            # Validate the model
            validate_model(model)
            return model
        except FileNotFoundError:
            logger.warning(f"Model not found at: {path}")
        except Exception as e:
            logger.error(f"Error loading model from {path}: {str(e)}")
    
    # If we get here, we couldn't load the model from any path
    raise ModelNotFoundError("Could not find the model file in any of the provided paths")