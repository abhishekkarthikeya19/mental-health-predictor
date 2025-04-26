# config.py
"""
Configuration settings for the Mental Health Predictor API.
"""
import os
from typing import List, Dict, Any
import importlib.util

# Check if pydantic-settings is available
pydantic_settings_spec = importlib.util.find_spec("pydantic_settings")
if pydantic_settings_spec is not None:
    # For Pydantic v2
    from pydantic_settings import BaseSettings
    from pydantic import Field, ConfigDict
else:
    # For Pydantic v1
    from pydantic import BaseSettings, Field

class Settings(BaseSettings):
    """
    Application settings that can be overridden with environment variables.
    """
    # API settings
    API_VERSION: str = "1.0.0"
    API_TITLE: str = "Mental Health Predictor API"
    API_DESCRIPTION: str = """
    This API analyzes text input to detect potential signs of mental distress.
    
    ## Usage
    
    Send a POST request to the `/predict/` endpoint with your text in the request body.
    
    ## Disclaimer
    
    This tool is for educational purposes only and is not a substitute for professional mental health advice.
    If you or someone you know is experiencing a mental health crisis, please contact a mental health professional
    or a crisis helpline immediately.
    """
    
    # CORS settings
    # Define as a string to be parsed later
    CORS_ORIGINS_STR: str = "http://localhost,http://localhost:3000,http://localhost:8000,http://127.0.0.1,http://127.0.0.1:8000,https://mentalhealthdetection19.netlify.app"
    
    # This will be populated in __init__ or after settings are loaded
    CORS_ORIGINS: List[str] = []
    CORS_ALLOW_CREDENTIALS: bool = True
    
    # Rate limiting
    RATE_LIMIT_ENABLED: bool = True
    RATE_LIMIT_REQUESTS: int = 100  # Number of requests
    RATE_LIMIT_PERIOD: int = 3600   # Time period in seconds (1 hour)
    
    # Model settings
    MODEL_PATHS: List[str] = [
        "../app/model/mental_health_model.pkl",
        "app/model/mental_health_model.pkl",
        "./app/model/mental_health_model.pkl"
    ]
    
    # Recommendations
    DISTRESS_RECOMMENDATION: str = "Based on your text, you may be experiencing some distress. Consider talking to someone you trust or a mental health professional."
    NORMAL_RECOMMENDATION: str = "Your text doesn't show significant signs of distress. Continue practicing good mental health habits."
    
    # Configuration - handle both Pydantic v1 and v2
    if pydantic_settings_spec is not None:
        # For Pydantic v2
        model_config = ConfigDict(
            env_file=".env",
            case_sensitive=True
        )
    else:
        # For Pydantic v1
        class Config:
            env_file = ".env"
            case_sensitive = True

# Create a global settings object
settings = Settings()

# Parse CORS_ORIGINS from the string
if settings.CORS_ORIGINS_STR:
    settings.CORS_ORIGINS = [origin.strip() for origin in settings.CORS_ORIGINS_STR.split(",") if origin.strip()]

# Allow overriding settings with environment variables
if os.environ.get("ENVIRONMENT") == "production":
    # In production, restrict CORS to specific origins if provided in environment
    cors_env = os.environ.get("CORS_ORIGINS", "")
    if cors_env:
        settings.CORS_ORIGINS = [origin.strip() for origin in cors_env.split(",") if origin.strip()]
    
    # Enable stricter rate limiting in production
    settings.RATE_LIMIT_ENABLED = True
    settings.RATE_LIMIT_REQUESTS = int(os.environ.get("RATE_LIMIT_REQUESTS", "50"))
    settings.RATE_LIMIT_PERIOD = int(os.environ.get("RATE_LIMIT_PERIOD", "3600"))