# models.py
"""
Pydantic models for request and response validation.
"""
import importlib.util
from typing import Optional

# Check if we're using Pydantic v1 or v2
pydantic_settings_spec = importlib.util.find_spec("pydantic_settings")
pydantic_v2 = pydantic_settings_spec is not None

if pydantic_v2:
    # Pydantic v2
    from pydantic import BaseModel, Field, field_validator
else:
    # Pydantic v1
    from pydantic import BaseModel, Field, validator

class PredictionRequest(BaseModel):
    """
    Request model for the prediction endpoint.
    """
    text_input: str = Field(
        ..., 
        min_length=1, 
        max_length=5000,
        description="Text to analyze for mental health indicators"
    )
    
    if pydantic_v2:
        # Pydantic v2 validator
        @field_validator('text_input')
        @classmethod
        def text_input_not_empty(cls, v):
            """Validate that text_input is not empty after stripping whitespace."""
            if not v.strip():
                raise ValueError('Text input cannot be empty or just whitespace')
            return v
    else:
        # Pydantic v1 validator
        @validator('text_input')
        def text_input_not_empty(cls, v):
            """Validate that text_input is not empty after stripping whitespace."""
            if not v.strip():
                raise ValueError('Text input cannot be empty or just whitespace')
            return v

class PredictionResponse(BaseModel):
    """
    Response model for the prediction endpoint.
    """
    prediction: int = Field(
        ..., 
        description="Prediction result: 0 for normal, 1 for distressed"
    )
    confidence: float = Field(
        ..., 
        ge=0.0, 
        le=1.0,
        description="Confidence score of the prediction (0.0 to 1.0)"
    )
    recommendation: str = Field(
        ..., 
        description="Recommendation based on the prediction"
    )

class ErrorResponse(BaseModel):
    """
    Standard error response model.
    """
    detail: str = Field(..., description="Error message")
    error_code: Optional[str] = Field(None, description="Error code for client reference")
    
class HealthCheckResponse(BaseModel):
    """
    Response model for the health check endpoint.
    """
    status: str = Field(..., description="Service status")
    version: str = Field(..., description="API version")