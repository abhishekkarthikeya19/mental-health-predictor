"""
Test script to verify that Pydantic is working correctly.
"""
import importlib.util

# Check if pydantic-settings is available
pydantic_settings_spec = importlib.util.find_spec("pydantic_settings")
if pydantic_settings_spec is not None:
    print("pydantic-settings is available")
    from pydantic_settings import BaseSettings
    print("Successfully imported BaseSettings from pydantic_settings")
else:
    print("pydantic-settings is NOT available")

# Check Pydantic version
try:
    import pydantic
    print(f"Pydantic version: {pydantic.__version__}")
    
    # Check if field_validator is available (Pydantic v2)
    if hasattr(pydantic, 'field_validator'):
        print("field_validator is available (Pydantic v2)")
    else:
        print("field_validator is NOT available (Pydantic v1)")
        
    # Check if validator is available (Pydantic v1)
    if hasattr(pydantic, 'validator'):
        print("validator is available (Pydantic v1)")
    else:
        print("validator is NOT available")
except ImportError:
    print("Pydantic is NOT installed")

# Test a simple model
from pydantic import BaseModel, Field

class TestModel(BaseModel):
    name: str = Field(..., min_length=1)
    age: int = Field(..., ge=0)

try:
    model = TestModel(name="Test", age=30)
    print(f"Model created successfully: {model}")
except Exception as e:
    print(f"Error creating model: {str(e)}")