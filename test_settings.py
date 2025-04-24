"""
Test script to verify that the Settings class works correctly.
"""
from backend.config import settings

print("Settings loaded successfully!")
print(f"API Version: {settings.API_VERSION}")
print(f"CORS Origins: {settings.CORS_ORIGINS}")