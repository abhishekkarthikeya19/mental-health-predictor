# test_api.py
"""
Tests for the Mental Health Predictor API.
"""
import pytest
from fastapi.testclient import TestClient
from .main import app

client = TestClient(app)

def test_read_root():
    """Test the root endpoint."""
    response = client.get("/")
    assert response.status_code == 200
    assert "message" in response.json()

def test_health_check():
    """Test the health check endpoint."""
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "healthy"

def test_predict_empty_input():
    """Test prediction with empty input."""
    response = client.post(
        "/predict/",
        json={"text_input": ""}
    )
    assert response.status_code == 422  # Validation error

def test_predict_valid_input():
    """Test prediction with valid input."""
    response = client.post(
        "/predict/",
        json={"text_input": "I feel happy today and everything is going well."}
    )
    assert response.status_code == 200
    data = response.json()
    assert "prediction" in data
    assert "confidence" in data
    assert "recommendation" in data
    assert isinstance(data["prediction"], int)
    assert isinstance(data["confidence"], float)
    assert isinstance(data["recommendation"], str)

def test_predict_legacy_endpoint():
    """Test the legacy prediction endpoint."""
    response = client.post(
        "/predict/legacy",
        json={"text_input": "I feel happy today and everything is going well."}
    )
    assert response.status_code == 200
    data = response.json()
    assert "prediction" in data
    assert isinstance(data["prediction"], int)

def test_predict_legacy_empty_input():
    """Test legacy prediction with empty input."""
    response = client.post(
        "/predict/legacy",
        json={"text_input": ""}
    )
    assert response.status_code == 400  # Bad request

if __name__ == "__main__":
    pytest.main(["-v", "test_api.py"])