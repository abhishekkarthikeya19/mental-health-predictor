"""
Test script to make a request to the API.
"""
import requests
import json

url = "http://127.0.0.1:8000/predict/"
data = {"text_input": "I feel sad today"}

try:
    response = requests.post(url, json=data)
    print(f"Status Code: {response.status_code}")
    print(f"Response: {response.json()}")
except Exception as e:
    print(f"Error: {str(e)}")