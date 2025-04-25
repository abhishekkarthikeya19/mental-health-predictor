"""
Script to test the trained Mental Health Predictor model with example inputs.
"""

import os
import sys
import joblib
import pandas as pd
import numpy as np
from tabulate import tabulate
import json

def load_model():
    """Load the trained model."""
    model_path = "app/model/mental_health_model.pkl"
    if not os.path.exists(model_path):
        print(f"Error: Model file not found at {model_path}")
        return None
    
    try:
        model = joblib.load(model_path)
        print(f"Model loaded successfully from {model_path}")
        return model
    except Exception as e:
        print(f"Error loading model: {e}")
        return None

def print_model_info():
    """Print information about the trained model."""
    info_path = "app/model/model_info.json"
    if not os.path.exists(info_path):
        print("Model info file not found")
        return
    
    try:
        with open(info_path, "r") as f:
            info = json.load(f)
        
        print("\nModel Information:")
        print("-----------------")
        print(f"Model name: {info.get('model_name', 'Unknown')}")
        print(f"Version: {info.get('version', 'Unknown')}")
        print(f"Created on: {info.get('creation_date', 'Unknown')}")
        print(f"Training dataset size: {info.get('training_dataset_size', 'Unknown')}")
        print(f"Test dataset size: {info.get('test_dataset_size', 'Unknown')}")
        print(f"Training time: {info.get('training_time_seconds', 0)/60:.2f} minutes")
        print(f"Accuracy: {info.get('accuracy', 0):.4f}")
        
        print("\nClassification Report:")
        print(info.get('classification_report', 'Not available'))
    except Exception as e:
        print(f"Error reading model info: {e}")

def test_examples(model, examples):
    """Test the model with example inputs."""
    if model is None:
        return
    
    results = []
    
    for text in examples:
        # Get prediction and confidence
        prediction = int(model.predict([text])[0])
        probabilities = model.predict_proba([text])[0]
        confidence = probabilities[prediction]
        
        # Create result dictionary
        result = {
            "text": text,
            "prediction": "Distressed" if prediction == 1 else "Normal",
            "confidence": f"{confidence:.2%}",
            "prob_normal": f"{probabilities[0]:.2%}",
            "prob_distressed": f"{probabilities[1]:.2%}"
        }
        
        results.append(result)
    
    # Convert to DataFrame for nice display
    results_df = pd.DataFrame(results)
    
    # Print results in a table
    print("\nTest Results:")
    print("------------")
    print(tabulate(results_df, headers="keys", tablefmt="grid", showindex=False))
    
    return results_df

def main():
    """Main function to test the model."""
    print("Mental Health Predictor - Model Testing")
    print("======================================")
    
    # Load the model
    model = load_model()
    if model is None:
        return 1
    
    # Print model info
    print_model_info()
    
    # Define test examples
    example_texts = [
        # Normal examples
        "I had a great day today and I'm feeling really good about everything.",
        "Just finished a productive meeting with my team. We made good progress on our project.",
        "I'm looking forward to the weekend. Planning to relax and spend time with friends.",
        
        # Ambiguous examples
        "I'm feeling tired today after a long day at work.",
        "I had an argument with my friend, but we'll work it out.",
        "The weather is gloomy today, matching my mood.",
        
        # Distressed examples
        "I feel completely hopeless and don't see any point in continuing.",
        "I can't stop crying and I don't know why. Everything feels overwhelming.",
        "I'm constantly anxious and can't sleep. I feel like I'm falling apart."
    ]
    
    # Test the model with examples
    results = test_examples(model, example_texts)
    
    # Save results to CSV
    if results is not None:
        results.to_csv("app/model/test_results.csv", index=False)
        print("\nTest results saved to app/model/test_results.csv")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())