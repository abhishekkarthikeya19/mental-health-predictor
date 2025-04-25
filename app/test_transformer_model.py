"""
Script to test the transformer-based mental health prediction model.
"""
import os
import sys
import logging
import pandas as pd
import numpy as np
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_model():
    """Test the transformer model with sample inputs and visualize results."""
    try:
        # Import the model loader
        from load_transformer_model import load_transformer_model
        
        # Load the model
        logger.info("Loading transformer model...")
        model = load_transformer_model()
        logger.info("Model loaded successfully")
        
        # Create test data
        test_data = pd.DataFrame({
            "text_input": [
                # Distressed examples
                "I feel so hopeless and empty",
                "I can't stop crying and I don't know why",
                "I'm having thoughts about ending it all",
                "I feel like a burden to everyone",
                "I'm constantly anxious and can't relax",
                "I haven't slept well in weeks",
                "I've lost interest in everything",
                "I feel overwhelmed by simple tasks",
                "My mind is filled with negative thoughts",
                "I feel like I'm drowning in my thoughts",
                
                # Normal examples
                "I'm having a good day today",
                "Just finished a productive meeting",
                "Feeling energized after exercise",
                "Looking forward to the weekend",
                "Had a nice chat with friends",
                "Enjoying this beautiful weather",
                "I accomplished my goals for today",
                "Learning new things is exciting",
                "Taking time for self-care today",
                "Feeling grateful for my support system"
            ],
            "expected_label": [
                # Labels for distressed examples
                1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                # Labels for normal examples
                0, 0, 0, 0, 0, 0, 0, 0, 0, 0
            ]
        })
        
        # Make predictions
        logger.info("Making predictions...")
        predictions = model.predict(test_data["text_input"])
        probabilities = model.predict_proba(test_data["text_input"])
        
        # Add predictions to the dataframe
        test_data["prediction"] = predictions
        test_data["confidence"] = [prob[pred] for prob, pred in zip(probabilities, predictions)]
        
        # Calculate accuracy
        accuracy = accuracy_score(test_data["expected_label"], test_data["prediction"])
        logger.info(f"Test accuracy: {accuracy:.4f}")
        
        # Print classification report
        print("\nClassification Report:")
        print(classification_report(test_data["expected_label"], test_data["prediction"]))
        
        # Print confusion matrix
        cm = confusion_matrix(test_data["expected_label"], test_data["prediction"])
        print("\nConfusion Matrix:")
        print(cm)
        
        # Visualize confusion matrix
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                    xticklabels=['Normal', 'Distressed'], 
                    yticklabels=['Normal', 'Distressed'])
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.title('Confusion Matrix')
        
        # Save the plot
        os.makedirs("app/results", exist_ok=True)
        plt.savefig("app/results/confusion_matrix.png")
        logger.info("Confusion matrix saved to app/results/confusion_matrix.png")
        
        # Print detailed results
        print("\nDetailed Results:")
        for i, row in test_data.iterrows():
            print(f"Text: {row['text_input']}")
            print(f"Expected: {'Distressed' if row['expected_label'] == 1 else 'Normal'}")
            print(f"Predicted: {'Distressed' if row['prediction'] == 1 else 'Normal'}")
            print(f"Confidence: {row['confidence']:.4f}")
            print()
        
        # Save results to CSV
        test_data.to_csv("app/results/test_results.csv", index=False)
        logger.info("Test results saved to app/results/test_results.csv")
        
        return True
    
    except Exception as e:
        logger.error(f"Error testing model: {str(e)}", exc_info=True)
        return False

if __name__ == "__main__":
    test_model()