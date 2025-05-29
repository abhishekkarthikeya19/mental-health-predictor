#!/usr/bin/env python
"""
Script to run the basic models from advanced_model.py without the transformer model
"""
import os
import sys
import pandas as pd
import numpy as np
from app.advanced_model import ModelTrainer

def main():
    # Sample data for demonstration
    sample_texts = [
        "I've been feeling really down lately and can't seem to enjoy anything.",
        "I had a great day today, everything went well!",
        "I'm so stressed about work, I can't sleep at night.",
        "Just finished a wonderful book, it was very uplifting.",
        "I feel worthless and think about ending it all sometimes.",
        "I'm excited about my upcoming vacation.",
        "I can't stop crying and I don't know why.",
        "Just got promoted at work, feeling accomplished!",
        "I haven't left my room in days, I just can't face people.",
        "Spent quality time with family today, feeling blessed."
    ]
    
    # 1 for negative mental health indicators, 0 for positive
    sample_labels = [1, 0, 1, 0, 1, 0, 1, 0, 1, 0]
    
    # Initialize the model trainer
    trainer = ModelTrainer()
    
    # Prepare the data
    X_train, X_test, y_train, y_test, feature_extractor = trainer.prepare_data(
        sample_texts, sample_labels, test_size=0.2
    )
    
    # Train individual models
    print("\n=== Training Random Forest ===")
    rf_model, rf_metrics = trainer.train_random_forest(X_train, y_train, X_test, y_test)
    
    print("\n=== Training Gradient Boosting ===")
    gb_model, gb_metrics = trainer.train_gradient_boosting(X_train, y_train, X_test, y_test)
    
    print("\n=== Training SVM ===")
    svm_model, svm_metrics = trainer.train_svm(X_train, y_train, X_test, y_test)
    
    print("\n=== Training Neural Network ===")
    nn_model, nn_metrics = trainer.train_neural_network(X_train, y_train, X_test, y_test)
    
    # Combine models into an ensemble
    print("\n=== Training Ensemble Model ===")
    models = {
        "Random Forest": rf_model,
        "Gradient Boosting": gb_model,
        "SVM": svm_model,
        "Neural Network": nn_model
    }
    
    ensemble_model, ensemble_metrics = trainer.train_ensemble_model(models, X_test, y_test)
    
    # Print results
    print("\n=== Model Performance Summary ===")
    all_metrics = {
        "Random Forest": rf_metrics,
        "Gradient Boosting": gb_metrics,
        "SVM": svm_metrics,
        "Neural Network": nn_metrics,
        "Ensemble": ensemble_metrics
    }
    
    for model_name, metrics in all_metrics.items():
        print(f"\n{model_name}:")
        for metric_name, value in metrics.items():
            print(f"  {metric_name}: {value:.4f}")
    
    # Test prediction on new text
    test_text = "I've been feeling sad and lonely lately"
    print(f"\n=== Prediction for: '{test_text}' ===")
    
    # Transform the text using the feature extractor
    test_features = feature_extractor.transform([test_text])
    
    # Make predictions with each model
    for model_name, model in models.items():
        pred = model.predict(test_features)[0]
        prob = model.predict_proba(test_features)[0][1]
        print(f"{model_name}: {'Negative' if pred == 1 else 'Positive'} mental health ({prob:.2f} probability of negative)")
    
    # Ensemble prediction
    ensemble_pred = ensemble_model.predict(test_features)[0]
    ensemble_prob = ensemble_model.predict_proba(test_features)[0][1]
    print(f"Ensemble: {'Negative' if ensemble_pred == 1 else 'Positive'} mental health ({ensemble_prob:.2f} probability of negative)")

if __name__ == "__main__":
    print("Starting Basic Models Training Demo")
    main()
    print("\nTraining and evaluation completed!")