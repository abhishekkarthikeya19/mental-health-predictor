from app.advanced_model import ModelTrainer
import pandas as pd
import numpy as np

def main():
    # Sample data for demonstration
    # In a real scenario, you would load your actual data
    sample_texts = [
        "I've been feeling really down lately and can't seem to enjoy anything.",
        "I had a great day today, everything went well!",
        "I'm so stressed about work, I can't sleep at night.",
        "Just finished a wonderful book, it was very uplifting.",
        "I feel worthless and think about ending it all sometimes."
    ]
    
    # 1 for negative mental health indicators, 0 for positive
    sample_labels = [1, 0, 1, 0, 1]
    
    # Initialize the model trainer
    trainer = ModelTrainer()
    
    # Train all models
    print("Training models on sample data...")
    results = trainer.train_all_models(sample_texts, sample_labels)
    
    # Print results
    print("\nModel Performance Summary:")
    for model_name, metrics in results.items():
        print(f"\n{model_name}:")
        for metric_name, value in metrics.items():
            print(f"  {metric_name}: {value:.4f}")

if __name__ == "__main__":
    print("Starting Advanced Model Training Demo")
    main()
    print("Training completed!")