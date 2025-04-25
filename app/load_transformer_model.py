"""
Utility module for loading the transformer-based mental health prediction model.
"""
import os
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
import numpy as np
import logging

logger = logging.getLogger(__name__)

class TransformerClassifier:
    """
    A wrapper class for transformer models that provides a scikit-learn compatible API.
    """
    def __init__(self, model_path=None, tokenizer_path=None):
        """
        Initialize the classifier with a pre-trained model and tokenizer.
        
        Args:
            model_path: Path to the saved model
            tokenizer_path: Path to the saved tokenizer
        """
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {self.device}")
        
        # If paths are provided, load from them
        if model_path and tokenizer_path and os.path.exists(model_path) and os.path.exists(tokenizer_path):
            logger.info(f"Loading model from {model_path}")
            self.model = AutoModelForSequenceClassification.from_pretrained(model_path)
            self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
        else:
            # Default to distilbert if no paths provided or paths don't exist
            logger.info("Loading default distilbert model")
            self.model = AutoModelForSequenceClassification.from_pretrained(
                "distilbert-base-uncased", 
                num_labels=2,
                id2label={0: "normal", 1: "distressed"},
                label2id={"normal": 0, "distressed": 1}
            )
            self.tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
        
        # Move model to the appropriate device
        self.model.to(self.device)
        
        # Create the pipeline
        self.nlp = pipeline(
            "text-classification", 
            model=self.model, 
            tokenizer=self.tokenizer,
            device=0 if torch.cuda.is_available() else -1
        )
    
    def predict(self, texts):
        """
        Predict the class for each text.
        
        Args:
            texts: List of text strings to classify
            
        Returns:
            numpy.ndarray: Array of predictions (0 for normal, 1 for distressed)
        """
        results = self.nlp(list(texts))
        # Convert label to int (0 for normal, 1 for distressed)
        return np.array([1 if result['label'] == 'LABEL_1' else 0 for result in results])
    
    def predict_proba(self, texts):
        """
        Predict class probabilities for each text.
        
        Args:
            texts: List of text strings to classify
            
        Returns:
            numpy.ndarray: Array of shape (n_samples, 2) with probabilities for each class
        """
        results = self.nlp(list(texts))
        # Create probability arrays [prob_normal, prob_distressed]
        probs = []
        for result in results:
            if result['label'] == 'LABEL_1':  # distressed
                probs.append([1 - result['score'], result['score']])
            else:  # normal
                probs.append([result['score'], 1 - result['score']])
        return np.array(probs)

def load_transformer_model():
    """
    Load the transformer model from the saved directory.
    
    Returns:
        TransformerClassifier: The loaded model
    """
    model_dir = "app/model/transformer_model"
    model_path = os.path.join(model_dir, "")
    tokenizer_path = os.path.join(model_dir, "")
    
    if os.path.exists(model_dir):
        return TransformerClassifier(model_path=model_dir, tokenizer_path=model_dir)
    else:
        logger.warning(f"Model directory {model_dir} not found. Loading default model.")
        return TransformerClassifier()

if __name__ == "__main__":
    # Test the model loading
    model = load_transformer_model()
    test_texts = [
        "I feel happy today",
        "I'm feeling very depressed and can't get out of bed"
    ]
    predictions = model.predict(test_texts)
    probabilities = model.predict_proba(test_texts)
    
    for i, text in enumerate(test_texts):
        print(f"Text: {text}")
        print(f"Prediction: {'Distressed' if predictions[i] == 1 else 'Normal'}")
        print(f"Confidence: {probabilities[i][predictions[i]]:.4f}")
        print()