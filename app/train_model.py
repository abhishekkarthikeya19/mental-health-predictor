import pandas as pd
import numpy as np
import os
import torch
import time
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
from transformers import pipeline
import joblib
from datasets import Dataset, load_dataset
import logging
import sys

# Add the app directory to the path so we can import data_preparation
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from app.data_preparation import load_and_prepare_datasets

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info(f"Using device: {device}")

# Check if preprocessed data exists, otherwise create it
if os.path.exists("app/data/train_data.csv") and os.path.exists("app/data/test_data.csv"):
    logger.info("Loading preprocessed datasets...")
    train_df = pd.read_csv("app/data/train_data.csv")
    test_df = pd.read_csv("app/data/test_data.csv")
    logger.info(f"Loaded {len(train_df)} training samples and {len(test_df)} test samples")
else:
    logger.info("Preprocessed datasets not found. Preparing datasets...")
    train_df, test_df = load_and_prepare_datasets()

# Display dataset statistics
logger.info(f"Training set size: {len(train_df)}")
logger.info(f"Test set size: {len(test_df)}")
logger.info(f"Training class distribution: {train_df['label'].value_counts(normalize=True)}")
logger.info(f"Test class distribution: {test_df['label'].value_counts(normalize=True)}")

# Convert to Hugging Face datasets
train_dataset = Dataset.from_pandas(train_df)
test_dataset = Dataset.from_pandas(test_df)

# Load pre-trained model and tokenizer
# Using DistilBERT which is smaller and faster than BERT but still powerful
model_name = "distilbert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Tokenize function
def tokenize_function(examples):
    return tokenizer(examples["text_input"], padding="max_length", truncation=True, max_length=128)

# Tokenize datasets
train_tokenized = train_dataset.map(tokenize_function, batched=True)
test_tokenized = test_dataset.map(tokenize_function, batched=True)

# Load pre-trained model
model = AutoModelForSequenceClassification.from_pretrained(
    model_name, 
    num_labels=2,
    id2label={0: "normal", 1: "distressed"},
    label2id={"normal": 0, "distressed": 1}
)

# Define training arguments with improved parameters
training_args = TrainingArguments(
    output_dir="./results",
    num_train_epochs=5,  # Increased from 3 to 5 for better learning
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    warmup_ratio=0.1,  # Use ratio instead of steps for better adaptation to dataset size
    weight_decay=0.01,
    logging_dir="./logs",
    logging_steps=10,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model="f1",  # Changed to F1 score which is better for imbalanced datasets
    greater_is_better=True,
    learning_rate=2e-5,  # Explicitly set learning rate
    fp16=torch.cuda.is_available(),  # Use mixed precision training if GPU is available
    gradient_accumulation_steps=2,  # Accumulate gradients to simulate larger batch sizes
    report_to="none",  # Disable reporting to avoid dependencies
)

# Enhanced compute_metrics function with more metrics
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    
    # Calculate accuracy
    acc = accuracy_score(labels, predictions)
    
    # Calculate confusion matrix
    tn, fp, fn, tp = confusion_matrix(labels, predictions, labels=[0, 1]).ravel()
    
    # Calculate precision, recall, and F1 score
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    
    # Get detailed classification report as string
    report = classification_report(labels, predictions)
    
    return {
        "accuracy": acc,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "true_negatives": tn,
        "false_positives": fp,
        "false_negatives": fn,
        "true_positives": tp,
        "classification_report": report
    }

# Initialize Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_tokenized,
    eval_dataset=test_tokenized,
    compute_metrics=compute_metrics,
)

# Record start time for training
start_time = time.time()

# Train the model
logger.info("Starting model training...")
trainer.train()

# Calculate training time
training_time = time.time() - start_time
logger.info(f"Training completed in {training_time:.2f} seconds ({training_time/60:.2f} minutes)")

# Evaluate the model
logger.info("Evaluating model...")
eval_results = trainer.evaluate()

# Log detailed evaluation results
for metric, value in eval_results.items():
    if isinstance(value, str):
        logger.info(f"{metric}:\n{value}")
    else:
        logger.info(f"{metric}: {value:.4f}")

# Create a text classification pipeline
classifier = pipeline(
    "text-classification", 
    model=model, 
    tokenizer=tokenizer,
    device=0 if torch.cuda.is_available() else -1
)

# Create a wrapper class that mimics the scikit-learn API with enhanced functionality
class TransformerClassifier:
    def __init__(self, pipeline, model_name=None, training_metrics=None):
        self.pipeline = pipeline
        self.model_name = model_name or "distilbert-base-uncased"
        self.training_metrics = training_metrics or {}
        self.version = "1.0.0"
        self.creation_date = time.strftime("%Y-%m-%d %H:%M:%S")
        
    def predict(self, texts):
        """Predict class labels for the input texts."""
        results = self.pipeline(list(texts))
        # Convert label to int (0 for normal, 1 for distressed)
        return np.array([1 if result['label'] == 'LABEL_1' else 0 for result in results])
    
    def predict_proba(self, texts):
        """Predict class probabilities for the input texts."""
        results = self.pipeline(list(texts))
        # Create probability arrays [prob_normal, prob_distressed]
        probs = []
        for result in results:
            if result['label'] == 'LABEL_1':  # distressed
                probs.append([1 - result['score'], result['score']])
            else:  # normal
                probs.append([result['score'], 1 - result['score']])
        return np.array(probs)
    
    def get_model_info(self):
        """Return information about the model."""
        return {
            "model_name": self.model_name,
            "version": self.version,
            "creation_date": self.creation_date,
            "training_metrics": self.training_metrics
        }

# Create the wrapper with additional metadata
model_wrapper = TransformerClassifier(
    classifier, 
    model_name=model_name,
    training_metrics={
        "accuracy": eval_results.get("eval_accuracy", 0),
        "f1": eval_results.get("eval_f1", 0),
        "precision": eval_results.get("eval_precision", 0),
        "recall": eval_results.get("eval_recall", 0)
    }
)

# Test the wrapper on the test set
logger.info("Testing model on test set...")
test_texts = test_df["text_input"].tolist()
test_labels = test_df["label"].tolist()

# Make predictions
y_pred = model_wrapper.predict(test_texts)
y_proba = model_wrapper.predict_proba(test_texts)

# Calculate metrics
accuracy = accuracy_score(test_labels, y_pred)
conf_matrix = confusion_matrix(test_labels, y_pred)
class_report = classification_report(test_labels, y_pred)

# Print evaluation metrics
print("\nModel Evaluation:")
print("-----------------")
print(f"Test set accuracy: {accuracy:.4f}")
print("\nConfusion Matrix:")
print(conf_matrix)
print("\nClassification Report:")
print(class_report)

# Save detailed evaluation results
eval_results_df = pd.DataFrame({
    "text": test_texts,
    "true_label": test_labels,
    "predicted_label": y_pred,
    "prob_normal": y_proba[:, 0],
    "prob_distressed": y_proba[:, 1]
})

# Save misclassified examples for analysis
misclassified = eval_results_df[eval_results_df["true_label"] != eval_results_df["predicted_label"]]
logger.info(f"Number of misclassified examples: {len(misclassified)}")

# Ensure the model directory exists
os.makedirs("app/model", exist_ok=True)

# Save evaluation results
eval_results_df.to_csv("app/model/evaluation_results.csv", index=False)
misclassified.to_csv("app/model/misclassified_examples.csv", index=False)

# Save the model wrapper
logger.info("Saving model...")
joblib.dump(model_wrapper, "app/model/mental_health_model.pkl")
print("Model saved to app/model/mental_health_model.pkl")

# Also save the model and tokenizer for future use
model_dir = "app/model/transformer_model"
os.makedirs(model_dir, exist_ok=True)
model.save_pretrained(model_dir)
tokenizer.save_pretrained(model_dir)
print(f"Transformer model and tokenizer saved to {model_dir}")

# Save model metadata
model_info = {
    "model_name": model_name,
    "version": "1.0.0",
    "creation_date": time.strftime("%Y-%m-%d %H:%M:%S"),
    "training_dataset_size": len(train_df),
    "test_dataset_size": len(test_df),
    "training_time_seconds": training_time,
    "accuracy": accuracy,
    "confusion_matrix": conf_matrix.tolist(),
    "classification_report": class_report
}

# Save model info as JSON
import json
with open("app/model/model_info.json", "w") as f:
    json.dump(model_info, f, indent=2)
print("Model metadata saved to app/model/model_info.json")
