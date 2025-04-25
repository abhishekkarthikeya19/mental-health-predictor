import pandas as pd
import numpy as np
import os
import torch
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
from transformers import pipeline
import joblib
from datasets import Dataset
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info(f"Using device: {device}")

# Create a more comprehensive dataset
# 0 = Normal, 1 = Distressed
data = pd.DataFrame({
    "text_input": [
        # Distressed examples (label 1)
        "I feel sad and empty inside",
        "I'm so depressed I can barely get out of bed",
        "Nothing brings me joy anymore",
        "I feel worthless and hopeless about the future",
        "I can't stop crying and I don't know why",
        "I'm having thoughts about ending it all",
        "I feel like a burden to everyone around me",
        "I'm constantly anxious and can't relax",
        "I haven't slept well in weeks",
        "I've lost interest in activities I used to enjoy",
        "I feel overwhelmed by simple daily tasks",
        "My mind is filled with negative thoughts I can't control",
        "I feel like I'm drowning in my own thoughts",
        "Everything feels like too much effort",
        "I'm constantly tired no matter how much I sleep",
        "I don't see any point in continuing like this",
        "I feel like nobody understands what I'm going through",
        "I'm struggling to find any reason to keep going",
        "My anxiety is making it hard to function normally",
        "I feel trapped in my own mind with no way out",
        
        # Normal examples (label 0)
        "Life is good, I'm enjoying my day",
        "I had a productive meeting at work today",
        "Feeling great after my workout",
        "I'm excited about my upcoming vacation",
        "Just finished a good book and feeling satisfied",
        "Had a nice dinner with friends tonight",
        "The weather is beautiful today",
        "I accomplished all my tasks for the day",
        "Looking forward to the weekend",
        "I learned something new today and it was interesting",
        "Feeling motivated to start this new project",
        "Had a good conversation with my family",
        "I'm proud of what I achieved today",
        "Taking time to relax and recharge",
        "Feeling content with where I am in life",
        "I'm grateful for the support of my friends",
        "Today was challenging but I handled it well",
        "I'm making progress on my personal goals",
        "I enjoyed spending time in nature today",
        "I'm feeling optimistic about the future"
    ],
    "label": [
        # Labels for distressed examples
        1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
        # Labels for normal examples
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
    ]
})

# Split data into training and testing sets
train_df, test_df = train_test_split(
    data, test_size=0.2, random_state=42, stratify=data["label"]
)

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

# Define training arguments
training_args = TrainingArguments(
    output_dir="./results",
    num_train_epochs=3,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    warmup_steps=500,
    weight_decay=0.01,
    logging_dir="./logs",
    logging_steps=10,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model="accuracy",
    greater_is_better=True,
)

# Define compute_metrics function
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return {
        "accuracy": accuracy_score(labels, predictions),
        "classification_report": classification_report(labels, predictions)
    }

# Initialize Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_tokenized,
    eval_dataset=test_tokenized,
    compute_metrics=compute_metrics,
)

# Train the model
logger.info("Starting model training...")
trainer.train()

# Evaluate the model
logger.info("Evaluating model...")
eval_results = trainer.evaluate()
logger.info(f"Evaluation results: {eval_results}")

# Create a text classification pipeline
classifier = pipeline(
    "text-classification", 
    model=model, 
    tokenizer=tokenizer,
    device=0 if torch.cuda.is_available() else -1
)

# Create a wrapper class that mimics the scikit-learn API
class TransformerClassifier:
    def __init__(self, pipeline):
        self.pipeline = pipeline
        
    def predict(self, texts):
        results = self.pipeline(list(texts))
        # Convert label to int (0 for normal, 1 for distressed)
        return np.array([1 if result['label'] == 'LABEL_1' else 0 for result in results])
    
    def predict_proba(self, texts):
        results = self.pipeline(list(texts))
        # Create probability arrays [prob_normal, prob_distressed]
        probs = []
        for result in results:
            if result['label'] == 'LABEL_1':  # distressed
                probs.append([1 - result['score'], result['score']])
            else:  # normal
                probs.append([result['score'], 1 - result['score']])
        return np.array(probs)

# Create the wrapper
model_wrapper = TransformerClassifier(classifier)

# Test the wrapper
test_texts = test_df["text_input"].tolist()
test_labels = test_df["label"].tolist()

# Make predictions
y_pred = model_wrapper.predict(test_texts)
y_proba = model_wrapper.predict_proba(test_texts)

# Print evaluation metrics
print("\nModel Evaluation:")
print("-----------------")
print(f"Test set accuracy: {accuracy_score(test_labels, y_pred):.4f}")
print("\nClassification Report:")
print(classification_report(test_labels, y_pred))

# Ensure the model directory exists
os.makedirs("app/model", exist_ok=True)

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
