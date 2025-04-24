import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report, accuracy_score
import joblib
import os
import numpy as np

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
        "Feeling content with where I am in life"
    ],
    "label": [
        # Labels for distressed examples
        1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
        # Labels for normal examples
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
    ]
})

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    data["text_input"], data["label"], test_size=0.2, random_state=42, stratify=data["label"]
)

# Create a more sophisticated pipeline
pipeline = Pipeline([
    ("vectorizer", TfidfVectorizer(max_features=1000, ngram_range=(1, 2))),
    ("model", RandomForestClassifier(n_estimators=100, random_state=42))
])

# Train the model
pipeline.fit(X_train, y_train)

# Evaluate the model
print("Model Evaluation:")
print("-----------------")

# Cross-validation
cv_scores = cross_val_score(pipeline, X_train, y_train, cv=5)
print(f"Cross-validation accuracy: {np.mean(cv_scores):.4f} (±{np.std(cv_scores):.4f})")

# Test set evaluation
y_pred = pipeline.predict(X_test)
print(f"Test set accuracy: {accuracy_score(y_test, y_pred):.4f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Ensure the model directory exists
os.makedirs("app/model", exist_ok=True)

# Save the model
joblib.dump(pipeline, "app/model/mental_health_model.pkl")
print("Model saved to app/model/mental_health_model.pkl")
