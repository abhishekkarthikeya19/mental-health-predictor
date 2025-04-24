from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.pipeline import Pipeline
import joblib
import pandas as pd

# Dummy training data with text inputs
data = pd.DataFrame({
    "text_input": ["I feel sad", "Life is good", "I'm depressed", "Feeling great"],
    "label": [1, 0, 1, 0]
})

# Create a pipeline with text vectorization and model
pipeline = Pipeline([
    ("vectorizer", CountVectorizer()),
    ("model", LogisticRegression())
])

# Train the model
pipeline.fit(data["text_input"], data["label"])

# Save the model
joblib.dump(pipeline, 'app/model/mental_health_model.pkl')
