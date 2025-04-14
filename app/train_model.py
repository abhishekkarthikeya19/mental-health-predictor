import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
import joblib

data = pd.DataFrame({
    "text_input": ["I feel sad", "Life is good", "I'm depressed", "Feeling great"],
    "label": [1, 0, 1, 0]
})

pipeline = Pipeline([
    ("vectorizer", CountVectorizer()),
    ("model", LogisticRegression())
])

pipeline.fit(data["text_input"], data["label"])
joblib.dump(pipeline, "app/model/mental_health_model.pkl")
