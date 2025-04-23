from sklearn.linear_model import LogisticRegression
import joblib
import pandas as pd

# Dummy training data
X = pd.DataFrame({'text_input': [0, 1, 2, 3]})
y = [0, 1, 0, 1]

model = LogisticRegression()
model.fit(X, y)

# Save the model
joblib.dump(model, 'model/model.joblib')
