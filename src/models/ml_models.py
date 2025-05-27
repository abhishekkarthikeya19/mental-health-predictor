import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from sklearn.preprocessing import StandardScaler
import joblib
import logging

class MentalHealthClassifier:
    def __init__(self):
        self.models = {
            'random_forest': RandomForestClassifier(random_state=42),
            'svm': SVC(probability=True, random_state=42),
            'neural_network': MLPClassifier(random_state=42, max_iter=1000)
        }
        self.scaler = StandardScaler()
        self.best_model = None
        self.logger = logging.getLogger(__name__)
    
    def prepare_features(self, df: pd.DataFrame, tfidf_matrix: np.ndarray) -> np.ndarray:
        """Combine all features into feature matrix"""
        feature_columns = [
            'sentiment_compound', 'sentiment_positive', 'sentiment_negative',
            'text_length', 'word_count', 'avg_word_length',
            'first_person_ratio', 'depression_ratio', 'anxiety_ratio'
        ]
        
        structured_features = df[feature_columns].values
        
        # Combine with TF-IDF features
        combined_features = np.hstack([structured_features, tfidf_matrix.toarray()])
        
        return combined_features
    
    def train_models(self, X: np.ndarray, y: np.ndarray):
        """Train and evaluate multiple models"""
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        results = {}
        
        for name, model in self.models.items():
            self.logger.info(f"Training {name}...")
            
            # Train model
            model.fit(X_train_scaled, y_train)
            
            # Evaluate
            train_score = model.score(X_train_scaled, y_train)
            test_score = model.score(X_test_scaled, y_test)
            cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=5)
            
            results[name] = {
                'model': model,
                'train_score': train_score,
                'test_score': test_score,
                'cv_mean': cv_scores.mean(),
                'cv_std': cv_scores.std()
            }
            
            self.logger.info(f"{name} - Test Score: {test_score:.3f}, CV: {cv_scores.mean():.3f} (+/- {cv_scores.std() * 2:.3f})")
        
        # Select best model
        best_model_name = max(results.keys(), key=lambda k: results[k]['cv_mean'])
        self.best_model = results[best_model_name]['model']
        
        return results