"""
Advanced model training module for mental health prediction.
This module implements more sophisticated machine learning models and techniques.
"""
import pandas as pd
import numpy as np
import os
import torch
import joblib
import logging
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc, precision_recall_curve
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
from transformers import pipeline
from datasets import Dataset
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from feature_extraction import FeatureExtractor

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ModelTrainer:
    """
    Class for training and evaluating machine learning models for mental health prediction.
    """
    def __init__(self, model_dir="app/model"):
        """
        Initialize the model trainer.
        
        Args:
            model_dir (str): Directory to save trained models
        """
        self.model_dir = model_dir
        os.makedirs(model_dir, exist_ok=True)
        self.feature_extractor = FeatureExtractor(use_transformers=False)
        
        # Set device for PyTorch
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {self.device}")
    
    def prepare_data(self, texts, labels, test_size=0.2, extract_features=True):
        """
        Prepare data for model training.
        
        Args:
            texts (list): List of text strings
            labels (list): List of labels (0 for normal, 1 for distressed)
            test_size (float): Proportion of data to use for testing
            extract_features (bool): Whether to extract features from texts
            
        Returns:
            tuple: (X_train, X_test, y_train, y_test) - training and testing data
        """
        if extract_features:
            # Extract features from texts
            features = self.feature_extractor.extract_all_features(texts)
            
            # Split data into training and testing sets
            X_train, X_test, y_train, y_test = train_test_split(
                features, labels, test_size=test_size, random_state=42, stratify=labels
            )
        else:
            # Use texts directly (for transformer models)
            X_train, X_test, y_train, y_test = train_test_split(
                texts, labels, test_size=test_size, random_state=42, stratify=labels
            )
        
        return X_train, X_test, y_train, y_test
    
    def train_random_forest(self, X_train, y_train, X_test, y_test):
        """
        Train a Random Forest classifier.
        
        Args:
            X_train: Training features
            y_train: Training labels
            X_test: Testing features
            y_test: Testing labels
            
        Returns:
            tuple: (trained_model, evaluation_metrics)
        """
        logger.info("Training Random Forest classifier...")
        
        # Create a pipeline with preprocessing and model
        numeric_features = X_train.select_dtypes(include=['int64', 'float64']).columns
        
        numeric_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler())
        ])
        
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', numeric_transformer, numeric_features)
            ],
            remainder='passthrough'
        )
        
        # Define the model pipeline
        model = Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('classifier', RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                min_samples_split=5,
                min_samples_leaf=2,
                random_state=42,
                class_weight='balanced'
            ))
        ])
        
        # Train the model
        model.fit(X_train, y_train)
        
        # Evaluate the model
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1]
        
        # Calculate evaluation metrics
        metrics = {
            'accuracy': (y_pred == y_test).mean(),
            'classification_report': classification_report(y_test, y_pred, output_dict=True),
            'confusion_matrix': confusion_matrix(y_test, y_pred).tolist(),
            'roc_auc': auc(roc_curve(y_test, y_proba)[0], roc_curve(y_test, y_proba)[1])
        }
        
        logger.info(f"Random Forest accuracy: {metrics['accuracy']:.4f}")
        logger.info(f"Random Forest ROC AUC: {metrics['roc_auc']:.4f}")
        
        # Save the model
        model_path = os.path.join(self.model_dir, "random_forest_model.pkl")
        joblib.dump(model, model_path)
        logger.info(f"Random Forest model saved to {model_path}")
        
        return model, metrics
    
    def train_gradient_boosting(self, X_train, y_train, X_test, y_test):
        """
        Train a Gradient Boosting classifier.
        
        Args:
            X_train: Training features
            y_train: Training labels
            X_test: Testing features
            y_test: Testing labels
            
        Returns:
            tuple: (trained_model, evaluation_metrics)
        """
        logger.info("Training Gradient Boosting classifier...")
        
        # Create a pipeline with preprocessing and model
        numeric_features = X_train.select_dtypes(include=['int64', 'float64']).columns
        
        numeric_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler())
        ])
        
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', numeric_transformer, numeric_features)
            ],
            remainder='passthrough'
        )
        
        # Define the model pipeline
        model = Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('classifier', GradientBoostingClassifier(
                n_estimators=100,
                learning_rate=0.1,
                max_depth=5,
                random_state=42
            ))
        ])
        
        # Train the model
        model.fit(X_train, y_train)
        
        # Evaluate the model
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1]
        
        # Calculate evaluation metrics
        metrics = {
            'accuracy': (y_pred == y_test).mean(),
            'classification_report': classification_report(y_test, y_pred, output_dict=True),
            'confusion_matrix': confusion_matrix(y_test, y_pred).tolist(),
            'roc_auc': auc(roc_curve(y_test, y_proba)[0], roc_curve(y_test, y_proba)[1])
        }
        
        logger.info(f"Gradient Boosting accuracy: {metrics['accuracy']:.4f}")
        logger.info(f"Gradient Boosting ROC AUC: {metrics['roc_auc']:.4f}")
        
        # Save the model
        model_path = os.path.join(self.model_dir, "gradient_boosting_model.pkl")
        joblib.dump(model, model_path)
        logger.info(f"Gradient Boosting model saved to {model_path}")
        
        return model, metrics
    
    def train_svm(self, X_train, y_train, X_test, y_test):
        """
        Train a Support Vector Machine classifier.
        
        Args:
            X_train: Training features
            y_train: Training labels
            X_test: Testing features
            y_test: Testing labels
            
        Returns:
            tuple: (trained_model, evaluation_metrics)
        """
        logger.info("Training SVM classifier...")
        
        # Create a pipeline with preprocessing and model
        numeric_features = X_train.select_dtypes(include=['int64', 'float64']).columns
        
        numeric_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler())
        ])
        
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', numeric_transformer, numeric_features)
            ],
            remainder='passthrough'
        )
        
        # Define the model pipeline
        model = Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('classifier', SVC(
                C=1.0,
                kernel='rbf',
                probability=True,
                class_weight='balanced',
                random_state=42
            ))
        ])
        
        # Train the model
        model.fit(X_train, y_train)
        
        # Evaluate the model
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1]
        
        # Calculate evaluation metrics
        metrics = {
            'accuracy': (y_pred == y_test).mean(),
            'classification_report': classification_report(y_test, y_pred, output_dict=True),
            'confusion_matrix': confusion_matrix(y_test, y_pred).tolist(),
            'roc_auc': auc(roc_curve(y_test, y_proba)[0], roc_curve(y_test, y_proba)[1])
        }
        
        logger.info(f"SVM accuracy: {metrics['accuracy']:.4f}")
        logger.info(f"SVM ROC AUC: {metrics['roc_auc']:.4f}")
        
        # Save the model
        model_path = os.path.join(self.model_dir, "svm_model.pkl")
        joblib.dump(model, model_path)
        logger.info(f"SVM model saved to {model_path}")
        
        return model, metrics
    
    def train_neural_network(self, X_train, y_train, X_test, y_test):
        """
        Train a Neural Network classifier.
        
        Args:
            X_train: Training features
            y_train: Training labels
            X_test: Testing features
            y_test: Testing labels
            
        Returns:
            tuple: (trained_model, evaluation_metrics)
        """
        logger.info("Training Neural Network classifier...")
        
        # Create a pipeline with preprocessing and model
        numeric_features = X_train.select_dtypes(include=['int64', 'float64']).columns
        
        numeric_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler())
        ])
        
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', numeric_transformer, numeric_features)
            ],
            remainder='passthrough'
        )
        
        # Define the model pipeline
        model = Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('classifier', MLPClassifier(
                hidden_layer_sizes=(100, 50),
                activation='relu',
                solver='adam',
                alpha=0.0001,
                batch_size='auto',
                learning_rate='adaptive',
                max_iter=500,
                random_state=42
            ))
        ])
        
        # Train the model
        model.fit(X_train, y_train)
        
        # Evaluate the model
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1]
        
        # Calculate evaluation metrics
        metrics = {
            'accuracy': (y_pred == y_test).mean(),
            'classification_report': classification_report(y_test, y_pred, output_dict=True),
            'confusion_matrix': confusion_matrix(y_test, y_pred).tolist(),
            'roc_auc': auc(roc_curve(y_test, y_proba)[0], roc_curve(y_test, y_proba)[1])
        }
        
        logger.info(f"Neural Network accuracy: {metrics['accuracy']:.4f}")
        logger.info(f"Neural Network ROC AUC: {metrics['roc_auc']:.4f}")
        
        # Save the model
        model_path = os.path.join(self.model_dir, "neural_network_model.pkl")
        joblib.dump(model, model_path)
        logger.info(f"Neural Network model saved to {model_path}")
        
        return model, metrics
    
    def train_transformer_model(self, texts, labels, test_texts=None, test_labels=None):
        """
        Train a transformer-based model for text classification.
        
        Args:
            texts (list): List of training text strings
            labels (list): List of training labels (0 for normal, 1 for distressed)
            test_texts (list): List of testing text strings
            test_labels (list): List of testing labels
            
        Returns:
            tuple: (trained_model, evaluation_metrics)
        """
        logger.info("Training transformer model...")
        
        # If test data is not provided, split the training data
        if test_texts is None or test_labels is None:
            train_texts, test_texts, train_labels, test_labels = train_test_split(
                texts, labels, test_size=0.2, random_state=42, stratify=labels
            )
        else:
            train_texts, train_labels = texts, labels
        
        # Create datasets
        train_dataset = Dataset.from_dict({
            'text': train_texts,
            'label': train_labels
        })
        
        test_dataset = Dataset.from_dict({
            'text': test_texts,
            'label': test_labels
        })
        
        # Load tokenizer and model
        model_name = "distilbert-base-uncased"
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        model = AutoModelForSequenceClassification.from_pretrained(
            model_name,
            num_labels=2,
            id2label={0: "normal", 1: "distressed"},
            label2id={"normal": 0, "distressed": 1}
        )
        
        # Tokenize function
        def tokenize_function(examples):
            return tokenizer(examples['text'], padding="max_length", truncation=True, max_length=128)
        
        # Tokenize datasets
        train_tokenized = train_dataset.map(tokenize_function, batched=True)
        test_tokenized = test_dataset.map(tokenize_function, batched=True)
        
        # Define training arguments
        training_args = TrainingArguments(
            output_dir=os.path.join(self.model_dir, "results"),
            num_train_epochs=3,
            per_device_train_batch_size=16,
            per_device_eval_batch_size=16,
            warmup_steps=500,
            weight_decay=0.01,
            logging_dir=os.path.join(self.model_dir, "logs"),
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
            
            # Calculate metrics
            accuracy = (predictions == labels).mean()
            report = classification_report(labels, predictions, output_dict=True)
            
            # Calculate ROC AUC
            probs = torch.nn.functional.softmax(torch.tensor(logits), dim=1).numpy()[:, 1]
            fpr, tpr, _ = roc_curve(labels, probs)
            roc_auc = auc(fpr, tpr)
            
            return {
                "accuracy": accuracy,
                "f1": report["macro avg"]["f1-score"],
                "precision": report["macro avg"]["precision"],
                "recall": report["macro avg"]["recall"],
                "roc_auc": roc_auc
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
        trainer.train()
        
        # Evaluate the model
        eval_results = trainer.evaluate()
        logger.info(f"Transformer model evaluation results: {eval_results}")
        
        # Save the model and tokenizer
        model_dir = os.path.join(self.model_dir, "transformer_model")
        os.makedirs(model_dir, exist_ok=True)
        model.save_pretrained(model_dir)
        tokenizer.save_pretrained(model_dir)
        logger.info(f"Transformer model and tokenizer saved to {model_dir}")
        
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
        
        # Save the wrapper
        wrapper_path = os.path.join(self.model_dir, "transformer_wrapper.pkl")
        joblib.dump(model_wrapper, wrapper_path)
        logger.info(f"Transformer wrapper saved to {wrapper_path}")
        
        # Evaluate the wrapper
        y_pred = model_wrapper.predict(test_texts)
        y_proba = model_wrapper.predict_proba(test_texts)[:, 1]
        
        # Calculate evaluation metrics
        metrics = {
            'accuracy': (y_pred == test_labels).mean(),
            'classification_report': classification_report(test_labels, y_pred, output_dict=True),
            'confusion_matrix': confusion_matrix(test_labels, y_pred).tolist(),
            'roc_auc': auc(roc_curve(test_labels, y_proba)[0], roc_curve(test_labels, y_proba)[1])
        }
        
        logger.info(f"Transformer wrapper accuracy: {metrics['accuracy']:.4f}")
        logger.info(f"Transformer wrapper ROC AUC: {metrics['roc_auc']:.4f}")
        
        return model_wrapper, metrics
    
    def train_ensemble_model(self, models, X_test, y_test):
        """
        Create an ensemble model from multiple trained models.
        
        Args:
            models (list): List of trained models
            X_test: Testing features
            y_test: Testing labels
            
        Returns:
            tuple: (ensemble_model, evaluation_metrics)
        """
        logger.info(f"Creating ensemble model from {len(models)} base models...")
        
        # Get predictions from each model
        predictions = []
        probabilities = []
        
        for model in models:
            try:
                y_pred = model.predict(X_test)
                y_proba = model.predict_proba(X_test)[:, 1]
                
                predictions.append(y_pred)
                probabilities.append(y_proba)
            except Exception as e:
                logger.error(f"Error getting predictions from model: {str(e)}")
        
        if not predictions:
            logger.error("No valid predictions from any model")
            return None, {}
        
        # Create ensemble predictions (majority vote)
        ensemble_pred = np.round(np.mean(predictions, axis=0)).astype(int)
        
        # Create ensemble probabilities (average)
        ensemble_proba = np.mean(probabilities, axis=0)
        
        # Calculate evaluation metrics
        metrics = {
            'accuracy': (ensemble_pred == y_test).mean(),
            'classification_report': classification_report(y_test, ensemble_pred, output_dict=True),
            'confusion_matrix': confusion_matrix(y_test, ensemble_pred).tolist(),
            'roc_auc': auc(roc_curve(y_test, ensemble_proba)[0], roc_curve(y_test, ensemble_proba)[1])
        }
        
        logger.info(f"Ensemble model accuracy: {metrics['accuracy']:.4f}")
        logger.info(f"Ensemble model ROC AUC: {metrics['roc_auc']:.4f}")
        
        # Create a wrapper for the ensemble model
        class EnsembleModel:
            def __init__(self, models):
                self.models = models
            
            def predict(self, X):
                predictions = []
                for model in self.models:
                    try:
                        y_pred = model.predict(X)
                        predictions.append(y_pred)
                    except Exception as e:
                        logger.error(f"Error in ensemble prediction: {str(e)}")
                
                if not predictions:
                    return np.zeros(len(X))
                
                # Majority vote
                return np.round(np.mean(predictions, axis=0)).astype(int)
            
            def predict_proba(self, X):
                probabilities = []
                for model in self.models:
                    try:
                        y_proba = model.predict_proba(X)
                        probabilities.append(y_proba)
                    except Exception as e:
                        logger.error(f"Error in ensemble probability: {str(e)}")
                
                if not probabilities:
                    return np.zeros((len(X), 2))
                
                # Average probabilities
                return np.mean(probabilities, axis=0)
        
        # Create the ensemble model
        ensemble_model = EnsembleModel(models)
        
        # Save the ensemble model
        model_path = os.path.join(self.model_dir, "ensemble_model.pkl")
        joblib.dump(ensemble_model, model_path)
        logger.info(f"Ensemble model saved to {model_path}")
        
        return ensemble_model, metrics
    
    def plot_confusion_matrix(self, y_true, y_pred, title="Confusion Matrix", save_path=None):
        """
        Plot a confusion matrix.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            title (str): Title for the plot
            save_path (str): Path to save the plot
        """
        cm = confusion_matrix(y_true, y_pred)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=['Normal', 'Distressed'],
                   yticklabels=['Normal', 'Distressed'])
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.title(title)
        
        if save_path:
            plt.savefig(save_path)
            logger.info(f"Confusion matrix saved to {save_path}")
        
        plt.close()
    
    def plot_roc_curve(self, y_true, y_proba, title="ROC Curve", save_path=None):
        """
        Plot a ROC curve.
        
        Args:
            y_true: True labels
            y_proba: Predicted probabilities
            title (str): Title for the plot
            save_path (str): Path to save the plot
        """
        fpr, tpr, _ = roc_curve(y_true, y_proba)
        roc_auc = auc(fpr, tpr)
        
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(title)
        plt.legend(loc="lower right")
        
        if save_path:
            plt.savefig(save_path)
            logger.info(f"ROC curve saved to {save_path}")
        
        plt.close()
    
    def plot_precision_recall_curve(self, y_true, y_proba, title="Precision-Recall Curve", save_path=None):
        """
        Plot a precision-recall curve.
        
        Args:
            y_true: True labels
            y_proba: Predicted probabilities
            title (str): Title for the plot
            save_path (str): Path to save the plot
        """
        precision, recall, _ = precision_recall_curve(y_true, y_proba)
        pr_auc = auc(recall, precision)
        
        plt.figure(figsize=(8, 6))
        plt.plot(recall, precision, color='blue', lw=2, label=f'PR curve (area = {pr_auc:.2f})')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title(title)
        plt.legend(loc="lower left")
        
        if save_path:
            plt.savefig(save_path)
            logger.info(f"Precision-recall curve saved to {save_path}")
        
        plt.close()
    
    def train_all_models(self, texts, labels):
        """
        Train all available models and create an ensemble.
        
        Args:
            texts (list): List of text strings
            labels (list): List of labels (0 for normal, 1 for distressed)
            
        Returns:
            dict: Dictionary of trained models and their metrics
        """
        # Create a directory for plots
        plots_dir = os.path.join(self.model_dir, "plots")
        os.makedirs(plots_dir, exist_ok=True)
        
        # Prepare data for traditional models (with feature extraction)
        X_train, X_test, y_train, y_test = self.prepare_data(texts, labels, extract_features=True)
        
        # Train traditional models
        rf_model, rf_metrics = self.train_random_forest(X_train, y_train, X_test, y_test)
        gb_model, gb_metrics = self.train_gradient_boosting(X_train, y_train, X_test, y_test)
        svm_model, svm_metrics = self.train_svm(X_train, y_train, X_test, y_test)
        nn_model, nn_metrics = self.train_neural_network(X_train, y_train, X_test, y_test)
        
        # Prepare data for transformer model (without feature extraction)
        train_texts, test_texts, train_labels, test_labels = self.prepare_data(texts, labels, extract_features=False)
        
        # Train transformer model
        transformer_model, transformer_metrics = self.train_transformer_model(train_texts, train_labels, test_texts, test_labels)
        
        # Create ensemble model from all models
        # We need to handle the fact that transformer model expects text input while others expect features
        class TransformerAdapter:
            def __init__(self, transformer_model, feature_extractor):
                self.transformer_model = transformer_model
                self.feature_extractor = feature_extractor
            
            def predict(self, X):
                # Extract the original texts from the features DataFrame
                # This assumes the original texts are available somewhere
                # In a real implementation, you would need to store the original texts
                # For now, we'll just use the transformer model's predictions on test_texts
                return self.transformer_model.predict(test_texts)
            
            def predict_proba(self, X):
                # Same issue as above
                return self.transformer_model.predict_proba(test_texts)
        
        # Create adapter for transformer model
        transformer_adapter = TransformerAdapter(transformer_model, self.feature_extractor)
        
        # Create ensemble from traditional models (excluding transformer for now)
        traditional_models = [rf_model, gb_model, svm_model, nn_model]
        ensemble_model, ensemble_metrics = self.train_ensemble_model(traditional_models, X_test, y_test)
        
        # Plot confusion matrices
        self.plot_confusion_matrix(
            y_test, rf_model.predict(X_test),
            title="Random Forest Confusion Matrix",
            save_path=os.path.join(plots_dir, "rf_confusion_matrix.png")
        )
        
        self.plot_confusion_matrix(
            y_test, gb_model.predict(X_test),
            title="Gradient Boosting Confusion Matrix",
            save_path=os.path.join(plots_dir, "gb_confusion_matrix.png")
        )
        
        self.plot_confusion_matrix(
            y_test, ensemble_model.predict(X_test),
            title="Ensemble Model Confusion Matrix",
            save_path=os.path.join(plots_dir, "ensemble_confusion_matrix.png")
        )
        
        # Plot ROC curves
        self.plot_roc_curve(
            y_test, rf_model.predict_proba(X_test)[:, 1],
            title="Random Forest ROC Curve",
            save_path=os.path.join(plots_dir, "rf_roc_curve.png")
        )
        
        self.plot_roc_curve(
            y_test, gb_model.predict_proba(X_test)[:, 1],
            title="Gradient Boosting ROC Curve",
            save_path=os.path.join(plots_dir, "gb_roc_curve.png")
        )
        
        self.plot_roc_curve(
            y_test, ensemble_model.predict_proba(X_test)[:, 1],
            title="Ensemble Model ROC Curve",
            save_path=os.path.join(plots_dir, "ensemble_roc_curve.png")
        )
        
        # Return all models and their metrics
        return {
            'random_forest': (rf_model, rf_metrics),
            'gradient_boosting': (gb_model, gb_metrics),
            'svm': (svm_model, svm_metrics),
            'neural_network': (nn_model, nn_metrics),
            'transformer': (transformer_model, transformer_metrics),
            'ensemble': (ensemble_model, ensemble_metrics)
        }


# Example usage
if __name__ == "__main__":
    # Sample data
    data = pd.DataFrame({
        "text": [
            # Distressed examples
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
            
            # Normal examples
            "Life is good, I'm enjoying my day",
            "I had a productive meeting at work today",
            "Feeling great after my workout",
            "I'm excited about my upcoming vacation",
            "Just finished a good book and feeling satisfied",
            "Had a nice dinner with friends tonight",
            "The weather is beautiful today",
            "I accomplished all my tasks for the day",
            "Looking forward to the weekend",
            "I learned something new today and it was interesting"
        ],
        "label": [
            # Labels for distressed examples
            1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
            # Labels for normal examples
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0
        ]
    })
    
    # Initialize model trainer
    trainer = ModelTrainer()
    
    # Train all models
    models = trainer.train_all_models(data["text"].tolist(), data["label"].tolist())
    
    # Print metrics for each model
    for model_name, (model, metrics) in models.items():
        print(f"\n{model_name.upper()} METRICS:")
        print(f"Accuracy: {metrics['accuracy']:.4f}")
        print(f"ROC AUC: {metrics['roc_auc']:.4f}")
        print(f"F1 Score: {metrics['classification_report']['macro avg']['f1-score']:.4f}")