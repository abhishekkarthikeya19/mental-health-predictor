import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_auc_score, 
    roc_curve, precision_recall_curve, f1_score, accuracy_score
)
from sklearn.model_selection import learning_curve, validation_curve
import joblib
import logging
from typing import Dict, List, Tuple, Any

class ModelEvaluator:
    def __init__(self):
        """Initialize model evaluator"""
        self.logger = logging.getLogger(__name__)
        self.evaluation_results = {}
    
    def comprehensive_evaluation(self, model, X_test: np.ndarray, y_test: np.ndarray, 
                               model_name: str = "Model") -> Dict[str, Any]:
        """Perform comprehensive model evaluation"""
        self.logger.info(f"Evaluating {model_name}...")
        
        # Predictions
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else None
        
        # Basic metrics
        accuracy = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        
        # Classification report
        class_report = classification_report(y_test, y_pred, output_dict=True)
        
        # Confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        
        # ROC AUC if probabilities available
        roc_auc = roc_auc_score(y_test, y_pred_proba) if y_pred_proba is not None else None
        
        results = {
            'model_name': model_name,
            'accuracy': accuracy,
            'f1_score': f1,
            'precision': class_report['1']['precision'],
            'recall': class_report['1']['recall'],
            'roc_auc': roc_auc,
            'confusion_matrix': cm,
            'classification_report': class_report,
            'predictions': y_pred,
            'probabilities': y_pred_proba
        }
        
        self.evaluation_results[model_name] = results
        return results
    
    def plot_confusion_matrix(self, model_name: str, save_path: str = None):
        """Plot confusion matrix"""
        if model_name not in self.evaluation_results:
            raise ValueError(f"No evaluation results found for {model_name}")
        
        cm = self.evaluation_results[model_name]['confusion_matrix']
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=['Low Risk', 'High Risk'],
                   yticklabels=['Low Risk', 'High Risk'])
        plt.title(f'Confusion Matrix - {model_name}')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        
        if save_path:
            plt.savefig(f"{save_path}/confusion_matrix_{model_name}.png", dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_roc_curve(self, y_test, model_names: List[str] = None, save_path: str = None):
        """Plot ROC curves for multiple models"""
        if model_names is None:
            model_names = list(self.evaluation_results.keys())
        
        plt.figure(figsize=(10, 8))
        
        for model_name in model_names:
            if model_name not in self.evaluation_results:
                continue
                
            results = self.evaluation_results[model_name]
            if results['probabilities'] is None:
                continue
            
            fpr, tpr, _ = roc_curve(y_test, results['probabilities'])
            roc_auc = results['roc_auc']
            
            plt.plot(fpr, tpr, label=f'{model_name} (AUC = {roc_auc:.3f})')
        
        plt.plot([0, 1], [0, 1], 'k--', label='Random')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curves Comparison')
        plt.legend()
        plt.grid(True)
        
        if save_path:
            plt.savefig(f"{save_path}/roc_curves.png", dpi=300, bbox_inches='tight')
        plt.show()