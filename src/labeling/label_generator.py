import pandas as pd
import numpy as np
from typing import Dict, List

class DataLabeler:
    def __init__(self):
        """Initialize data labeler with heuristic rules"""
        self.depression_indicators = [
            'depressed', 'hopeless', 'worthless', 'empty', 'numb',
            'cant sleep', 'no energy', 'nothing matters'
        ]
        
        self.anxiety_indicators = [
            'panic attack', 'cant breathe', 'heart racing', 'worried sick',
            'anxious', 'nervous breakdown', 'scared'
        ]
    
    def create_labels(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create labels based on keyword presence and sentiment"""
        df = df.copy()
        
        # Initialize labels
        df['mental_health_risk'] = 0
        
        # High negative sentiment + mental health keywords
        high_risk_condition = (
            (df['sentiment_compound'] < -0.5) &
            ((df['depression_ratio'] > 0.02) | (df['anxiety_ratio'] > 0.02))
        )
        
        df.loc[high_risk_condition, 'mental_health_risk'] = 1
        
        return df