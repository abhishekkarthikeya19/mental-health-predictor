from flask import Flask, request, redirect, url_for, session
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.linear_model import LogisticRegression
import io
import base64
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import json

app = Flask(__name__)
app.secret_key = 'mental-health-complete-app'

# Built-in Sample Datasets
SAMPLE_DATASETS = {
    'social_media_posts': {
        'name': 'Social Media Mental Health Posts',
        'description': 'Social media posts with mental health indicators',
        'data': [
            {'post_text': 'Feeling really sad and hopeless today. Nothing seems to matter anymore.', 'depression_score': 8, 'anxiety_level': 'high', 'age': 25, 'gender': 'female'},
            {'post_text': 'Had an amazing day! Feeling grateful and happy about life.', 'depression_score': 2, 'anxiety_level': 'low', 'age': 30, 'gender': 'male'},
            {'post_text': 'Constantly worried about everything. Cannot sleep at night.', 'depression_score': 7, 'anxiety_level': 'high', 'age': 28, 'gender': 'female'},
            {'post_text': 'Life is beautiful. Enjoying every moment with family and friends.', 'depression_score': 1, 'anxiety_level': 'low', 'age': 35, 'gender': 'male'},
            {'post_text': 'Feeling overwhelmed with work stress. Need a break desperately.', 'depression_score': 6, 'anxiety_level': 'medium', 'age': 32, 'gender': 'female'},
            {'post_text': 'Excited about new opportunities coming my way!', 'depression_score': 2, 'anxiety_level': 'low', 'age': 27, 'gender': 'male'},
            {'post_text': 'Cannot get out of bed. Everything feels pointless and dark.', 'depression_score': 9, 'anxiety_level': 'high', 'age': 24, 'gender': 'female'},
            {'post_text': 'Peaceful morning meditation. Feeling centered and calm.', 'depression_score': 1, 'anxiety_level': 'low', 'age': 40, 'gender': 'male'},
            {'post_text': 'Panic attacks are getting worse. Scared to leave the house.', 'depression_score': 8, 'anxiety_level': 'high', 'age': 26, 'gender': 'female'},
            {'post_text': 'Celebrating small wins today. Progress feels good!', 'depression_score': 3, 'anxiety_level': 'low', 'age': 29, 'gender': 'male'},
            {'post_text': 'Lonely and isolated. Feel like nobody understands me.', 'depression_score': 7, 'anxiety_level': 'medium', 'age': 31, 'gender': 'female'},
            {'post_text': 'Grateful for supportive friends and family around me.', 'depression_score': 2, 'anxiety_level': 'low', 'age': 33, 'gender': 'male'},
            {'post_text': 'Tired of pretending everything is okay when it is not.', 'depression_score': 8, 'anxiety_level': 'high', 'age': 27, 'gender': 'female'},
            {'post_text': 'Looking forward to weekend adventures and new experiences.', 'depression_score': 2, 'anxiety_level': 'low', 'age': 28, 'gender': 'male'},
            {'post_text': 'Anxiety is controlling my life. Cannot make simple decisions.', 'depression_score': 6, 'anxiety_level': 'high', 'age': 25, 'gender': 'female'},
        ]
    },
    'therapy_notes': {
        'name': 'Therapy Session Notes',
        'description': 'Anonymized therapy session notes with outcomes',
        'data': [
            {'session_notes': 'Patient expressed feelings of worthlessness and hopelessness. Discussed coping strategies.', 'improvement_score': 3, 'sessions_attended': 5, 'diagnosis': 'depression'},
            {'session_notes': 'Significant progress in managing anxiety. Patient reports better sleep patterns.', 'improvement_score': 8, 'sessions_attended': 12, 'diagnosis': 'anxiety'},
            {'session_notes': 'Patient struggling with panic attacks. Introduced breathing exercises.', 'improvement_score': 4, 'sessions_attended': 3, 'diagnosis': 'panic_disorder'},
            {'session_notes': 'Excellent breakthrough session. Patient gained new insights about trauma.', 'improvement_score': 9, 'sessions_attended': 15, 'diagnosis': 'ptsd'},
            {'session_notes': 'Mood swings continue to be challenging. Medication adjustment discussed.', 'improvement_score': 5, 'sessions_attended': 8, 'diagnosis': 'bipolar'},
            {'session_notes': 'Patient reports feeling more confident and optimistic about future.', 'improvement_score': 8, 'sessions_attended': 10, 'diagnosis': 'depression'},
            {'session_notes': 'Social anxiety preventing patient from work interactions. Role-playing exercises.', 'improvement_score': 4, 'sessions_attended': 6, 'diagnosis': 'social_anxiety'},
            {'session_notes': 'Remarkable progress in trauma processing. Patient feels empowered.', 'improvement_score': 9, 'sessions_attended': 20, 'diagnosis': 'ptsd'},
            {'session_notes': 'Obsessive thoughts decreasing with cognitive behavioral therapy techniques.', 'improvement_score': 7, 'sessions_attended': 14, 'diagnosis': 'ocd'},
            {'session_notes': 'Patient experiencing setback after family conflict. Processing emotions.', 'improvement_score': 3, 'sessions_attended': 7, 'diagnosis': 'depression'},
        ]
    },
    'survey_responses': {
        'name': 'Mental Health Survey Responses',
        'description': 'PHQ-9 and GAD-7 survey responses',
        'data': [
            {'phq9_score': 15, 'gad7_score': 12, 'sleep_hours': 4, 'exercise_weekly': 0, 'social_support': 'low', 'stress_level': 8},
            {'phq9_score': 3, 'gad7_score': 2, 'sleep_hours': 8, 'exercise_weekly': 4, 'social_support': 'high', 'stress_level': 2},
            {'phq9_score': 18, 'gad7_score': 15, 'sleep_hours': 3, 'exercise_weekly': 1, 'social_support': 'low', 'stress_level': 9},
            {'phq9_score': 6, 'gad7_score': 4, 'sleep_hours': 7, 'exercise_weekly': 3, 'social_support': 'medium', 'stress_level': 4},
            {'phq9_score': 12, 'gad7_score': 10, 'sleep_hours': 5, 'exercise_weekly': 2, 'social_support': 'medium', 'stress_level': 7},
            {'phq9_score': 2, 'gad7_score': 1, 'sleep_hours': 8, 'exercise_weekly': 5, 'social_support': 'high', 'stress_level': 1},
            {'phq9_score': 20, 'gad7_score': 18, 'sleep_hours': 2, 'exercise_weekly': 0, 'social_support': 'low', 'stress_level': 10},
            {'phq9_score': 5, 'gad7_score': 3, 'sleep_hours': 7, 'exercise_weekly': 4, 'social_support': 'high', 'stress_level': 3},
            {'phq9_score': 14, 'gad7_score': 11, 'sleep_hours': 4, 'exercise_weekly': 1, 'social_support': 'low', 'stress_level': 8},
            {'phq9_score': 7, 'gad7_score': 5, 'sleep_hours': 6, 'exercise_weekly': 3, 'social_support': 'medium', 'stress_level': 5},
        ]
    }
}

def analyze_mental_health_data(df, dataset_type):
    """Comprehensive mental health analysis"""
    analysis = {
        'dataset_info': {
            'name': SAMPLE_DATASETS[dataset_type]['name'],
            'description': SAMPLE_DATASETS[dataset_type]['description'],
            'total_records': len(df),
            'columns': list(df.columns)
        },
        'statistical_analysis': {},
        'text_analysis': {},
        'ml_predictions': {},
        'visualizations': {},
        'insights': [],
        'recommendations': []
    }
    
    # Statistical Analysis
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        analysis['statistical_analysis'][col] = {
            'mean': float(df[col].mean()),
            'median': float(df[col].median()),
            'std': float(df[col].std()),
            'min': float(df[col].min()),
            'max': float(df[col].max()),
            'correlation_with_others': {}
        }
    
    # Correlation analysis
    if len(numeric_cols) > 1:
        corr_matrix = df[numeric_cols].corr()
        for col1 in numeric_cols:
            for col2 in numeric_cols:
                if col1 != col2:
                    analysis['statistical_analysis'][col1]['correlation_with_others'][col2] = float(corr_matrix.loc[col1, col2])
    
    # Text Analysis
    text_cols = df.select_dtypes(include=['object']).columns
    mental_health_keywords = [
        'sad', 'happy', 'depressed', 'anxious', 'worried', 'hopeless', 'excited',
        'stressed', 'calm', 'panic', 'fear', 'lonely', 'grateful', 'overwhelmed',
        'tired', 'energetic', 'confident', 'worthless', 'optimistic', 'isolated'
    ]
    
    for col in text_cols:
        if df[col].dtype == 'object':
            text_data = df[col].dropna().astype(str)
            keyword_analysis = {}
            
            for keyword in mental_health_keywords:
                count = text_data.str.lower().str.contains(keyword, na=False).sum()
                if count > 0:
                    keyword_analysis[keyword] = int(count)
            
            analysis['text_analysis'][col] = {
                'total_entries': len(text_data),
                'avg_length': float(text_data.str.len().mean()),
                'keyword_frequency': keyword_analysis
            }
    
    # Machine Learning Predictions
    try:
        if dataset_type == 'social_media_posts':
            # Predict depression score from text
            text_col = 'post_text'
            target_col = 'depression_score'
            
            if text_col in df.columns and target_col in df.columns:
                clean_df = df[[text_col, target_col]].dropna()
                
                # Create binary target (high depression: score > 5)
                y = (clean_df[target_col] > 5).astype(int)
                
                # Vectorize text
                vectorizer = TfidfVectorizer(max_features=100, stop_words='english')
                X = vectorizer.fit_transform(clean_df[text_col])
                
                # Train model
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
                model = RandomForestClassifier(n_estimators=50, random_state=42)
                model.fit(X_train, y_train)
                
                # Predictions
                y_pred = model.predict(X_test)
                accuracy = accuracy_score(y_test, y_pred)
                
                # Feature importance
                feature_names = vectorizer.get_feature_names_out()
                importances = model.feature_importances_
                top_features = sorted(zip(feature_names, importances), key=lambda x: x[1], reverse=True)[:10]
                
                analysis['ml_predictions'] = {
                    'model_type': 'Depression Risk Prediction',
                    'accuracy': f"{accuracy:.2%}",
                    'target': 'High Depression Risk (score > 5)',
                    'training_samples': len(X_train),
                    'test_samples': len(X_test),
                    'top_predictive_words': dict(top_features),
                    'high_risk_predictions': int(y_pred.sum()),
                    'low_risk_predictions': int(len(y_pred) - y_pred.sum())
                }
        
        elif dataset_type == 'survey_responses':
            # Predict high stress from other factors
            feature_cols = ['phq9_score', 'gad7_score', 'sleep_hours', 'exercise_weekly']
            target_col = 'stress_level'
            
            if all(col in df.columns for col in feature_cols + [target_col]):
                clean_df = df[feature_cols + [target_col]].dropna()
                
                X = clean_df[feature_cols]
                y = (clean_df[target_col] > 6).astype(int)  # High stress > 6
                
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
                model = LogisticRegression(random_state=42)
                model.fit(X_train, y_train)
                
                y_pred = model.predict(X_test)
                accuracy = accuracy_score(y_test, y_pred)
                
                analysis['ml_predictions'] = {
                    'model_type': 'High Stress Prediction',
                    'accuracy': f"{accuracy:.2%}",
                    'target': 'High Stress Level (> 6)',
                    'features_used': feature_cols,
                    'training_samples': len(X_train),
                    'test_samples': len(X_test),
                    'high_stress_predictions': int(y_pred.sum()),
                    'low_stress_predictions': int(len(y_pred) - y_pred.sum())
                }
    except Exception as e:
        analysis['ml_predictions'] = {
            'error': f"ML analysis failed: {str(e)}"
        }