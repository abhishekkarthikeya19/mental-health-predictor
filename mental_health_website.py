from flask import Flask, render_template_string, jsonify, request
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
import json
import base64
import io
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

app = Flask(__name__)

# Sample Mental Health Data
SAMPLE_DATASETS = {
    'social_media_posts': {
        'name': 'Social Media Mental Health Posts',
        'description': 'Social media posts with mental health indicators',
        'data': [
            {'post_text': 'Feeling really sad and hopeless today. Nothing seems to matter anymore.', 'depression_score': 8, 'anxiety_level': 'high', 'age': 25},
            {'post_text': 'Had an amazing day! Feeling grateful and happy about life.', 'depression_score': 2, 'anxiety_level': 'low', 'age': 30},
            {'post_text': 'Constantly worried about everything. Cannot sleep at night.', 'depression_score': 7, 'anxiety_level': 'high', 'age': 28},
            {'post_text': 'Life is beautiful. Enjoying every moment with family and friends.', 'depression_score': 1, 'anxiety_level': 'low', 'age': 35},
            {'post_text': 'Feeling overwhelmed with work stress. Need a break desperately.', 'depression_score': 6, 'anxiety_level': 'medium', 'age': 32},
            {'post_text': 'Excited about new opportunities coming my way!', 'depression_score': 2, 'anxiety_level': 'low', 'age': 27},
            {'post_text': 'Cannot get out of bed. Everything feels pointless and dark.', 'depression_score': 9, 'anxiety_level': 'high', 'age': 24},
            {'post_text': 'Peaceful morning meditation. Feeling centered and calm.', 'depression_score': 1, 'anxiety_level': 'low', 'age': 40},
            {'post_text': 'Panic attacks are getting worse. Scared to leave the house.', 'depression_score': 8, 'anxiety_level': 'high', 'age': 26},
            {'post_text': 'Celebrating small wins today. Progress feels good!', 'depression_score': 3, 'anxiety_level': 'low', 'age': 29},
        ]
    },
    'survey_responses': {
        'name': 'Mental Health Survey Data',
        'description': 'PHQ-9 and GAD-7 survey responses',
        'data': [
            {'phq9_score': 15, 'gad7_score': 12, 'sleep_hours': 4, 'exercise_weekly': 0, 'stress_level': 8},
            {'phq9_score': 3, 'gad7_score': 2, 'sleep_hours': 8, 'exercise_weekly': 4, 'stress_level': 2},
            {'phq9_score': 18, 'gad7_score': 15, 'sleep_hours': 3, 'exercise_weekly': 1, 'stress_level': 9},
            {'phq9_score': 6, 'gad7_score': 4, 'sleep_hours': 7, 'exercise_weekly': 3, 'stress_level': 4},
            {'phq9_score': 12, 'gad7_score': 10, 'sleep_hours': 5, 'exercise_weekly': 2, 'stress_level': 7},
            {'phq9_score': 2, 'gad7_score': 1, 'sleep_hours': 8, 'exercise_weekly': 5, 'stress_level': 1},
            {'phq9_score': 20, 'gad7_score': 18, 'sleep_hours': 2, 'exercise_weekly': 0, 'stress_level': 10},
            {'phq9_score': 5, 'gad7_score': 3, 'sleep_hours': 7, 'exercise_weekly': 4, 'stress_level': 3},
        ]
    }
}

def create_chart(data, chart_type, title):
    """Create charts and return as base64 string"""
    plt.figure(figsize=(10, 6))
    
    if chart_type == 'bar':
        keys = list(data.keys())
        values = list(data.values())
        colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7', '#DDA0DD', '#98D8C8']
        plt.bar(keys, values, color=colors[:len(keys)])
        plt.xticks(rotation=45)
    elif chart_type == 'pie':
        colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7']
        plt.pie(data.values(), labels=data.keys(), autopct='%1.1f%%', startangle=90, colors=colors)
    
    plt.title(title, fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    # Convert to base64
    img = io.BytesIO()
    plt.savefig(img, format='png', bbox_inches='tight', dpi=150)
    img.seek(0)
    chart_url = base64.b64encode(img.getvalue()).decode()
    plt.close()
    
    return chart_url

def analyze_dataset(dataset_key):
    """Backend analysis function"""
    dataset = SAMPLE_DATASETS[dataset_key]
    df = pd.DataFrame(dataset['data'])
    
    analysis = {
        'dataset_info': {
            'name': dataset['name'],
            'description': dataset['description'],
            'total_records': len(df),
            'columns': list(df.columns)
        },
        'statistics': {},
        'text_analysis': {},
        'ml_results': {},
        'insights': [],
        'charts': {}
    }
    
    # Statistical Analysis
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        analysis['statistics'][col] = {
            'mean': round(df[col].mean(), 2),
            'median': round(df[col].median(), 2),
            'min': int(df[col].min()),
            'max': int(df[col].max()),
            'std': round(df[col].std(), 2)
        }
    
    # Text Analysis
    if 'post_text' in df.columns:
        mental_keywords = ['sad', 'happy', 'depressed', 'anxious', 'worried', 'hopeless', 'excited', 'stressed', 'calm', 'panic']
        text_data = df['post_text'].astype(str)
        
        keyword_counts = {}
        for keyword in mental_keywords:
            count = text_data.str.lower().str.contains(keyword, na=False).sum()
            if count > 0:
                keyword_counts[keyword] = count
        
        analysis['text_analysis'] = {
            'total_posts': len(text_data),
            'avg_length': round(text_data.str.len().mean(), 1),
            'keyword_frequency': keyword_counts
        }
    
    # Machine Learning Analysis
    try:
        if dataset_key == 'social_media_posts':
            # Depression prediction
            text_data = df['post_text']
            y = (df['depression_score'] > 5).astype(int)
            
            vectorizer = TfidfVectorizer(max_features=50, stop_words='english')
            X = vectorizer.fit_transform(text_data)
            
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
            model = RandomForestClassifier(n_estimators=30, random_state=42)
            model.fit(X_train, y_train)
            
            y_pred = model.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            
            analysis['ml_results'] = {
                'model_type': 'Depression Risk Prediction',
                'accuracy': f"{accuracy:.1%}",
                'high_risk_predicted': int(y_pred.sum()),
                'low_risk_predicted': int(len(y_pred) - y_pred.sum())
            }
            
            # Create charts
            depression_dist = df['depression_score'].value_counts().sort_index().to_dict()
            anxiety_dist = df['anxiety_level'].value_counts().to_dict()
            
            analysis['charts']['depression_scores'] = create_chart(depression_dist, 'bar', 'Depression Score Distribution')
            analysis['charts']['anxiety_levels'] = create_chart(anxiety_dist, 'pie', 'Anxiety Level Distribution')
            
        elif dataset_key == 'survey_responses':
            # Stress prediction
            features = df[['phq9_score', 'gad7_score', 'sleep_hours', 'exercise_weekly']]
            target = (df['stress_level'] > 6).astype(int)
            
            X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.3, random_state=42)
            model = LogisticRegression(random_state=42)
            model.fit(X_train, y_train)
            
            y_pred = model.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            
            analysis['ml_results'] = {
                'model_type': 'High Stress Prediction',
                'accuracy': f"{accuracy:.1%}",
                'high_stress_predicted': int(y_pred.sum()),
                'low_stress_predicted': int(len(y_pred) - y_pred.sum())
            }
            
            # Create charts
            stress_dist = df['stress_level'].value_counts().sort_index().to_dict()
            sleep_dist = df['sleep_hours'].value_counts().sort_index().to_dict()
            
            analysis['charts']['stress_levels'] = create_chart(stress_dist, 'bar', 'Stress Level Distribution')
            analysis['charts']['sleep_hours'] = create_chart(sleep_dist, 'pie', 'Sleep Hours Distribution')
    
    except Exception as e:
        analysis['ml_results'] = {'error': str(e)}
    
    # Generate Insights
    if 'depression_score' in df.columns:
        high_depression = (df['depression_score'] > 6).sum()
        analysis['insights'].append(f"🔍 {high_depression} out of {len(df)} records show high depression scores (>6)")
    
    if 'stress_level' in df.columns:
        avg_stress = df['stress_level'].mean()
        analysis['insights'].append(f"📊 Average stress level: {avg_stress:.1f}/10")
        high_stress = (df['stress_level'] > 7).sum()
        analysis['insights'].append(f"⚠️ {high_stress} individuals report high stress levels (>7)")
    
    if 'anxiety_level' in df.columns:
        high_anxiety = (df['anxiety_level'] == 'high').sum()
        analysis['insights'].append(f"😰 {high_anxiety} records indicate high anxiety levels")
    
    return analysis

# FRONTEND ROUTES
@app.route('/')
def home():
    return render_template_string('''
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Mental Health Analytics Platform</title>
        <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.1.3/dist/css/bootstrap.min.css" rel="stylesheet">
        <link href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0/css/all.min.css" rel="stylesheet">
        <style>
            body {
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                min-height: 100vh;
                font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            }
            .main-container {
                background: rgba(255, 255, 255, 0.95);
                border-radius: 20px;
                box-shadow: 0 20px 40px rgba(0,0,0,0.1);
                backdrop-filter: blur(10px);
            }
            .dataset-card {
                transition: all 0.3s ease;
                cursor: pointer;
                border: none;
                border-radius: 15px;
                overflow: hidden;
            }
            .dataset-card:hover {
                transform: translateY(-10px);
                box-shadow: 0 20px 40px rgba(0,0,0,0.2);
            }
            .feature-icon {
                font-size: 3rem;
                margin-bottom: 1rem;
            }
            .hero-section {
                background: linear-gradient(45deg, #667eea, #764ba2);
                color: white;
                border-radius: 15px;
                padding: 3rem;
                margin-bottom: 3rem;
            }
            .stats-card {
                background: linear-gradient(45deg, #4facfe, #00f2fe);
                color: white;
                border-radius: 15px;
                padding: 2rem;
                text-align: center;
            }
        </style>
    </head>
    <body>
        <div class="container py-5">
            <div class="main-container p-5">
                <!-- Hero Section -->
                <div class="hero-section text-center">
                    <h1 class="display-3 mb-3">
                        <i class="fas fa-brain"></i> Mental Health Analytics Platform
                    </h1>
                    <p class="lead fs-4">Advanced AI-powered analysis of mental health data with interactive visualizations</p>
                    <div class="row mt-4">
                        <div class="col-md-4">
                            <div class="stats-card">
                                <h3><i class="fas fa-chart-line"></i></h3>
                                <h4>Real-time Analytics</h4>
                                <p>Live data processing</p>
                            </div>
                        </div>
                                            <div class="col-md-4">
                                                <div class="stats-card">
                                                    <h3><i class="fas fa-robot"></i></h3>
                                                    <h4>AI Predictions</h4>
                                                    <p>Machine learning models</p>
                                                </div>
                                            </div>
                                            <div class="col-md-4">
                                                <div class="stats-card">
                                                    <h3><i class="fas fa-users"></i></h3>
                                                    <h4>User Insights</h4>
                                                    <p>Personalized feedback</p>
                                                </div>
                                            </div>
                                        </div>
                                    </div>
                                    <!-- End Hero Section -->
                                    <!-- Add more content here if needed -->
                                </div>
                            </div>
                        </body>
                        </html>
                        ''')