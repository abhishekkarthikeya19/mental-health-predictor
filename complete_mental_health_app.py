from flask import Flask, render_template_string
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression

app = Flask(__name__)

# Built-in Sample Datasets
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
        ]
    }
}

def analyze_dataset(dataset_key):
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
        'insights': []
    }
    
    # Basic Statistics
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        analysis['statistics'][col] = {
            'mean': round(df[col].mean(), 2),
            'median': round(df[col].median(), 2),
            'min': int(df[col].min()),
            'max': int(df[col].max())
        }
    
    # Text Analysis
    if 'post_text' in df.columns:
        mental_keywords = ['sad', 'happy', 'depressed', 'anxious', 'worried', 'hopeless', 'excited', 'stressed', 'calm']
        text_data = df['post_text'].astype(str)
        
        keyword_counts = {}
        for keyword in mental_keywords:
            count = text_data.str.lower().str.contains(keyword).sum()
            if count > 0:
                keyword_counts[keyword] = count
        
        analysis['text_analysis'] = {
            'total_posts': len(text_data),
            'avg_length': round(text_data.str.len().mean(), 1),
            'keyword_frequency': keyword_counts
        }
    
    # Generate Insights
    if 'depression_score' in df.columns:
        high_depression = (df['depression_score'] > 6).sum()
        analysis['insights'].append(f"🔍 {high_depression} out of {len(df)} records show high depression scores")
    
    if 'stress_level' in df.columns:
        avg_stress = df['stress_level'].mean()
        analysis['insights'].append(f"📊 Average stress level: {avg_stress:.1f}/10")
    
    return analysis

@app.route('/')
def home():
    return render_template_string('''
    <!DOCTYPE html>
    <html>
    <head>
        <title>Mental Health Analyzer</title>
        <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.1.3/dist/css/bootstrap.min.css" rel="stylesheet">
    </head>
    <body style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); min-height: 100vh;">
        <div class="container py-5">
            <div style="background: white; border-radius: 15px; padding: 40px;">
                <div class="text-center mb-5">
                    <h1 class="display-4 text-primary">🧠 Mental Health Analyzer</h1>
                    <p class="lead">Built-in datasets with AI analysis</p>
                </div>
                
                <div class="row g-4">
                    <div class="col-md-6">
                        <div class="card h-100">
                            <div class="card-body text-center p-4">
                                <div style="font-size: 4rem;">💬</div>
                                <h4>Social Media Posts</h4>
                                <p>Analyze mental health from text data</p>
                                <a href="/analyze/social_media_posts" class="btn btn-primary">Analyze →</a>
                            </div>
                        </div>
                    </div>
                    
                    <div class="col-md-6">
                        <div class="card h-100">
                            <div class="card-body text-center p-4">
                                <div style="font-size: 4rem;">📋</div>
                                <h4>Survey Responses</h4>
                                <p>PHQ-9 and GAD-7 assessments</p>
                                <a href="/analyze/survey_responses" class="btn btn-success">Analyze →</a>
                            </div>
                        </div>
                    </div>
                </div>
            </div>
        </div>
    </body>
    </html>
    ''')

@app.route('/analyze/<dataset_key>')
def analyze_page(dataset_key):
    if dataset_key not in SAMPLE_DATASETS:
        return "Dataset not found", 404
    
    analysis = analyze_dataset(dataset_key)
    
    return render_template_string('''
    <!DOCTYPE html>
    <html>
    <head>
        <title>Analysis Results</title>
        <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.1.3/dist/css/bootstrap.min.css" rel="stylesheet">
    </head>
    <body style="background-color: #f8f9fa;">
        <div class="container py-4">
            <div class="row">
                <div class="col-md-10 mx-auto">
                    <div class="card">
                        <div class="card-header bg-primary text-white">
                            <h3>📊 {{ analysis.dataset_info.name }}</h3>
                            <p class="mb-0">{{ analysis.dataset_info.description }}</p>
                            <a href="/" class="btn btn-light btn-sm float-end">← Back to Home</a>
                        </div>
                        <div class="card-body">
                            
                            <!-- Dataset Overview -->
                            <div class="alert alert-info">
                                <h5>📋 Dataset Overview</h5>
                                <p><strong>Total Records:</strong> {{ analysis.dataset_info.total_records }}</p>
                                <p><strong>Columns:</strong> {{ analysis.dataset_info.columns|join(', ') }}</p>
                            </div>
                            
                            <!-- Statistics -->
                            {% if analysis.statistics %}
                            <div class="mb-4">
                                <h5>📈 Statistical Analysis</h5>
                                {% for col, stats in analysis.statistics.items() %}
                                <div class="card mb-2">
                                    <div class="card-body">
                                        <h6>{{ col }}</h6>
                                        <p>Mean: {{ stats.mean }} | Median: {{ stats.median }} | Range: {{ stats.min }} - {{ stats.max }}</p>
                                    </div>
                                </div>
                                {% endfor %}
                            </div>
                            {% endif %}
                            
                            <!-- Text Analysis -->
                            {% if analysis.text_analysis %}
                            <div class="mb-4">
                                <h5>💬 Text Analysis</h5>
                                <div class="alert alert-warning">
                                    <p><strong>Total Posts:</strong> {{ analysis.text_analysis.total_posts }}</p>
                                    <p><strong>Average Length:</strong> {{ analysis.text_analysis.avg_length }} characters</p>
                                </div>
                                {% if analysis.text_analysis.keyword_frequency %}
                                <h6>Mental Health Keywords Found:</h6>
                                {% for keyword, count in analysis.text_analysis.keyword_frequency.items() %}
                                    <span class="badge bg-info me-2">{{ keyword }}: {{ count }}</span>
                                {% endfor %}
                                {% endif %}
                            </div>
                            {% endif %}
                            
                            <!-- Insights -->
                            {% if analysis.insights %}
                            <div class="mb-4">
                                <h5>💡 Key Insights</h5>
                                {% for insight in analysis.insights %}
                                <div class="alert alert-success">{{ insight }}</div>
                                {% endfor %}
                            </div>
                            {% endif %}
                            
                        </div>
                    </div>
                </div>
            </div>
        </div>
    </body>
    </html>
    ''', analysis=analysis)

if __name__ == '__main__':
    print("🚀 Starting Complete Mental Health Analyzer...")
    print("🌐 Open: http://localhost:5000")
    app.run(debug=True, host='127.0.0.1', port=5000)
