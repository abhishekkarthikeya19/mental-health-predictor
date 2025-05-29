from flask import Flask, request, render_template_string
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import os
from werkzeug.utils import secure_filename

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max

# Create upload directory
os.makedirs('uploads', exist_ok=True)

def analyze_dataset(df):
    """Analyze mental health dataset"""
    results = {
        'basic_info': {
            'rows': len(df),
            'columns': len(df.columns),
            'column_names': list(df.columns)
        },
        'mental_health_analysis': {},
        'text_analysis': {},
        'predictions': {}
    }
    
    # Find mental health related columns
    mental_health_keywords = ['mental', 'health', 'depression', 'anxiety', 'stress', 'mood', 'wellbeing']
    mental_cols = [col for col in df.columns if any(keyword in col.lower() for keyword in mental_health_keywords)]
    
    # Find text columns
    text_cols = [col for col in df.columns if df[col].dtype == 'object' and df[col].str.len().mean() > 10]
    
    results['mental_health_analysis']['mental_health_columns'] = mental_cols
    results['text_analysis']['text_columns'] = text_cols
    
    # Analyze mental health columns
    for col in mental_cols:
        if df[col].dtype in ['int64', 'float64']:
            results['mental_health_analysis'][col] = {
                'mean': float(df[col].mean()),
                'median': float(df[col].median()),
                'min': float(df[col].min()),
                'max': float(df[col].max()),
                'std': float(df[col].std())
            }
        else:
            results['mental_health_analysis'][col] = df[col].value_counts().head().to_dict()
    
    # Text analysis for mental health keywords
    if text_cols:
        keywords = ['sad', 'happy', 'depressed', 'anxious', 'stressed', 'worried', 'calm', 'angry', 'hopeless', 'excited']
        for col in text_cols[:2]:  # Analyze first 2 text columns
            text_data = df[col].dropna().astype(str)
            keyword_counts = {}
            for keyword in keywords:
                count = text_data.str.lower().str.contains(keyword).sum()
                if count > 0:
                    keyword_counts[keyword] = int(count)
            results['text_analysis'][f'{col}_keywords'] = keyword_counts
    
    # Simple prediction model
    if mental_cols and text_cols:
        try:
            target_col = mental_cols[0]
            text_col = text_cols[0]
            
            clean_df = df[[target_col, text_col]].dropna()
            if len(clean_df) > 20:
                # Create binary target
                if clean_df[target_col].dtype in ['int64', 'float64']:
                    median_val = clean_df[target_col].median()
                    y = (clean_df[target_col] > median_val).astype(int)
                else:
                    top_cats = clean_df[target_col].value_counts().head(2).index
                    clean_df = clean_df[clean_df[target_col].isin(top_cats)]
                    y = (clean_df[target_col] == top_cats[0]).astype(int)
                
                # Vectorize text
                vectorizer = TfidfVectorizer(max_features=50, stop_words='english')
                X = vectorizer.fit_transform(clean_df[text_col].astype(str))
                
                # Train model
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
                model = RandomForestClassifier(n_estimators=50, random_state=42)
                model.fit(X_train, y_train)
                
                # Get accuracy
                y_pred = model.predict(X_test)
                accuracy = accuracy_score(y_test, y_pred)
                
                results['predictions'] = {
                    'model_accuracy': f"{accuracy:.2%}",
                    'target_variable': target_col,
                    'text_variable': text_col,
                    'training_samples': len(X_train),
                    'test_samples': len(X_test)
                }
        except Exception as e:
            results['predictions']['error'] = str(e)
    
    return results

@app.route('/')
def home():
    return render_template_string('''
    <!DOCTYPE html>
    <html>
    <head>
        <title>Mental Health Dataset Analyzer</title>
        <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.1.3/dist/css/bootstrap.min.css" rel="stylesheet">
        <style>
            body { background-color: #f8f9fa; }
            .main-card { box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1); }
        </style>
    </head>
    <body>
        <div class="container mt-5">
            <div class="row justify-content-center">
                <div class="col-md-8">
                    <div class="card main-card">
                        <div class="card-header bg-primary text-white text-center">
                            <h2>🧠 Mental Health Dataset Analyzer</h2>
                            <p class="mb-0">Upload CSV files for AI-powered mental health analysis</p>
                        </div>
                        <div class="card-body p-4">
                            <form action="/analyze" method="post" enctype="multipart/form-data">
                                <div class="mb-4">
                                    <label for="file" class="form-label"><strong>Select CSV Dataset:</strong></label>
                                    <input type="file" class="form-control form-control-lg" id="file" name="file" accept=".csv" required>
                                    <div class="form-text">Supported: CSV files with mental health data, survey responses, or text data</div>
                                </div>
                                <div class="d-grid">
                                    <button type="submit" class="btn btn-primary btn-lg">📊 Analyze Dataset</button>
                                </div>
                            </form>
                        </div>
                    </div>
                    
                    <div class="row mt-4">
                        <div class="col-md-4">
                            <div class="card text-center h-100">
                                <div class="card-body">
                                    <h5>📁 Upload</h5>
                                    <p class="small">CSV files with mental health indicators</p>
                                </div>
                            </div>
                        </div>
                        <div class="col-md-4">
                            <div class="card text-center h-100">
                                <div class="card-body">
                                    <h5>🤖 Analyze</h5>
                                    <p class="small">AI-powered pattern detection</p>
                                </div>
                            </div>
                        </div>
                        <div class="col-md-4">
                            <div class="card text-center h-100">
                                <div class="card-body">
                                    <h5>📈 Results</h5>
                                    <p class="small">Comprehensive insights & predictions</p>
                                </div>
                            </div>
                        </div>
                    </div>
                </div>
            </div>
        </div>
    </body>
    </html>
    ''')

@app.route('/analyze', methods=['POST'])
def analyze():
    if 'file' not in request.files:
        return "No file uploaded", 400
    
    file = request.files['file']
    if file.filename == '':
        return "No file selected", 400
    
    if file and file.filename.endswith('.csv'):
        try:
            # Read CSV
            df = pd.read_csv(file)
            
            # Analyze dataset
            results = analyze_dataset(df)
            
            return render_template_string('''
            <!DOCTYPE html>
            <html>
            <head>
                <title>Analysis Results</title>
                <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.1.3/dist/css/bootstrap.min.css" rel="stylesheet">
            </head>
            <body style="background-color: #f8f9fa;">
                <div class="container mt-4">
                    <div class="row">
                        <div class="col-md-10 mx-auto">
                            <div class="card">
                                <div class="card-header bg-success text-white">
                                    <h3>✅ Analysis Complete</h3>
                                    <a href="/" class="btn btn-light btn-sm float-end">Upload New Dataset</a>
                                </div>
                                <div class="card-body">
                                    
                                    <!-- Basic Info -->
                                    <div class="row mb-4">
                                        <div class="col-md-12">
                                            <h5>📊 Dataset Overview</h5>
                                            <div class="alert alert-info">
                                                <strong>Rows:</strong> {{ results.basic_info.rows }}<br>
                                                <strong>Columns:</strong> {{ results.basic_info.columns }}<br>
                                                <strong>Column Names:</strong> {{ results.basic_info.column_names|join(', ') }}
                                            </div>
                                        </div>
                                    </div>
                                    
                                    <!-- Mental Health Analysis -->
                                    {% if results.mental_health_analysis.mental_health_columns %}
                                    <div class="row mb-4">
                                        <div class="col-md-12">
                                            <h5>🧠 Mental Health Analysis</h5>
                                            <div class="alert alert-warning">
                                                <strong>Mental Health Columns Found:</strong> {{ results.mental_health_analysis.mental_health_columns|join(', ') }}
                                            </div>
                                            {% for col, stats in results.mental_health_analysis.items() %}
                                                {% if col != 'mental_health_columns' %}
                                                <div class="card mb-2">
                                                    <div class="card-body">
                                                        <h6>{{ col }}</h6>
                                                        {% if stats is mapping and 'mean' in stats %}
                                                            <p><strong>Mean:</strong> {{ "%.2f"|format(stats.mean) }} | 
                                                               <strong>Median:</strong> {{ "%.2f"|format(stats.median) }} | 
                                                               <strong>Range:</strong> {{ "%.2f"|format(stats.min) }} - {{ "%.2f"|format(stats.max) }}</p>
                                                        {% else %}
                                                            <p>{{ stats }}</p>
                                                        {% endif %}
                                                    </div>
                                                </div>
                                                {% endif %}
                                            {% endfor %}
                                        </div>
                                    </div>
                                    {% endif %}
                                    
                                    <!-- Text Analysis -->
                                    {% if results.text_analysis.text_columns %}
                                    <div class="row mb-4">
                                        <div class="col-md-12">
                                            <h5>💬 Text Analysis</h5>
                                            <div class="alert alert-primary">
                                                <strong>Text Columns Found:</strong> {{ results.text_analysis.text_columns|join(', ') }}
                                            </div>
                                            {% for col, keywords in results.text_analysis.items() %}
                                                {% if col != 'text_columns' and keywords %}
                                                <div class="card mb-2">
                                                    <div class="card-body">
                                                        <h6>{{ col.replace('_keywords', '') }} - Mental Health Keywords</h6>
                                                        {% for keyword, count in keywords.items() %}
                                                            <span class="badge bg-info me-2">{{ keyword }}: {{ count }}</span>
                                                        {% endfor %}
                                                    </div>
                                                </div>
                                                {% endif %}
                                            {% endfor %}
                                        </div>
                                    </div>
                                    {% endif %}
                                    
                                    <!-- Predictions -->
                                    {% if results.predictions and 'model_accuracy' in results.predictions %}
                                    <div class="row mb-4">
                                        <div class="col-md-12">
                                            <h5>🤖 AI Predictions</h5>
                                            <div class="alert alert-success">
                                                <h6>Model Performance:</h6>
                                                <p><strong>Accuracy:</strong> {{ results.predictions.model_accuracy }}</p>
                                                <p><strong>Target Variable:</strong> {{ results.predictions.target_variable }}</p>
                                                <p><strong>Text Variable:</strong> {{ results.predictions.text_variable }}</p>
                                                <p><strong>Training Samples:</strong> {{ results.predictions.training_samples }} | 
                                                   <strong>Test Samples:</strong> {{ results.predictions.test_samples }}</p>
                                            </div>
                                        </div>
                                    </div>
                                    {% endif %}
                                    
                                    {% if results.predictions and 'error' in results.predictions %}
                                    <div class="alert alert-warning">
                                        <strong>Prediction Model:</strong> Could not build prediction model - {{ results.predictions.error }}
                                    </div>
                                    {% endif %}
                                    
                                </div>
                            </div>
                        </div>
                    </div>
                </div>
            </body>
            </html>
            ''', results=results)
            
        except Exception as e:
            return f"Error processing file: {str(e)}", 400
    
    return "Invalid file format. Please upload a CSV file.", 400

if __name__ == '__main__':
    print("🚀 Starting Mental Health Dataset Analyzer...")
    print("📁 Upload CSV files with mental health data")
    print("🌐 Open: http://localhost:5000")
    app.run(debug=True, host='127.0.0.1', port=5000)