from flask import Flask, render_template, request, jsonify, redirect, url_for, flash
from flask_sqlalchemy import SQLAlchemy
from flask_login import LoginManager, UserMixin, login_user, logout_user, login_required, current_user
from werkzeug.security import generate_password_hash, check_password_hash
import pandas as pd
import numpy as np
import json
import os
from datetime import datetime
import threading
import logging

# Import your modules
from src.data_collection.social_media_collector import SocialMediaCollector
from src.preprocessing.data_cleaner import DataPreprocessor
from src.features.nlp_features import NLPFeatureExtractor
from src.models.ml_models import MentalHealthClassifier
from src.evaluation.model_evaluator import ModelEvaluator
from src.ethics.compliance_checker import EthicsComplianceChecker

app = Flask(__name__)
app.config['SECRET_KEY'] = 'your-secret-key-here'
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///mental_health_app.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

db = SQLAlchemy(app)
login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'login'

# Database Models
class User(UserMixin, db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False)
    email = db.Column(db.String(120), unique=True, nullable=False)
    password_hash = db.Column(db.String(120), nullable=False)
    role = db.Column(db.String(20), default='user')  # user, admin, researcher
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

class AnalysisJob(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    job_name = db.Column(db.String(100), nullable=False)
    status = db.Column(db.String(20), default='pending')  # pending, running, completed, failed
    parameters = db.Column(db.Text)  # JSON string of parameters
    results_path = db.Column(db.String(200))
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    completed_at = db.Column(db.DateTime)

class PredictionResult(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    job_id = db.Column(db.Integer, db.ForeignKey('analysis_job.id'), nullable=False)
    text_input = db.Column(db.Text, nullable=False)
    risk_score = db.Column(db.Float, nullable=False)
    sentiment_score = db.Column(db.Float)
    confidence = db.Column(db.Float)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))

# Global variables for models
ml_classifier = None
feature_extractor = None
preprocessor = None

def load_models():
    """Load trained models on startup"""
    global ml_classifier, feature_extractor, preprocessor
    try:
        import joblib
        ml_classifier = joblib.load('models/best_model.pkl')
        feature_extractor = NLPFeatureExtractor()
        preprocessor = DataPreprocessor()
        app.logger.info("Models loaded successfully")
    except Exception as e:
        app.logger.error(f"Error loading models: {e}")

# Routes
@app.route('/')
def index():
    """Home page"""
    return render_template('index.html')

@app.route('/register', methods=['GET', 'POST'])
def register():
    """User registration"""
    if request.method == 'POST':
        username = request.form['username']
        email = request.form['email']
        password = request.form['password']
        
        if User.query.filter_by(username=username).first():
            flash('Username already exists')
            return redirect(url_for('register'))
        
        if User.query.filter_by(email=email).first():
            flash('Email already exists')
            return redirect(url_for('register'))
        
        user = User(
            username=username,
            email=email,
            password_hash=generate_password_hash(password)
        )
        db.session.add(user)
        db.session.commit()
        
        flash('Registration successful')
        return redirect(url_for('login'))
    
    return render_template('register.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    """User login"""
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        
        user = User.query.filter_by(username=username).first()
        
        if user and check_password_hash(user.password_hash, password):
            login_user(user)
            return redirect(url_for('dashboard'))
        else:
            flash('Invalid username or password')
    
    return render_template('login.html')

@app.route('/logout')
@login_required
def logout():
    """User logout"""
    logout_user()
    return redirect(url_for('index'))

@app.route('/dashboard')
@login_required
def dashboard():
    """User dashboard"""
    user_jobs = AnalysisJob.query.filter_by(user_id=current_user.id).order_by(AnalysisJob.created_at.desc()).all()
    return render_template('dashboard.html', jobs=user_jobs)

@app.route('/analyze', methods=['GET', 'POST'])
@login_required
def analyze():
    """Text analysis page"""
    if request.method == 'POST':
        text_input = request.form['text_input']
        
        if not text_input.strip():
            flash('Please enter some text to analyze')
            return redirect(url_for('analyze'))
        
        try:
            # Process the text
            result = analyze_text(text_input)
            
            # Save to database
            prediction = PredictionResult(
                job_id=0,  # Individual prediction, not part of a job
                text_input=text_input,
                risk_score=result['risk_score'],
                sentiment_score=result['sentiment_score'],
                confidence=result['confidence']
            )
            db.session.add(prediction)
            db.session.commit()
            
            return render_template('analyze.html', result=result, text_input=text_input)
            
        except Exception as e:
            flash(f'Error analyzing text: {str(e)}')
            return redirect(url_for('analyze'))
    
    return render_template('analyze.html')

@app.route('/batch_analysis', methods=['GET', 'POST'])
@login_required
def batch_analysis():
    """Batch analysis page"""
    if request.method == 'POST':
        job_name = request.form['job_name']
        data_source = request.form['data_source']
        keywords = request.form['keywords']
        
        # Create analysis job
        job = AnalysisJob(
            user_id=current_user.id,
            job_name=job_name,
            status='pending',
            parameters=json.dumps({
                'data_source': data_source,
                'keywords': keywords.split(',')
            })
        )
        db.session.add(job)
        db.session.commit()
        
        # Start background job
        thread = threading.Thread(target=run_batch_analysis, args=(job.id,))
        thread.start()
        
        flash('Batch analysis job started')
        return redirect(url_for('dashboard'))
    
    return render_template('batch_analysis.html')

@app.route('/results/<int:job_id>')
@login_required
def view_results(job_id):
    """View analysis results"""
    job = AnalysisJob.query.get_or_404(job_id)
    
    if job.user_id != current_user.id and current_user.role != 'admin':
        flash('Access denied')
        return redirect(url_for('dashboard'))
    
    if job.status != 'completed':
        flash('Analysis not completed yet')
        return redirect(url_for('dashboard'))
    
    # Load results
    try:
        results_df = pd.read_csv(job.results_path)
        results_summary = generate_results_summary(results_df)
        return render_template('results.html', job=job, summary=results_summary)
    except Exception as e:
        flash(f'Error loading results: {str(e)}')
        return redirect(url_for('dashboard'))

@app.route('/api/predict', methods=['POST'])
@login_required
def api_predict():
    """API endpoint for text prediction"""
    data = request.get_json()
    text = data.get('text', '')
    
    if not text:
        return jsonify({'error': 'No text provided'}), 400
    
    try:
        result = analyze_text(text)
        return jsonify(result)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/job_status/<int:job_id>')
@login_required
def api_job_status(job_id):
    """API endpoint to check job status"""
    job = AnalysisJob.query.get_or_404(job_id)
    
    if job.user_id != current_user.id and current_user.role != 'admin':
        return jsonify({'error': 'Access denied'}), 403
    
    return jsonify({
        'id': job.id,
        'status': job.status,
        'created_at': job.created_at.isoformat(),
        'completed_at': job.completed_at.isoformat() if job.completed_at else None
    })

# Helper Functions
def analyze_text(text):
    """Analyze a single text input"""
    global ml_classifier, feature_extractor, preprocessor
    
    if not ml_classifier:
        raise Exception("Model not loaded")
    
    # Preprocess text
    cleaned_text = preprocessor.clean_text(text)
    
    # Create DataFrame for feature extraction
    df = pd.DataFrame({'cleaned_text': [cleaned_text], 'combined_text': [text]})
    
    # Extract features
    df = feature_extractor.extract_sentiment_features(df)
    df = feature_extractor.extract_linguistic_features(df)
    df = feature_extractor.extract_keyword_features(df)
    
    # Prepare features for prediction
    feature_columns = [
        'sentiment_compound', 'sentiment_positive', 'sentiment_negative',
        'text_length', 'word_count', 'avg_word_length',
        'first_person_ratio', 'depression_ratio', 'anxiety_ratio'
    ]
    
    features = df[feature_columns].values
    
    # Make prediction
    risk_score = ml_classifier.predict_proba(features)[0][1]
    risk_prediction = ml_classifier.predict(features)[0]
    
    return {
        'risk_score': float(risk_score),
        'risk_prediction': int(risk_prediction),
        'sentiment_score': float(df['sentiment_compound'].iloc[0]),
        'confidence': float(max(ml_classifier.predict_proba(features)[0])),
        'risk_level': 'High' if risk_prediction == 1 else 'Low'
    }

def run_batch_analysis(job_id):
    """Run batch analysis in background"""
    job = AnalysisJob.query.get(job_id)
    
    try:
        job.status = 'running'
        db.session.commit()
        
        parameters = json.loads(job.parameters)
        
        # Run the analysis pipeline
        collector = SocialMediaCollector('config/api_config.json')
        
        if parameters['data_source'] == 'twitter':
            data = collector.collect_twitter_data(parameters['keywords'], max_results=1000)
        elif parameters['data_source'] == 'reddit':
            data = collector.collect_reddit_data(parameters['keywords'], max_posts=500)
        
        # Process and analyze data
        preprocessor = DataPreprocessor()
        feature_extractor = NLPFeatureExtractor()
        
        processed_data = preprocessor.preprocess_dataframe(data)
        featured_data = feature_extractor.extract_sentiment_features(processed_data)
        featured_data = feature_extractor.extract_linguistic_features(featured_data)
        featured_data = feature_extractor.extract_keyword_features(featured_data)
        
        # Save results
        results_path = f'data/results/job_{job_id}_results.csv'
        featured_data.to_csv(results_path, index=False)
        
        job.status = 'completed'
        job.results_path = results_path
        job.completed_at = datetime.utcnow()
        db.session.commit()
        
    except Exception as e:
        job.status = 'failed'
        db.session.commit()
        app.logger.error(f"Job {job_id} failed: {str(e)}")

def generate_results_summary(df):
    """Generate summary statistics for results"""
    return {
        'total_posts': len(df),
        'avg_sentiment': df['sentiment_compound'].mean(),
        'high_risk_posts': len(df[df.get('mental_health_risk', 0) == 1]),
        'platforms': df['platform'].value_counts().to_dict() if 'platform' in df.columns else {},
        'date_range': {
            'start': df['created_at'].min() if 'created_at' in df.columns else None,
            'end': df['created_at'].max() if 'created_at' in df.columns else None
        }
    }

if __name__ == '__main__':
    with app.app_context():
        db.create_all()
        load_models()
    
    app.run(debug=True, host='0.0.0.0', port=5000)