"""
Simple Mental Health Detection Web App
Run this to test the web interface
"""

from flask import Flask, render_template, request, jsonify, redirect, url_for, flash
from flask_sqlalchemy import SQLAlchemy
from flask_login import LoginManager, UserMixin, login_user, logout_user, login_required, current_user
from werkzeug.security import generate_password_hash, check_password_hash
import os
from datetime import datetime
import random
import re

app = Flask(__name__)
app.config['SECRET_KEY'] = 'your-secret-key-change-this'
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///mental_health_app.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

db = SQLAlchemy(app)
login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'login'

# Create templates directory if it doesn't exist
if not os.path.exists('templates'):
    os.makedirs('templates')

# Database Models
class User(UserMixin, db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False)
    email = db.Column(db.String(120), unique=True, nullable=False)
    password_hash = db.Column(db.String(120), nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

class Analysis(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    text_input = db.Column(db.Text, nullable=False)
    risk_score = db.Column(db.Float, nullable=False)
    sentiment_score = db.Column(db.Float)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))

# Simple analysis function (mock for now)
def analyze_text_simple(text):
    """Simple mock analysis - replace with real ML model later"""
    
    # Simple keyword-based analysis
    negative_words = ['sad', 'depressed', 'anxious', 'worried', 'hopeless', 'alone', 'tired', 'stressed']
    positive_words = ['happy', 'good', 'great', 'excited', 'love', 'amazing', 'wonderful']
    
    text_lower = text.lower()
    
    negative_count = sum(1 for word in negative_words if word in text_lower)
    positive_count = sum(1 for word in positive_words if word in text_lower)
    
    # Calculate scores
    total_words = len(text.split())
    risk_score = min(0.9, (negative_count / max(total_words, 1)) * 5)
    sentiment_score = (positive_count - negative_count) / max(total_words, 1)
    
    return {
        'risk_score': risk_score,
        'sentiment_score': sentiment_score,
        'risk_level': 'High' if risk_score > 0.3 else 'Low',
        'confidence': 0.75
    }

# Routes
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form['username']
        email = request.form['email']
        password = request.form['password']
        
        if User.query.filter_by(username=username).first():
            flash('Username already exists')
            return redirect(url_for('register'))
        
        user = User(
            username=username,
            email=email,
            password_hash=generate_password_hash(password)
        )
        db.session.add(user)
        db.session.commit()
        
        flash('Registration successful! Please login.')
        return redirect(url_for('login'))
    
    return render_template('register.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
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
    logout_user()
    return redirect(url_for('index'))

@app.route('/dashboard')
@login_required
def dashboard():
    analyses = Analysis.query.filter_by(user_id=current_user.id).order_by(Analysis.created_at.desc()).limit(10).all()
    return render_template('dashboard.html', analyses=analyses)

@app.route('/analyze', methods=['GET', 'POST'])
@login_required
def analyze():
    result = None
    text_input = ""
    
    if request.method == 'POST':
        text_input = request.form['text_input']
        
        if not text_input.strip():
            flash('Please enter some text to analyze')
            return redirect(url_for('analyze'))
        
        # Analyze the text
        result = analyze_text_simple(text_input)
        
        # Save to database
        analysis = Analysis(
            user_id=current_user.id,
            text_input=text_input,
            risk_score=result['risk_score'],
            sentiment_score=result['sentiment_score']
        )
        db.session.add(analysis)
        db.session.commit()
        
        flash('Analysis completed!')
    
    return render_template('analyze.html', result=result, text_input=text_input)

@app.route('/api/analyze', methods=['POST'])
@login_required
def api_analyze():
    data = request.get_json()
    text = data.get('text', '')
    
    if not text:
        return jsonify({'error': 'No text provided'}), 400
    
    result = analyze_text_simple(text)
    return jsonify(result)

if __name__ == '__main__':
    with app.app_context():
        db.create_all()
        print("Database created successfully!")
    
    print("Starting Mental Health Detection Web App...")
    print("Open your browser and go to: http://localhost:5000")
    app.run(debug=True, host='0.0.0.0', port=5000)