from flask import Flask, request, redirect, url_for
from flask_sqlalchemy import SQLAlchemy
from flask_login import LoginManager, UserMixin, login_user, logout_user, login_required, current_user
from werkzeug.security import generate_password_hash, check_password_hash
from datetime import datetime

app = Flask(__name__)
app.config['SECRET_KEY'] = 'mental-health-secret-key'
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///mental_health.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

db = SQLAlchemy(app)
login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'login'

# Database Models
class User(UserMixin, db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False)
    password_hash = db.Column(db.String(120), nullable=False)

class Analysis(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    text_input = db.Column(db.Text, nullable=False)
    risk_score = db.Column(db.Float, nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))

def analyze_mental_health(text):
    negative_words = ['sad', 'depressed', 'anxious', 'worried', 'hopeless', 'alone', 'tired', 'stressed', 'down', 'upset']
    words = text.lower().split()
    negative_count = sum(1 for word in negative_words if word in words)
    risk_score = min(0.9, negative_count / max(len(words), 1) * 2)
    return {'risk_score': risk_score, 'risk_level': 'High' if risk_score > 0.3 else 'Low'}

@app.route('/')
def home():
    return '''
    <html>
    <head><title>Mental Health AI</title></head>
    <body style="font-family: Arial; margin: 40px; background: #f5f5f5;">
        <div style="max-width: 800px; margin: 0 auto; background: white; padding: 40px; border-radius: 10px;">
            <h1 style="color: #333; text-align: center;">🧠 Mental Health Detection System</h1>
            <p style="text-align: center; font-size: 18px; color: #666;">AI-powered text analysis for mental health risk detection</p>
            <div style="text-align: center; margin: 30px 0;">
                ''' + ('<a href="/analyze" style="background: #007bff; color: white; padding: 15px 30px; text-decoration: none; border-radius: 5px; margin: 10px;">Start Analysis</a>' if current_user.is_authenticated else '<a href="/register" style="background: #28a745; color: white; padding: 15px 30px; text-decoration: none; border-radius: 5px; margin: 10px;">Register</a><a href="/login" style="background: #007bff; color: white; padding: 15px 30px; text-decoration: none; border-radius: 5px; margin: 10px;">Login</a>') + '''
            </div>
            <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(250px, 1fr)); gap: 20px; margin-top: 40px;">
                <div style="background: #e3f2fd; padding: 20px; border-radius: 8px; text-align: center;">
                    <h3>💬 Text Analysis</h3>
                    <p>Analyze text for mental health indicators</p>
                </div>
                <div style="background: #e8f5e8; padding: 20px; border-radius: 8px; text-align: center;">
                    <h3>📊 Risk Assessment</h3>
                    <p>Get detailed risk scores</p>
                </div>
                <div style="background: #fff3e0; padding: 20px; border-radius: 8px; text-align: center;">
                    <h3>🔒 Privacy First</h3>
                    <p>Secure and anonymous</p>
                </div>
            </div>
        </div>
    </body>
    </html>
    '''

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        
        if User.query.filter_by(username=username).first():
            return '<script>alert("Username exists!"); window.location="/register";</script>'
        
        user = User(username=username, password_hash=generate_password_hash(password))
        db.session.add(user)
        db.session.commit()
        return '<script>alert("Registration successful!"); window.location="/login";</script>'
    
    return '''
    <html>
    <head><title>Register</title></head>
    <body style="font-family: Arial; margin: 40px; background: #f5f5f5;">
        <div style="max-width: 400px; margin: 0 auto; background: white; padding: 40px; border-radius: 10px;">
            <h2 style="text-align: center; color: #333;">Register</h2>
            <form method="POST">
                <div style="margin: 20px 0;">
                    <label style="display: block; margin-bottom: 5px;">Username:</label>
                    <input type="text" name="username" required style="width: 100%; padding: 10px; border: 1px solid #ddd; border-radius: 5px;">
                </div>
                <div style="margin: 20px 0;">
                    <label style="display: block; margin-bottom: 5px;">Password:</label>
                    <input type="password" name="password" required style="width: 100%; padding: 10px; border: 1px solid #ddd; border-radius: 5px;">
                </div>
                <button type="submit" style="width: 100%; padding: 12px; background: #28a745; color: white; border: none; border-radius: 5px; font-size: 16px;">Register</button>
            </form>
            <p style="text-align: center; margin-top: 20px;"><a href="/login">Already have an account? Login</a></p>
            <p style="text-align: center;"><a href="/">← Back to Home</a></p>
        </div>
    </body>
    </html>
    '''

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        user = User.query.filter_by(username=username).first()
        
        if user and check_password_hash(user.password_hash, password):
            login_user(user)
            return redirect(url_for('analyze'))
        else:
            return '<script>alert("Invalid credentials!"); window.location="/login";</script>'
    
    return '''
    <html>
    <head><title>Login</title></head>
    <body style="font-family: Arial; margin: 40px; background: #f5f5f5;">
        <div style="max-width: 400px; margin: 0 auto; background: white; padding: 40px; border-radius: 10px;">
            <h2 style="text-align: center; color: #333;">Login</h2>
            <form method="POST">
                <div style="margin: 20px 0;">
                    <label style="display: block; margin-bottom: 5px;">Username:</label>
                    <input type="text" name="username" required style="width: 100%; padding: 10px; border: 1px solid #ddd; border-radius: 5px;">
                </div>
                <div style="margin: 20px 0;">
                    <label style="display: block; margin-bottom: 5px;">Password:</label>
                    <input type="password" name="password" required style="width: 100%; padding: 10px; border: 1px solid #ddd; border-radius: 5px;">
                </div>
                <button type="submit" style="width: 100%; padding: 12px; background: #007bff; color: white; border: none; border-radius: 5px; font-size: 16px;">Login</button>
            </form>
            <p style="text-align: center; margin-top: 20px;"><a href="/register">Don't have an account? Register</a></p>
            <p style="text-align: center;"><a href="/">← Back to Home</a></p>
        </div>
    </body>
    </html>
    '''

@app.route('/logout')
@login_required
def logout():
    logout_user()
    return redirect(url_for('home'))

@app.route('/analyze', methods=['GET', 'POST'])
@login_required
def analyze():
    result = None
    if request.method == 'POST':
        text_input = request.form['text_input']
        if text_input.strip():
            result = analyze_mental_health(text_input)
            analysis = Analysis(user_id=current_user.id, text_input=text_input, risk_score=result['risk_score'])
            db.session.add(analysis)
            db.session.commit()
    
    result_html = ""
    if result:
        color = "#dc3545" if result['risk_score'] > 0.3 else "#28a745"
        result_html = f'''
        <div style="background: {color}; color: white; padding: 20px; border-radius: 8px; margin: 20px 0;">
            <h3>Analysis Result:</h3>
            <p><strong>Risk Level:</strong> {result['risk_level']}</p>
            <p><strong>Risk Score:</strong> {result['risk_score']:.1%}</p>
        </div>
        '''
    
    return f'''
    <html>
    <head><title>Analyze Text</title></head>
    <body style="font-family: Arial; margin: 40px; background: #f5f5f5;">
        <div style="max-width: 800px; margin: 0 auto; background: white; padding: 40px; border-radius: 10px;">
            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 30px;">
                <h2 style="color: #333;">Text Analysis Dashboard</h2>
                <div>
                    <span style="margin-right: 20px;">Welcome, {current_user.username}!</span>
                    <a href="/logout" style="background: #dc3545; color: white; padding: 8px 16px; text-decoration: none; border-radius: 4px;">Logout</a>
                </div>
            </div>
            
            {result_html}
            
            <div style="background: #f8f9fa; padding: 30px; border-radius: 8px;">
                <h3 style="margin-bottom: 20px;">Enter Text for Analysis:</h3>
                <form method="POST">
                    <textarea name="text_input" rows="6" placeholder="Enter any text for mental health risk analysis..." required 
                              style="width: 100%; padding: 15px; border: 1px solid #ddd; border-radius: 5px; font-size: 16px; resize: vertical;"></textarea>
                    <button type="submit" style="margin-top: 15px; padding: 12px 30px; background: #007bff; color: white; border: none; border-radius: 5px; font-size: 16px;">🔍 Analyze Text</button>
                </form>
            </div>
            
            <div style="margin-top: 30px; text-align: center;">
                <a href="/" style="color: #007bff; text-decoration: none;">← Back to Home</a>
            </div>
        </div>
    </body>
    </html>
    '''

if __name__ == '__main__':
    with app.app_context():
        db.create_all()
        print("✅ Database created successfully!")
        print("🚀 Starting Mental Health Detection Web App...")
        print("🌐 Open your browser and go to: http://localhost:5000")
        print("📝 Or try: http://127.0.0.1:5000")
    
    app.run(debug=True, host='127.0.0.1', port=5000)