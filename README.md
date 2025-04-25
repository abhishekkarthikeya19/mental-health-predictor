# Mental Health Predictor

A comprehensive web application that uses advanced natural language processing, machine learning, and transformer models to analyze text for potential signs of mental distress, track sentiment trends, and provide actionable insights.

## Overview

This project implements a mental health analysis system that:
1. Collects and preprocesses social media data for analysis
2. Extracts relevant features using NLP techniques
3. Applies machine learning models to classify mental health states
4. Analyzes sentiment and language trends over time
5. Provides visualizations and insights through an interactive dashboard
6. Adheres to ethical guidelines for mental health technology

The system consists of:
- Multiple machine learning models (transformer-based and traditional)
- A FastAPI backend that serves the models and provides analysis
- A React-based frontend with interactive visualizations
- Data collection and preprocessing modules
- Sentiment and trend analysis capabilities

## Features

- Advanced text analysis for mental health indicators using transformer models
- Feature extraction using NLP techniques (sentiment, linguistic patterns, emotion detection)
- Multiple machine learning models (Random Forest, SVM, Neural Networks, Transformers)
- Sentiment and trend analysis over time
- Interactive visualization dashboard
- Real-time prediction with confidence scores
- Personalized recommendations based on prediction results
- Historical tracking of mental health indicators
- Responsive web interface
- Comprehensive API documentation
- Ethical considerations and privacy compliance

## Project Structure

```
mental-health-predictor/
├── app/                           # Model training and serving code
│   ├── train_model.py             # Script to train the transformer model
│   ├── advanced_model.py          # Advanced ML model implementations
│   ├── data_collection.py         # Data collection and preprocessing module
│   ├── feature_extraction.py      # NLP feature extraction module
│   ├── sentiment_analysis.py      # Sentiment and trend analysis module
│   ├── load_transformer_model.py  # Utility for loading the transformer model
│   ├── test_transformer_model.py  # Script to test the model performance
│   ├── main.py                    # FastAPI application for the app
│   ├── model/                     # Directory for storing trained models
│   │   ├── mental_health_model.pkl  # Serialized model
│   │   ├── transformer_model/     # Saved transformer model and tokenizer
│   │   └── plots/                 # Model evaluation visualizations
│   ├── data/                      # Directory for storing collected data
│   └── requirements.txt           # Python dependencies for model training
├── backend/                       # API server
│   ├── main.py                    # FastAPI application
│   ├── models.py                  # Pydantic models for request/response
│   ├── utils/                     # Utility modules
│   │   └── model_loader.py        # Model loading utilities
│   ├── run.py                     # Server startup script
│   └── requirements.txt           # Python dependencies for backend
├── frontend/                      # Web interface
│   ├── src/                       # React source code
│   │   ├── components/            # React components
│   │   │   ├── Dashboard.js       # Visualization dashboard
│   │   │   ├── HistorySection.js  # History tracking component
│   │   │   └── ResultDisplay.js   # Results display component
│   │   ├── App.js                 # Main application component
│   │   ├── index.js               # Application entry point
│   │   └── styles.css             # Application styles
│   ├── public/                    # Static assets
│   └── package.json               # Frontend dependencies
├── ETHICAL_CONSIDERATIONS.md      # Ethical guidelines document
└── IMPACT_ASSESSMENT.md           # Impact assessment document
```

## Getting Started

### Prerequisites

- Python 3.8 or higher
- pip (Python package manager)
- PyTorch (for transformer models)

### Installation

1. Clone the repository:
   ```
   git clone https://github.com/yourusername/mental-health-predictor.git
   cd mental-health-predictor
   ```

2. Set up a virtual environment:
   ```
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```

3. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

4. Train the transformer model (or use the pre-trained model):
   ```
   python app/train_model.py
   ```

### Running the Application

1. Start the backend server:
   ```
   python run_app.py
   ```

2. Open the frontend in a web browser:
   - Navigate to the `frontend` directory
   - Open `index.html` in your web browser

Alternatively, you can serve the frontend using a simple HTTP server:
```
cd frontend
python -m http.server
```
Then visit `http://localhost:8000` in your browser.

## API Documentation

Once the backend server is running, you can access the API documentation at:
- Swagger UI: `http://localhost:8000/docs`
- ReDoc: `http://localhost:8000/redoc`

## Model Information

The mental health prediction system includes multiple models:

### Transformer Model
- Uses a transformer-based architecture (DistilBERT) for advanced text understanding
- Fine-tuned on mental health text data to detect signs of distress
- Provides confidence scores and personalized recommendations
- Achieves higher accuracy than traditional machine learning approaches

### Traditional Machine Learning Models
- Random Forest classifier for robust prediction
- Support Vector Machine (SVM) for handling complex decision boundaries
- Gradient Boosting for improved accuracy
- Neural Network for capturing non-linear patterns

### Ensemble Model
- Combines predictions from multiple models for improved reliability
- Weighted voting mechanism for final prediction
- Reduces the impact of individual model biases

### Feature Extraction
- Sentiment analysis using VADER
- Linguistic feature extraction (word usage, sentence structure)
- Emotion detection using specialized models
- Topic modeling to identify themes in text

### Testing the Models

You can test the models' performance using:
```
python app/test_transformer_model.py  # Test transformer model
python app/advanced_model.py          # Test and compare all models
```
This will generate confusion matrices, ROC curves, and detailed performance metrics in the `app/model/plots` directory.

## Data Collection and Analysis

### Data Collection
The system includes modules for collecting data from:
- Social media platforms (Twitter, Reddit)
- User inputs through the web interface
- Local datasets for training and testing

### Sentiment and Trend Analysis
The sentiment analysis module provides:
- Temporal tracking of emotional states
- Identification of linguistic patterns associated with mental health
- Word frequency and topic analysis
- Visualization of trends over time

### Dashboard Visualization
The interactive dashboard displays:
- Sentiment trends over time
- Word frequency analysis
- Emotion distribution
- Language pattern comparisons
- User activity metrics
- Prediction history

## Ethical Considerations

This project takes ethical considerations seriously. For detailed information, please see the [Ethical Considerations](ETHICAL_CONSIDERATIONS.md) document, which covers:

- Privacy and data protection
- Accuracy and reliability
- Potential harms and mitigation strategies
- Inclusivity and accessibility
- Professional standards and compliance
- Recommendations for responsible deployment

## Impact Assessment

For an assessment of the potential impact of this system, its limitations, and future directions, please see the [Impact Assessment](IMPACT_ASSESSMENT.md) document, which covers:

- Potential impact on early detection and awareness
- Technical, ethical, and practical limitations
- Areas for future research and enhancement
- Recommendations for responsible implementation

## Disclaimer

This application is for educational and demonstration purposes only. It is not a substitute for professional mental health advice, diagnosis, or treatment. If you or someone you know is experiencing a mental health crisis, please contact a mental health professional or a crisis helpline immediately.

The predictions and analyses provided by this system are probabilistic in nature and may not be accurate for all individuals or contexts. Always consult with qualified mental health professionals for proper evaluation and treatment.

## License

[MIT License](LICENSE)

## Acknowledgments

- This project was created as a demonstration of applying NLP and machine learning in mental health analysis.
- The models are trained on synthetic datasets and should not be used for actual mental health assessment without proper validation.
- Uses Hugging Face Transformers library for state-of-the-art NLP capabilities.
- Incorporates NLTK and scikit-learn for traditional NLP and machine learning approaches.
- Visualization components use Chart.js for interactive data display.