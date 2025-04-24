# Mental Health Predictor

A web application that uses machine learning to analyze text for potential signs of mental distress.

## Overview

This project consists of:
- A machine learning model trained to detect signs of distress in text
- A FastAPI backend that serves the model
- A simple web frontend for user interaction

## Features

- Text analysis for mental health indicators
- Real-time prediction with confidence scores
- Responsive web interface
- API documentation

## Project Structure

```
mental-health-predictor/
├── app/                    # Model training code
│   ├── train_model.py      # Script to train the ML model
│   ├── model/              # Directory for storing the trained model
│   └── requirements.txt    # Python dependencies for model training
├── backend/                # API server
│   ├── main.py             # FastAPI application
│   ├── run.py              # Server startup script
│   └── requirements.txt    # Python dependencies for backend
└── frontend/               # Web interface
    └── index.html          # HTML/CSS/JS for the web interface
```

## Getting Started

### Prerequisites

- Python 3.8 or higher
- pip (Python package manager)

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

3. Install backend dependencies:
   ```
   pip install -r backend/requirements.txt
   ```

4. Train the model (or use the pre-trained model):
   ```
   pip install -r app/requirements.txt
   python app/train_model.py
   ```

### Running the Application

1. Start the backend server:
   ```
   cd backend
   python run.py
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

The mental health prediction model:
- Uses TF-IDF vectorization for text feature extraction
- Implements a Random Forest classifier
- Was trained on a dataset of text samples labeled as normal or distressed

## Disclaimer

This application is for educational and demonstration purposes only. It is not a substitute for professional mental health advice, diagnosis, or treatment. If you or someone you know is experiencing a mental health crisis, please contact a mental health professional or a crisis helpline immediately.

## License

[MIT License](LICENSE)

## Acknowledgments

- This project was created as a demonstration of machine learning applications in mental health.
- The model is trained on a small synthetic dataset and should not be used for actual mental health assessment.