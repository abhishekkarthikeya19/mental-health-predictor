from fastapi import FastAPI, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import joblib
import pandas as pd
import numpy as np
import os
import logging
import traceback
import json
from datetime import datetime, timedelta
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Mental Health Predictor API",
    description="API for predicting mental health status from text using transformer models",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Define request and response models
class PredictionRequest(BaseModel):
    text_input: str = Field(..., min_length=1, description="Text to analyze for mental health indicators")

class PredictionResponse(BaseModel):
    prediction: int = Field(..., description="Prediction result: 0 for normal, 1 for distressed")
    confidence: float = Field(..., description="Confidence score of the prediction (0.0 to 1.0)")
    recommendation: str = Field(..., description="Recommendation based on the prediction")

class AnalysisRequest(BaseModel):
    text_input: str = Field(..., min_length=1, description="Text to analyze for mental health indicators")
    history: Optional[List[Dict[str, Any]]] = Field(None, description="Optional history of previous analyses")

class AnalysisResponse(BaseModel):
    prediction: int = Field(..., description="Prediction result: 0 for normal, 1 for distressed")
    confidence: float = Field(..., description="Confidence score of the prediction (0.0 to 1.0)")
    recommendation: str = Field(..., description="Recommendation based on the prediction")
    sentiment: Dict[str, float] = Field(..., description="Sentiment analysis scores")
    language_features: Dict[str, Any] = Field(..., description="Extracted language features")
    insights: List[Dict[str, str]] = Field(..., description="Insights derived from the analysis")

# Load the model
try:
    # Try to load the transformer model first
    if os.path.exists("app/model/transformer_model"):
        logger.info("Loading transformer model...")
        from load_transformer_model import load_transformer_model
        model = load_transformer_model()
        logger.info("Transformer model loaded successfully")
    else:
        # Fall back to the traditional model if transformer model doesn't exist
        logger.info("Loading traditional model...")
        model = joblib.load("app/model/mental_health_model.pkl")
        logger.info("Traditional model loaded successfully")
        
    # Try to load the sentiment analyzer
    try:
        import nltk
        from nltk.sentiment import SentimentIntensityAnalyzer
        nltk.download('vader_lexicon', quiet=True)
        sentiment_analyzer = SentimentIntensityAnalyzer()
        logger.info("Sentiment analyzer loaded successfully")
    except Exception as e:
        logger.warning(f"Failed to load sentiment analyzer: {str(e)}")
        sentiment_analyzer = None
        
    # Try to load the feature extractor
    try:
        from feature_extraction import FeatureExtractor
        feature_extractor = FeatureExtractor(use_transformers=False)
        logger.info("Feature extractor loaded successfully")
    except Exception as e:
        logger.warning(f"Failed to load feature extractor: {str(e)}")
        feature_extractor = None
        
except Exception as e:
    logger.error(f"Error loading model: {str(e)}")
    logger.error(traceback.format_exc())
    raise RuntimeError(f"Failed to load model: {str(e)}")

# Define recommendations
DISTRESS_RECOMMENDATION = """
Based on your message, you may be experiencing emotional distress. 
Consider reaching out to a mental health professional or a trusted person for support. 
Remember that it's okay to ask for help, and resources like crisis helplines are available 24/7.
"""

NORMAL_RECOMMENDATION = """
Based on your message, you appear to be in a normal emotional state. 
Continue practicing self-care and maintaining your mental well-being through regular exercise, 
adequate sleep, and social connections.
"""

@app.get("/")
def read_root():
    return {"message": "Mental Health Predictor API running"}

@app.post("/predict/", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    try:
        text = request.text_input
        
        # Get prediction
        prediction = int(model.predict([text])[0])
        
        # Get confidence score
        confidence = float(model.predict_proba([text])[0][prediction])
        
        # Generate recommendation based on prediction
        recommendation = DISTRESS_RECOMMENDATION if prediction == 1 else NORMAL_RECOMMENDATION
        
        # Log the prediction (without the full text for privacy)
        logger.info(f"Prediction: {prediction}, Confidence: {confidence:.4f}")
        
        return {
            "prediction": prediction,
            "confidence": confidence,
            "recommendation": recommendation
        }
    except Exception as e:
        logger.error(f"Error in prediction: {str(e)}")
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

@app.post("/analyze/", response_model=AnalysisResponse)
async def analyze(request: AnalysisRequest):
    try:
        text = request.text_input
        
        # Get prediction
        prediction = int(model.predict([text])[0])
        
        # Get confidence score
        confidence = float(model.predict_proba([text])[0][prediction])
        
        # Generate recommendation based on prediction
        recommendation = DISTRESS_RECOMMENDATION if prediction == 1 else NORMAL_RECOMMENDATION
        
        # Analyze sentiment if available
        if sentiment_analyzer:
            sentiment = sentiment_analyzer.polarity_scores(text)
        else:
            sentiment = {
                "neg": 0.0,
                "neu": 0.0,
                "pos": 0.0,
                "compound": 0.0
            }
        
        # Extract language features if available
        if feature_extractor:
            # Extract basic linguistic features
            linguistic_features = feature_extractor.extract_linguistic_features([text]).iloc[0].to_dict()
            
            # Extract emotion features if available
            try:
                emotion_features = feature_extractor.extract_emotion_features([text]).iloc[0].to_dict()
                linguistic_features.update(emotion_features)
            except:
                pass
                
            # Extract mental health indicators if available
            try:
                mh_features = feature_extractor.extract_mental_health_indicators([text]).iloc[0].to_dict()
                linguistic_features.update(mh_features)
            except:
                pass
        else:
            # Basic features if feature extractor not available
            words = text.split()
            sentences = text.split('.')
            
            linguistic_features = {
                "word_count": len(words),
                "sentence_count": len([s for s in sentences if s.strip()]),
                "avg_word_length": np.mean([len(w) for w in words]) if words else 0,
                "question_marks": text.count('?'),
                "exclamation_marks": text.count('!')
            }
        
        # Generate insights
        insights = []
        
        # Sentiment-based insights
        if sentiment["compound"] < -0.5:
            insights.append({
                "type": "warning",
                "text": "Your message shows strong negative emotions. Consider talking to someone you trust about how you're feeling."
            })
        elif sentiment["compound"] < -0.2:
            insights.append({
                "type": "info",
                "text": "Your message indicates some negative emotions. Taking time for self-care might be helpful."
            })
        elif sentiment["compound"] > 0.5:
            insights.append({
                "type": "positive",
                "text": "Your message shows strong positive emotions. That's great! Keep nurturing these positive feelings."
            })
        
        # Language feature insights
        if linguistic_features.get("first_person_pronouns", 0) > 10:
            insights.append({
                "type": "info",
                "text": "You're using many self-referential terms, which can sometimes indicate introspection or self-focus."
            })
            
        if linguistic_features.get("question_marks", 0) > 3:
            insights.append({
                "type": "info",
                "text": "Your message contains several questions, which might indicate uncertainty or seeking answers."
            })
            
        # Add prediction-based insight
        if prediction == 1:
            insights.append({
                "type": "warning",
                "text": "The analysis suggests potential signs of emotional distress in your message."
            })
        else:
            insights.append({
                "type": "positive",
                "text": "The analysis doesn't detect significant signs of emotional distress in your message."
            })
            
        # Add general insight
        insights.append({
            "type": "info",
            "text": "Regular journaling can help track emotional patterns and identify triggers for distress."
        })
        
        # Log the analysis (without the full text for privacy)
        logger.info(f"Analysis: prediction={prediction}, confidence={confidence:.4f}, sentiment={sentiment['compound']:.4f}")
        
        return {
            "prediction": prediction,
            "confidence": confidence,
            "recommendation": recommendation,
            "sentiment": sentiment,
            "language_features": linguistic_features,
            "insights": insights
        }
    except Exception as e:
        logger.error(f"Error in analysis: {str(e)}")
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Analysis error: {str(e)}")

# Legacy endpoint for backward compatibility
@app.post("/predict/legacy")
async def predict_legacy(request: Request):
    try:
        data = await request.json()
        text = data.get("text_input", "")
        
        if not text or not text.strip():
            raise HTTPException(status_code=400, detail="Missing or empty text_input field")
        
        # Get prediction
        prediction = int(model.predict([text])[0])
        
        # Log the prediction (without the full text for privacy)
        logger.info(f"Legacy prediction: {prediction}")
        
        # Return just the prediction for backward compatibility
        return {"prediction": prediction}
    except HTTPException:
        # Re-raise HTTP exceptions
        raise
    except Exception as e:
        logger.error(f"Error in legacy prediction: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")
