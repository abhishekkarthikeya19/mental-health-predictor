import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from nltk.sentiment import SentimentIntensityAnalyzer
from textstat import flesch_reading_ease, flesch_kincaid_grade
import re
from typing import Dict, List, Tuple
import logging

class NLPFeatureExtractor:
    def __init__(self):
        """Initialize NLP feature extractor"""
        self.sia = SentimentIntensityAnalyzer()
        self.tfidf_vectorizer = TfidfVectorizer(max_features=1000, ngram_range=(1, 2))
        self.logger = logging.getLogger(__name__)
        
        # Mental health related keywords
        self.mental_health_keywords = {
            'depression': ['depressed', 'sad', 'hopeless', 'worthless', 'empty', 'numb'],
            'anxiety': ['anxious', 'worried', 'panic', 'nervous', 'scared', 'fear'],
            'stress': ['stressed', 'overwhelmed', 'pressure', 'burden', 'exhausted'],
            'isolation': ['alone', 'lonely', 'isolated', 'nobody', 'abandoned'],
            'self_harm': ['hurt myself', 'self harm', 'cut myself', 'end it all'],
            'positive': ['happy', 'grateful', 'blessed', 'excited', 'joy', 'love']
        }
    
    def extract_sentiment_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Extract sentiment-based features"""
        self.logger.info("Extracting sentiment features...")
        
        sentiment_scores = df['cleaned_text'].apply(lambda x: self.sia.polarity_scores(x))
        
        df['sentiment_compound'] = sentiment_scores.apply(lambda x: x['compound'])
        df['sentiment_positive'] = sentiment_scores.apply(lambda x: x['pos'])
        df['sentiment_negative'] = sentiment_scores.apply(lambda x: x['neg'])
        df['sentiment_neutral'] = sentiment_scores.apply(lambda x: x['neu'])
        
        return df
    
    def extract_linguistic_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Extract linguistic style features"""
        self.logger.info("Extracting linguistic features...")
        
        # Text length features
        df['text_length'] = df['cleaned_text'].str.len()
        df['word_count'] = df['cleaned_text'].str.split().str.len()
        df['avg_word_length'] = df['cleaned_text'].apply(
            lambda x: np.mean([len(word) for word in x.split()]) if x.split() else 0
        )
        
        # Readability features
        df['flesch_reading_ease'] = df['cleaned_text'].apply(
            lambda x: flesch_reading_ease(x) if len(x) > 0 else 0
        )
        df['flesch_kincaid_grade'] = df['cleaned_text'].apply(
            lambda x: flesch_kincaid_grade(x) if len(x) > 0 else 0
        )
        
        # Punctuation and capitalization features
        df['exclamation_count'] = df['combined_text'].str.count('!')
        df['question_count'] = df['combined_text'].str.count('\?')
        df['caps_ratio'] = df['combined_text'].apply(
            lambda x: sum(1 for c in x if c.isupper()) / len(x) if len(x) > 0 else 0
        )
        
        # Personal pronoun usage (indicator of self-focus)
        first_person_pronouns = r'\b(i|me|my|mine|myself)\b'
        df['first_person_count'] = df['cleaned_text'].str.count(first_person_pronouns, flags=re.IGNORECASE)
        df['first_person_ratio'] = df['first_person_count'] / df['word_count']
        
        return df
    
    def extract_keyword_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Extract mental health keyword features"""
        self.logger.info("Extracting keyword features...")
        
        for category, keywords in self.mental_health_keywords.items():
            pattern = r'\b(' + '|'.join(keywords) + r')\b'
            df[f'{category}_keywords'] = df['cleaned_text'].str.count(pattern, flags=re.IGNORECASE)