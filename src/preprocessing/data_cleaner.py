import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import string
from typing import List, Dict
import logging

class DataPreprocessor:
    def __init__(self):
        """Initialize data preprocessor with NLTK resources"""
        self.setup_nltk()
        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = set(stopwords.words('english'))
        self.logger = logging.getLogger(__name__)
        
        # Patterns for spam/advertisement detection
        self.spam_patterns = [
            r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+',
            r'@\w+',  # mentions
            r'#\w+',  # hashtags (optional removal)
            r'\b(?:buy|sale|discount|offer|deal|click|link)\b',  # commercial keywords
        ]
    
    def setup_nltk(self):
        """Download required NLTK data"""
        nltk_downloads = ['punkt', 'stopwords', 'wordnet', 'vader_lexicon']
        for item in nltk_downloads:
            try:
                nltk.data.find(f'tokenizers/{item}')
            except LookupError:
                nltk.download(item)
    
    def clean_text(self, text: str) -> str:
        """Clean individual text entry"""
        if pd.isna(text) or text == '':
            return ''
        
        # Convert to lowercase
        text = text.lower()
        
        # Remove URLs
        text = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', text)
        
        # Remove mentions and hashtags (optional - might contain useful info)
        text = re.sub(r'@\w+', '', text)
        text = re.sub(r'#\w+', '', text)
        
        # Remove punctuation except emoticons
        text = re.sub(r'[^\w\s:;=\-\)\(\[\]]+', '', text)
        
        # Remove extra whitespace
        text = ' '.join(text.split())
        
        return text
    
    def detect_spam(self, text: str) -> bool:
        """Detect if text is likely spam or advertisement"""
        spam_indicators = 0
        
        for pattern in self.spam_patterns:
            if re.search(pattern, text.lower()):
                spam_indicators += 1
        
        # Simple heuristic: if multiple spam patterns detected
        return spam_indicators >= 2
    
    def tokenize_and_lemmatize(self, text: str) -> List[str]:
        """Tokenize and lemmatize text"""
        if not text:
            return []
        
        tokens = word_tokenize(text)
        tokens = [self.lemmatizer.lemmatize(token) for token in tokens 
                 if token not in self.stop_words and len(token) > 2]
        
        return tokens
    
    def preprocess_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """Preprocess entire dataframe"""
        self.logger.info("Starting data preprocessing...")
        
        # Create a copy to avoid modifying original
        processed_df = df.copy()
        
        # Combine text fields for analysis
        if 'title' in processed_df.columns:
            processed_df['combined_text'] = processed_df['title'].fillna('') + ' ' + processed_df['text'].fillna('')
        else:
            processed_df['combined_text'] = processed_df['text'].fillna('')
        
        # Remove empty posts
        processed_df = processed_df[processed_df['combined_text'].str.strip() != '']
        
        # Detect and remove spam
        processed_df['is_spam'] = processed_df['combined_text'].apply(self.detect_spam)
        processed_df = processed_df[~processed_df['is_spam']]
        
        # Clean text
        processed_df['cleaned_text'] = processed_df['combined_text'].apply(self.clean_text)
        
        # Remove posts that are too short after cleaning
        processed_df = processed_df[processed_df['cleaned_text'].str.len() > 10]
        
        # Tokenize
        processed_df['tokens'] = processed_df['cleaned_text'].apply(self.tokenize_and_lemmatize)
        
        # Remove posts with too few tokens
        processed_df = processed_df[processed_df['tokens'].apply(len) >= 3]
        
        self.logger.info(f"Preprocessing complete. {len(processed_df)} posts remaining from {len(df)} original posts.")
        
        return processed_df
    
    def anonymize_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Anonymize user data for privacy"""
        anonymized_df = df.copy()
        
        # Hash user IDs
        if 'author_id' in anonymized_df.columns:
            anonymized_df['user_hash'] = pd.util.hash_pandas_object(anonymized_df['author_id'])
            anonymized_df = anonymized_df.drop('author_id', axis=1)
        
        if 'author' in anonymized_df.columns:
            anonymized_df['user_hash'] = pd.util.hash_pandas_object(anonymized_df['author'])
            anonymized_df = anonymized_df.drop('author', axis=1)
        
        return anonymized_df