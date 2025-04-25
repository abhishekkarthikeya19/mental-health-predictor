"""
Data collection and preprocessing module for mental health prediction.
This module handles collecting social media data and preprocessing it for analysis.
"""
import re
import pandas as pd
import numpy as np
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import logging
import os
import json
from datetime import datetime
import nltk

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Ensure NLTK resources are downloaded
try:
    nltk.download('punkt', quiet=True)
    nltk.download('stopwords', quiet=True)
    nltk.download('wordnet', quiet=True)
except Exception as e:
    logger.warning(f"Failed to download NLTK resources: {str(e)}")

class DataPreprocessor:
    """
    Class for preprocessing text data from social media sources.
    """
    def __init__(self):
        """Initialize the preprocessor with necessary resources."""
        try:
            self.stop_words = set(stopwords.words('english'))
            self.lemmatizer = WordNetLemmatizer()
        except LookupError:
            logger.warning("NLTK resources not available. Some preprocessing features will be limited.")
            self.stop_words = set()
            self.lemmatizer = None
    
    def clean_text(self, text):
        """
        Clean and normalize text data.
        
        Args:
            text (str): Raw text input
            
        Returns:
            str: Cleaned text
        """
        if not isinstance(text, str):
            return ""
            
        # Convert to lowercase
        text = text.lower()
        
        # Remove URLs
        text = re.sub(r'https?://\S+|www\.\S+', '', text)
        
        # Remove user mentions (e.g., @username)
        text = re.sub(r'@\w+', '', text)
        
        # Remove hashtags symbols but keep the text
        text = re.sub(r'#(\w+)', r'\1', text)
        
        # Remove non-alphanumeric characters
        text = re.sub(r'[^\w\s]', '', text)
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
    
    def remove_stopwords(self, text):
        """
        Remove common stopwords from text.
        
        Args:
            text (str): Input text
            
        Returns:
            str: Text with stopwords removed
        """
        if not self.stop_words:
            return text
            
        word_tokens = word_tokenize(text)
        filtered_text = [word for word in word_tokens if word not in self.stop_words]
        return ' '.join(filtered_text)
    
    def lemmatize_text(self, text):
        """
        Lemmatize text to reduce words to their base form.
        
        Args:
            text (str): Input text
            
        Returns:
            str: Lemmatized text
        """
        if not self.lemmatizer:
            return text
            
        word_tokens = word_tokenize(text)
        lemmatized_text = [self.lemmatizer.lemmatize(word) for word in word_tokens]
        return ' '.join(lemmatized_text)
    
    def preprocess(self, text, remove_stops=True, lemmatize=True):
        """
        Apply full preprocessing pipeline to text.
        
        Args:
            text (str): Raw input text
            remove_stops (bool): Whether to remove stopwords
            lemmatize (bool): Whether to lemmatize text
            
        Returns:
            str: Fully preprocessed text
        """
        # Clean the text
        cleaned_text = self.clean_text(text)
        
        # Remove stopwords if requested
        if remove_stops:
            cleaned_text = self.remove_stopwords(cleaned_text)
        
        # Lemmatize if requested
        if lemmatize:
            cleaned_text = self.lemmatize_text(cleaned_text)
        
        return cleaned_text
    
    def batch_preprocess(self, texts, remove_stops=True, lemmatize=True):
        """
        Preprocess a batch of texts.
        
        Args:
            texts (list): List of text strings
            remove_stops (bool): Whether to remove stopwords
            lemmatize (bool): Whether to lemmatize text
            
        Returns:
            list: List of preprocessed texts
        """
        return [self.preprocess(text, remove_stops, lemmatize) for text in texts]


class SocialMediaCollector:
    """
    Class for collecting data from social media platforms.
    This is a placeholder implementation that would be connected to actual APIs in production.
    """
    def __init__(self, data_dir="app/data"):
        """
        Initialize the collector with a directory to store data.
        
        Args:
            data_dir (str): Directory to store collected data
        """
        self.data_dir = data_dir
        os.makedirs(data_dir, exist_ok=True)
        self.preprocessor = DataPreprocessor()
    
    def collect_twitter_data(self, keywords=None, count=100, preprocess=True):
        """
        Collect data from Twitter (placeholder implementation).
        In a real implementation, this would use the Twitter API.
        
        Args:
            keywords (list): List of keywords to search for
            count (int): Number of tweets to collect
            preprocess (bool): Whether to preprocess the collected data
            
        Returns:
            pd.DataFrame: Collected and optionally preprocessed data
        """
        logger.info(f"Collecting {count} tweets with keywords: {keywords}")
        
        # This is a placeholder. In a real implementation, this would call the Twitter API
        # For now, we'll return a sample dataset
        sample_data = {
            "text": [
                "I'm feeling really down today, nothing seems to help #depression",
                "Just had a great therapy session! Making progress with my anxiety",
                "Can't sleep again. Third night in a row. My mind won't stop racing",
                "Feeling blessed today. Grateful for my support system",
                "Why does everything feel so overwhelming? I can't cope anymore"
            ],
            "timestamp": [datetime.now().isoformat() for _ in range(5)],
            "platform": ["twitter" for _ in range(5)]
        }
        
        df = pd.DataFrame(sample_data)
        
        if preprocess:
            df["processed_text"] = self.preprocessor.batch_preprocess(df["text"].tolist())
        
        # Save the collected data
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = os.path.join(self.data_dir, f"twitter_data_{timestamp}.csv")
        df.to_csv(output_file, index=False)
        logger.info(f"Data saved to {output_file}")
        
        return df
    
    def collect_reddit_data(self, subreddits=None, count=100, preprocess=True):
        """
        Collect data from Reddit (placeholder implementation).
        In a real implementation, this would use the Reddit API.
        
        Args:
            subreddits (list): List of subreddits to collect from
            count (int): Number of posts to collect
            preprocess (bool): Whether to preprocess the collected data
            
        Returns:
            pd.DataFrame: Collected and optionally preprocessed data
        """
        logger.info(f"Collecting {count} posts from subreddits: {subreddits}")
        
        # This is a placeholder. In a real implementation, this would call the Reddit API
        sample_data = {
            "text": [
                "I've been struggling with depression for years and nothing seems to help",
                "Started a new medication last week and I'm already feeling better",
                "Does anyone else feel like they're just going through the motions?",
                "I finally opened up to my family about my mental health and they were so supportive",
                "Some days are harder than others, but I'm trying to stay positive"
            ],
            "timestamp": [datetime.now().isoformat() for _ in range(5)],
            "platform": ["reddit" for _ in range(5)],
            "subreddit": ["depression", "mentalhealth", "anxiety", "mentalhealth", "depression"]
        }
        
        df = pd.DataFrame(sample_data)
        
        if preprocess:
            df["processed_text"] = self.preprocessor.batch_preprocess(df["text"].tolist())
        
        # Save the collected data
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = os.path.join(self.data_dir, f"reddit_data_{timestamp}.csv")
        df.to_csv(output_file, index=False)
        logger.info(f"Data saved to {output_file}")
        
        return df
    
    def load_local_dataset(self, file_path, preprocess=True):
        """
        Load a dataset from a local file.
        
        Args:
            file_path (str): Path to the dataset file (CSV or JSON)
            preprocess (bool): Whether to preprocess the text data
            
        Returns:
            pd.DataFrame: Loaded and optionally preprocessed data
        """
        logger.info(f"Loading dataset from {file_path}")
        
        if file_path.endswith('.csv'):
            df = pd.read_csv(file_path)
        elif file_path.endswith('.json'):
            df = pd.read_json(file_path)
        else:
            raise ValueError("Unsupported file format. Please provide a CSV or JSON file.")
        
        if "text" not in df.columns:
            raise ValueError("Dataset must contain a 'text' column")
        
        if preprocess:
            df["processed_text"] = self.preprocessor.batch_preprocess(df["text"].tolist())
        
        return df
    
    def combine_datasets(self, datasets):
        """
        Combine multiple datasets into one.
        
        Args:
            datasets (list): List of pandas DataFrames to combine
            
        Returns:
            pd.DataFrame: Combined dataset
        """
        if not datasets:
            return pd.DataFrame()
            
        combined_df = pd.concat(datasets, ignore_index=True)
        logger.info(f"Combined {len(datasets)} datasets with {len(combined_df)} total records")
        
        return combined_df


# Example usage
if __name__ == "__main__":
    # Initialize the data collector
    collector = SocialMediaCollector()
    
    # Collect sample data
    twitter_data = collector.collect_twitter_data(
        keywords=["depression", "anxiety", "mental health"],
        count=100
    )
    
    reddit_data = collector.collect_reddit_data(
        subreddits=["depression", "anxiety", "mentalhealth"],
        count=100
    )
    
    # Combine datasets
    combined_data = collector.combine_datasets([twitter_data, reddit_data])
    
    print(f"Collected {len(combined_data)} total records")
    print("Sample processed text:")
    for text in combined_data["processed_text"].head(3):
        print(f"- {text}")