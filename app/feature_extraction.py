"""
Feature extraction module for mental health prediction.
This module extracts relevant features from text data using NLP techniques.
"""
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation, TruncatedSVD
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
import logging
import re
import os
from collections import Counter
from transformers import pipeline

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Ensure NLTK resources are downloaded
try:
    nltk.download('vader_lexicon', quiet=True)
except Exception as e:
    logger.warning(f"Failed to download NLTK resources: {str(e)}")

class FeatureExtractor:
    """
    Class for extracting features from text data for mental health analysis.
    """
    def __init__(self, use_transformers=True):
        """
        Initialize the feature extractor.
        
        Args:
            use_transformers (bool): Whether to use transformer models for feature extraction
        """
        self.use_transformers = use_transformers
        
        # Initialize sentiment analyzer
        try:
            self.sentiment_analyzer = SentimentIntensityAnalyzer()
        except Exception as e:
            logger.warning(f"Failed to initialize sentiment analyzer: {str(e)}")
            self.sentiment_analyzer = None
        
        # Initialize transformer pipelines if requested
        if use_transformers:
            try:
                # Zero-shot classification pipeline
                self.zero_shot = pipeline("zero-shot-classification", 
                                         model="facebook/bart-large-mnli",
                                         device=-1)  # Use CPU
                
                # Emotion detection pipeline
                self.emotion_classifier = pipeline("text-classification", 
                                                 model="j-hartmann/emotion-english-distilroberta-base",
                                                 device=-1)  # Use CPU
            except Exception as e:
                logger.warning(f"Failed to initialize transformer pipelines: {str(e)}")
                self.zero_shot = None
                self.emotion_classifier = None
    
    def extract_sentiment_features(self, texts):
        """
        Extract sentiment features from texts using VADER.
        
        Args:
            texts (list): List of text strings
            
        Returns:
            pd.DataFrame: DataFrame with sentiment features
        """
        if not self.sentiment_analyzer:
            logger.warning("Sentiment analyzer not available")
            return pd.DataFrame(index=range(len(texts)))
        
        sentiment_features = []
        
        for text in texts:
            if not isinstance(text, str) or not text.strip():
                sentiment_features.append({
                    'sentiment_neg': 0.0,
                    'sentiment_neu': 0.0,
                    'sentiment_pos': 0.0,
                    'sentiment_compound': 0.0
                })
                continue
                
            scores = self.sentiment_analyzer.polarity_scores(text)
            sentiment_features.append({
                'sentiment_neg': scores['neg'],
                'sentiment_neu': scores['neu'],
                'sentiment_pos': scores['pos'],
                'sentiment_compound': scores['compound']
            })
        
        return pd.DataFrame(sentiment_features)
    
    def extract_linguistic_features(self, texts):
        """
        Extract linguistic features from texts.
        
        Args:
            texts (list): List of text strings
            
        Returns:
            pd.DataFrame: DataFrame with linguistic features
        """
        features = []
        
        for text in texts:
            if not isinstance(text, str):
                text = ""
                
            # Count words
            words = text.split()
            word_count = len(words)
            
            # Count sentences
            sentences = re.split(r'[.!?]+', text)
            sentence_count = len([s for s in sentences if s.strip()])
            
            # Calculate average word length
            avg_word_length = np.mean([len(word) for word in words]) if words else 0
            
            # Calculate average sentence length
            avg_sentence_length = np.mean([len(s.split()) for s in sentences if s.strip()]) if sentence_count > 0 else 0
            
            # Count question marks and exclamation marks
            question_marks = text.count('?')
            exclamation_marks = text.count('!')
            
            # Count first-person pronouns
            first_person_pronouns = len(re.findall(r'\b(i|me|my|mine|myself)\b', text.lower()))
            
            # Count negative words (simplified approach)
            negative_words = ['no', 'not', 'never', 'none', 'nothing', 'nowhere', 'nobody', 'cant', "can't", 
                             'wont', "won't", 'dont', "don't", 'isnt', "isn't"]
            negative_count = sum(1 for word in words if word.lower() in negative_words)
            
            features.append({
                'word_count': word_count,
                'sentence_count': sentence_count,
                'avg_word_length': avg_word_length,
                'avg_sentence_length': avg_sentence_length,
                'question_marks': question_marks,
                'exclamation_marks': exclamation_marks,
                'first_person_pronouns': first_person_pronouns,
                'negative_words': negative_count,
                'negative_ratio': negative_count / word_count if word_count > 0 else 0
            })
        
        return pd.DataFrame(features)
    
    def extract_emotion_features(self, texts):
        """
        Extract emotion features using transformer models.
        
        Args:
            texts (list): List of text strings
            
        Returns:
            pd.DataFrame: DataFrame with emotion features
        """
        if not self.use_transformers or not self.emotion_classifier:
            logger.warning("Emotion classifier not available")
            return pd.DataFrame(index=range(len(texts)))
        
        emotion_features = []
        
        # Process in batches to avoid memory issues
        batch_size = 8
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i+batch_size]
            valid_texts = [text if isinstance(text, str) and text.strip() else "neutral" for text in batch_texts]
            
            try:
                results = self.emotion_classifier(valid_texts)
                
                for result in results:
                    emotion = result['label']
                    score = result['score']
                    
                    feature = {
                        'emotion': emotion,
                        'emotion_score': score
                    }
                    emotion_features.append(feature)
            except Exception as e:
                logger.error(f"Error in emotion classification: {str(e)}")
                # Add placeholder features for this batch
                for _ in range(len(batch_texts)):
                    emotion_features.append({
                        'emotion': 'unknown',
                        'emotion_score': 0.0
                    })
        
        return pd.DataFrame(emotion_features)
    
    def extract_mental_health_indicators(self, texts):
        """
        Extract features specifically related to mental health using zero-shot classification.
        
        Args:
            texts (list): List of text strings
            
        Returns:
            pd.DataFrame: DataFrame with mental health indicator features
        """
        if not self.use_transformers or not self.zero_shot:
            logger.warning("Zero-shot classifier not available")
            return pd.DataFrame(index=range(len(texts)))
        
        # Define mental health categories
        categories = [
            "depression", "anxiety", "stress", "loneliness", 
            "hopelessness", "suicidal thoughts", "insomnia",
            "positive outlook", "happiness", "contentment"
        ]
        
        indicator_features = []
        
        # Process in batches to avoid memory issues
        batch_size = 4
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i+batch_size]
            valid_texts = [text if isinstance(text, str) and text.strip() else "neutral" for text in batch_texts]
            
            try:
                for text in valid_texts:
                    result = self.zero_shot(text, categories)
                    
                    # Create a dictionary of category scores
                    scores = {f"mh_{cat.replace(' ', '_')}": score 
                             for cat, score in zip(result['labels'], result['scores'])}
                    
                    indicator_features.append(scores)
            except Exception as e:
                logger.error(f"Error in zero-shot classification: {str(e)}")
                # Add placeholder features for this batch
                for _ in range(len(batch_texts)):
                    scores = {f"mh_{cat.replace(' ', '_')}": 0.0 for cat in categories}
                    indicator_features.append(scores)
        
        return pd.DataFrame(indicator_features)
    
    def extract_tfidf_features(self, texts, max_features=100):
        """
        Extract TF-IDF features from texts.
        
        Args:
            texts (list): List of text strings
            max_features (int): Maximum number of features to extract
            
        Returns:
            scipy.sparse.csr.csr_matrix: Sparse matrix of TF-IDF features
        """
        valid_texts = [text if isinstance(text, str) else "" for text in texts]
        
        vectorizer = TfidfVectorizer(
            max_features=max_features,
            min_df=2,
            max_df=0.85,
            stop_words='english',
            ngram_range=(1, 2)
        )
        
        try:
            tfidf_features = vectorizer.fit_transform(valid_texts)
            feature_names = vectorizer.get_feature_names_out()
            logger.info(f"Extracted {len(feature_names)} TF-IDF features")
            return tfidf_features, feature_names
        except Exception as e:
            logger.error(f"Error extracting TF-IDF features: {str(e)}")
            return None, []
    
    def extract_topic_features(self, texts, n_topics=5):
        """
        Extract topic features using Latent Dirichlet Allocation.
        
        Args:
            texts (list): List of text strings
            n_topics (int): Number of topics to extract
            
        Returns:
            pd.DataFrame: DataFrame with topic features
        """
        valid_texts = [text if isinstance(text, str) else "" for text in texts]
        
        # Create a document-term matrix
        vectorizer = CountVectorizer(
            max_features=1000,
            min_df=2,
            max_df=0.85,
            stop_words='english'
        )
        
        try:
            dtm = vectorizer.fit_transform(valid_texts)
            
            # Create and fit the LDA model
            lda = LatentDirichletAllocation(
                n_components=n_topics,
                random_state=42,
                max_iter=10
            )
            
            topic_distributions = lda.fit_transform(dtm)
            
            # Create a DataFrame with topic features
            topic_columns = [f'topic_{i}' for i in range(n_topics)]
            topic_df = pd.DataFrame(topic_distributions, columns=topic_columns)
            
            # Get the top words for each topic for interpretability
            feature_names = vectorizer.get_feature_names_out()
            top_words_per_topic = []
            
            for topic_idx, topic in enumerate(lda.components_):
                top_words_idx = topic.argsort()[:-10-1:-1]
                top_words = [feature_names[i] for i in top_words_idx]
                top_words_per_topic.append((topic_idx, top_words))
                logger.info(f"Topic {topic_idx}: {', '.join(top_words)}")
            
            return topic_df, top_words_per_topic
        except Exception as e:
            logger.error(f"Error extracting topic features: {str(e)}")
            return pd.DataFrame(index=range(len(texts))), []
    
    def extract_all_features(self, texts, include_tfidf=True, include_topics=True):
        """
        Extract all available features from texts.
        
        Args:
            texts (list): List of text strings
            include_tfidf (bool): Whether to include TF-IDF features
            include_topics (bool): Whether to include topic features
            
        Returns:
            pd.DataFrame: DataFrame with all extracted features
        """
        logger.info(f"Extracting features from {len(texts)} texts")
        
        # Extract basic features
        sentiment_features = self.extract_sentiment_features(texts)
        linguistic_features = self.extract_linguistic_features(texts)
        
        # Combine basic features
        features_df = pd.concat([sentiment_features, linguistic_features], axis=1)
        
        # Add transformer-based features if available
        if self.use_transformers:
            try:
                emotion_features = self.extract_emotion_features(texts)
                features_df = pd.concat([features_df, emotion_features], axis=1)
                
                mental_health_features = self.extract_mental_health_indicators(texts)
                features_df = pd.concat([features_df, mental_health_features], axis=1)
            except Exception as e:
                logger.error(f"Error extracting transformer features: {str(e)}")
        
        # Add TF-IDF features if requested
        if include_tfidf:
            try:
                tfidf_features, feature_names = self.extract_tfidf_features(texts)
                if tfidf_features is not None:
                    # Convert sparse matrix to DataFrame
                    tfidf_df = pd.DataFrame.sparse.from_spmatrix(
                        tfidf_features,
                        columns=[f'tfidf_{name}' for name in feature_names]
                    )
                    features_df = pd.concat([features_df, tfidf_df], axis=1)
            except Exception as e:
                logger.error(f"Error adding TF-IDF features: {str(e)}")
        
        # Add topic features if requested
        if include_topics:
            try:
                topic_df, _ = self.extract_topic_features(texts)
                features_df = pd.concat([features_df, topic_df], axis=1)
            except Exception as e:
                logger.error(f"Error adding topic features: {str(e)}")
        
        logger.info(f"Extracted {features_df.shape[1]} total features")
        return features_df


# Example usage
if __name__ == "__main__":
    # Sample texts
    texts = [
        "I feel so depressed today. Nothing seems to matter anymore.",
        "Had a great day at the park with friends. Feeling refreshed!",
        "Anxiety is overwhelming me. Can't stop worrying about everything.",
        "Just finished a good book and feeling satisfied with life.",
        "I don't know if I can go on like this. Everything feels hopeless."
    ]
    
    # Initialize feature extractor
    extractor = FeatureExtractor(use_transformers=True)
    
    # Extract features
    features = extractor.extract_all_features(texts, include_tfidf=True, include_topics=True)
    
    print(f"Extracted {features.shape[1]} features for {features.shape[0]} texts")
    print("Feature columns:", features.columns.tolist()[:10], "...")  # Show first 10 columns