import logging
import pandas as pd
from src.data_collection.social_media_collector import SocialMediaCollector
from src.preprocessing.data_cleaner import DataPreprocessor
from src.features.nlp_features import NLPFeatureExtractor
from src.labeling.label_generator import DataLabeler
from src.models.ml_models import MentalHealthClassifier

def main():
    # Setup logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    
    # 1. Data Collection
    logger.info("Starting data collection...")
    collector = SocialMediaCollector('config/api_config.json')
    
    # Collect data
    mental_health_keywords = ['depression', 'anxiety', 'mental health', 'feeling down']
    mental_health_subreddits = ['depression', 'anxiety', 'mentalhealth']
    
    twitter_data = collector.collect_twitter_data(mental_health_keywords, max_results=5000)
    reddit_data = collector.collect_reddit_data(mental_health_subreddits, max_posts=2000)
    
    # 2. Data Preprocessing
    logger.info("Starting data preprocessing...")
    preprocessor = DataPreprocessor()
    
    clean_twitter = preprocessor.preprocess_dataframe(twitter_data)
    clean_reddit = preprocessor.preprocess_dataframe(reddit_data)
    
    # Combine datasets
    combined_data = pd.concat([clean_twitter, clean_reddit], ignore_index=True)
    
    # 3. Feature Extraction
    logger.info("Extracting features...")
    feature_extractor = NLPFeatureExtractor()
    
    featured_data = feature_extractor.extract_sentiment_features(combined_data)
    featured_data = feature_extractor.extract_linguistic_features(featured_data)
    featured_data = feature_extractor.extract_keyword_features(featured_data)
    featured_data, tfidf_matrix = feature_extractor.extract_tfidf_features(featured_data)
    
    # 4. Label Generation
    logger.info("Generating labels...")
    labeler = DataLabeler()
    labeled_data = labeler.create_labels(featured_data)
    
    # 5. Model Training
    logger.info("Training models...")
    classifier = MentalHealthClassifier()
    X = classifier.prepare_features(labeled_data, tfidf_matrix)
    y = labeled_data['mental_health_risk'].values
    
    results = classifier.train_models(X, y)
    
    logger.info("Pipeline completed successfully!")

if __name__ == "__main__":
    main()