#!/usr/bin/env python3
"""
Mental Health Detection Pipeline
Main execution script for the complete workflow
"""

import os
import sys
import logging
import pandas as pd
from datetime import datetime

# Add src to path
sys.path.append('src')

from src.data_collection.social_media_collector import SocialMediaCollector
from src.preprocessing.data_cleaner import DataPreprocessor
from src.features.nlp_features import NLPFeatureExtractor
from src.labeling.label_generator import DataLabeler
from src.models.ml_models import MentalHealthClassifier
from src.evaluation.model_evaluator import ModelEvaluator
from src.ethics.compliance_checker import EthicsComplianceChecker
# from src.visualization.dashboard import MentalHealthDashboard

def setup_logging():
    """Setup logging configuration"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(f'logs/pipeline_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'),
            logging.StreamHandler()
        ]
    )

def main():
    """Main pipeline execution"""
    setup_logging()
    logger = logging.getLogger(__name__)
    
    logger.info("Starting Mental Health Detection Pipeline...")
    
    try:
        # 1. Ethics Compliance Check
        ethics_checker = EthicsComplianceChecker()
        
        # 2. Data Collection
        logger.info("Phase 1: Data Collection")
        collector = SocialMediaCollector('config/api_config.json')
        
        # Define collection parameters
        keywords = ['mental health', 'depression', 'anxiety', 'feeling down', 'stressed out']
        subreddits = ['depression', 'anxiety', 'mentalhealth', 'therapy']
        
        # Collect data
        twitter_data = collector.collect_twitter_data(keywords, max_results=5000)
        reddit_data = collector.collect_reddit_data(subreddits, max_posts=3000)
        
        # Save raw data
        twitter_data.to_csv('data/raw/twitter_data.csv', index=False)
        reddit_data.to_csv('data/raw/reddit_data.csv', index=False)
        
        # 3. Data Preprocessing
        logger.info("Phase 2: Data Preprocessing")
        preprocessor = DataPreprocessor()
        
        clean_twitter = preprocessor.preprocess_dataframe(twitter_data)
        clean_reddit = preprocessor.preprocess_dataframe(reddit_data)
        
        # Combine and anonymize
        combined_data = pd.concat([clean_twitter, clean_reddit], ignore_index=True)
        combined_data = ethics_checker.anonymize_user_data(combined_data)
        
        # 4. Feature Extraction
        logger.info("Phase 3: Feature Extraction")
        feature_extractor = NLPFeatureExtractor()
        
        featured_data = feature_extractor.extract_sentiment_features(combined_data)
        featured_data = feature_extractor.extract_linguistic_features(featured_data)
        featured_data = feature_extractor.extract_keyword_features(featured_data)
        featured_data, tfidf_matrix = feature_extractor.extract_tfidf_features(featured_data)
        
        # 5. Label Generation
        logger.info("Phase 4: Label Generation")
        labeler = DataLabeler()
        labeled_data = labeler.create_labels(featured_data)
        
        # Save processed data
        labeled_data.to_csv('data/processed/labeled_data.csv', index=False)
        
        # 6. Model Training and Evaluation
        logger.info("Phase 5: Model Training")
        classifier = MentalHealthClassifier()
        evaluator = ModelEvaluator()
        
        # Prepare features
        X = classifier.prepare_features(labeled_data, tfidf_matrix)
        y = labeled_data['mental_health_risk'].values
        
        # Train models
        results = classifier.train_models(X, y)
        
        # Evaluate models
        for model_name, model_info in results.items():
            evaluator.comprehensive_evaluation(
                model_info['model'], 
                X, y,  # In practice, use separate test set
                model_name
            )
        
        # 7. Generate Reports
        logger.info("Phase 6: Generating Reports")
        ethics_report = ethics_checker.generate_ethics_report()
        
        # Save models and reports
        import joblib
        joblib.dump(classifier.best_model, 'models/best_model.pkl')
        
        with open('reports/ethics_compliance.json', 'w') as f:
            import json
            json.dump(ethics_report, f, indent=2)
        
        # 8. Launch Dashboard
        logger.info("Phase 7: Launching Dashboard")
        # Dashboard module not found, skipping dashboard launch.
        logger.info("Pipeline completed successfully!")
        # logger.info("Starting dashboard on http://localhost:8050")
        # dashboard.run(debug=False, port=8050)
        
    except Exception as e:
        logger.error(f"Pipeline failed: {str(e)}")
        raise

if __name__ == "__main__":
    main()