"""
Sentiment and trend analysis module for mental health prediction.
This module analyzes sentiment trends over time and identifies patterns in language use.
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from nltk.sentiment import SentimentIntensityAnalyzer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.probability import FreqDist
from collections import Counter
import re
import os
import logging
from datetime import datetime, timedelta
import nltk
from wordcloud import WordCloud
import json
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Ensure NLTK resources are downloaded
try:
    nltk.download('vader_lexicon', quiet=True)
    nltk.download('punkt', quiet=True)
    nltk.download('stopwords', quiet=True)
except Exception as e:
    logger.warning(f"Failed to download NLTK resources: {str(e)}")

class SentimentAnalyzer:
    """
    Class for analyzing sentiment in text data and tracking trends over time.
    """
    def __init__(self, output_dir="app/analysis_results"):
        """
        Initialize the sentiment analyzer.
        
        Args:
            output_dir (str): Directory to save analysis results
        """
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        try:
            self.sia = SentimentIntensityAnalyzer()
            self.stop_words = set(stopwords.words('english'))
        except Exception as e:
            logger.warning(f"Failed to initialize sentiment analyzer: {str(e)}")
            self.sia = None
            self.stop_words = set()
    
    def analyze_sentiment(self, text):
        """
        Analyze the sentiment of a single text.
        
        Args:
            text (str): Text to analyze
            
        Returns:
            dict: Dictionary of sentiment scores
        """
        if not self.sia or not isinstance(text, str) or not text.strip():
            return {
                'neg': 0.0,
                'neu': 0.0,
                'pos': 0.0,
                'compound': 0.0
            }
        
        return self.sia.polarity_scores(text)
    
    def batch_analyze_sentiment(self, texts):
        """
        Analyze sentiment for a batch of texts.
        
        Args:
            texts (list): List of text strings
            
        Returns:
            list: List of sentiment score dictionaries
        """
        return [self.analyze_sentiment(text) for text in texts]
    
    def analyze_sentiment_trends(self, texts, timestamps, time_window='day'):
        """
        Analyze sentiment trends over time.
        
        Args:
            texts (list): List of text strings
            timestamps (list): List of timestamp strings or datetime objects
            time_window (str): Time window for aggregation ('hour', 'day', 'week', 'month')
            
        Returns:
            pd.DataFrame: DataFrame with sentiment trends over time
        """
        # Convert timestamps to datetime if they are strings
        if timestamps and isinstance(timestamps[0], str):
            timestamps = [pd.to_datetime(ts) for ts in timestamps]
        
        # Create a DataFrame with texts and timestamps
        df = pd.DataFrame({
            'text': texts,
            'timestamp': timestamps
        })
        
        # Analyze sentiment for each text
        sentiments = self.batch_analyze_sentiment(texts)
        
        # Add sentiment scores to the DataFrame
        df['sentiment_neg'] = [s['neg'] for s in sentiments]
        df['sentiment_neu'] = [s['neu'] for s in sentiments]
        df['sentiment_pos'] = [s['pos'] for s in sentiments]
        df['sentiment_compound'] = [s['compound'] for s in sentiments]
        
        # Add a binary sentiment label (positive/negative)
        df['sentiment_label'] = df['sentiment_compound'].apply(
            lambda x: 'positive' if x >= 0 else 'negative'
        )
        
        # Group by time window
        if time_window == 'hour':
            df['time_group'] = df['timestamp'].dt.floor('H')
        elif time_window == 'day':
            df['time_group'] = df['timestamp'].dt.floor('D')
        elif time_window == 'week':
            df['time_group'] = df['timestamp'].dt.floor('W')
        elif time_window == 'month':
            df['time_group'] = df['timestamp'].dt.floor('M')
        else:
            raise ValueError(f"Invalid time window: {time_window}")
        
        # Aggregate sentiment by time window
        agg_df = df.groupby('time_group').agg({
            'sentiment_compound': 'mean',
            'sentiment_pos': 'mean',
            'sentiment_neg': 'mean',
            'sentiment_neu': 'mean',
            'sentiment_label': lambda x: (x == 'positive').mean(),
            'text': 'count'
        }).reset_index()
        
        # Rename columns
        agg_df = agg_df.rename(columns={
            'sentiment_label': 'positive_ratio',
            'text': 'count'
        })
        
        return agg_df
    
    def plot_sentiment_trends(self, trend_df, save_path=None):
        """
        Plot sentiment trends over time.
        
        Args:
            trend_df (pd.DataFrame): DataFrame with sentiment trends
            save_path (str): Path to save the plot
            
        Returns:
            matplotlib.figure.Figure: The created figure
        """
        plt.figure(figsize=(12, 8))
        
        # Plot compound sentiment
        plt.subplot(2, 1, 1)
        plt.plot(trend_df['time_group'], trend_df['sentiment_compound'], 'b-', label='Compound')
        plt.axhline(y=0, color='r', linestyle='--')
        plt.fill_between(trend_df['time_group'], trend_df['sentiment_compound'], 0,
                        where=(trend_df['sentiment_compound'] >= 0),
                        color='green', alpha=0.3, interpolate=True)
        plt.fill_between(trend_df['time_group'], trend_df['sentiment_compound'], 0,
                        where=(trend_df['sentiment_compound'] < 0),
                        color='red', alpha=0.3, interpolate=True)
        plt.ylabel('Compound Sentiment')
        plt.title('Sentiment Trend Over Time')
        plt.legend()
        
        # Plot positive, negative, and neutral sentiment
        plt.subplot(2, 1, 2)
        plt.plot(trend_df['time_group'], trend_df['sentiment_pos'], 'g-', label='Positive')
        plt.plot(trend_df['time_group'], trend_df['sentiment_neg'], 'r-', label='Negative')
        plt.plot(trend_df['time_group'], trend_df['sentiment_neu'], 'b-', label='Neutral')
        plt.ylabel('Sentiment Score')
        plt.xlabel('Time')
        plt.legend()
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path)
            logger.info(f"Sentiment trend plot saved to {save_path}")
        
        return plt.gcf()
    
    def extract_frequent_words(self, texts, top_n=50, min_length=3):
        """
        Extract the most frequent words from a collection of texts.
        
        Args:
            texts (list): List of text strings
            top_n (int): Number of top words to extract
            min_length (int): Minimum word length to consider
            
        Returns:
            list: List of (word, frequency) tuples
        """
        # Combine all texts
        all_text = ' '.join([text for text in texts if isinstance(text, str)])
        
        # Tokenize
        words = word_tokenize(all_text.lower())
        
        # Filter words
        filtered_words = [
            word for word in words 
            if word.isalpha() and 
            word not in self.stop_words and 
            len(word) >= min_length
        ]
        
        # Count frequencies
        freq_dist = FreqDist(filtered_words)
        
        # Get top N words
        top_words = freq_dist.most_common(top_n)
        
        return top_words
    
    def generate_word_cloud(self, texts, save_path=None, max_words=100, width=800, height=400):
        """
        Generate a word cloud from texts.
        
        Args:
            texts (list): List of text strings
            save_path (str): Path to save the word cloud image
            max_words (int): Maximum number of words to include
            width (int): Width of the word cloud image
            height (int): Height of the word cloud image
            
        Returns:
            wordcloud.WordCloud: The generated word cloud
        """
        # Combine all texts
        all_text = ' '.join([text for text in texts if isinstance(text, str)])
        
        # Create word cloud
        wordcloud = WordCloud(
            width=width,
            height=height,
            max_words=max_words,
            background_color='white',
            stopwords=self.stop_words,
            min_word_length=3,
            collocations=False
        ).generate(all_text)
        
        # Plot the word cloud
        plt.figure(figsize=(10, 5))
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis('off')
        
        if save_path:
            plt.savefig(save_path)
            logger.info(f"Word cloud saved to {save_path}")
        
        return wordcloud
    
    def analyze_language_patterns(self, texts, labels, save_dir=None):
        """
        Analyze language patterns in texts, comparing different label groups.
        
        Args:
            texts (list): List of text strings
            labels (list): List of labels (0 for normal, 1 for distressed)
            save_dir (str): Directory to save analysis results
            
        Returns:
            dict: Dictionary of analysis results
        """
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
        
        # Create a DataFrame with texts and labels
        df = pd.DataFrame({
            'text': texts,
            'label': labels
        })
        
        # Separate texts by label
        normal_texts = df[df['label'] == 0]['text'].tolist()
        distressed_texts = df[df['label'] == 1]['text'].tolist()
        
        # Extract frequent words for each group
        normal_words = self.extract_frequent_words(normal_texts, top_n=50)
        distressed_words = self.extract_frequent_words(distressed_texts, top_n=50)
        
        # Generate word clouds
        if save_dir:
            self.generate_word_cloud(
                normal_texts,
                save_path=os.path.join(save_dir, 'normal_wordcloud.png')
            )
            
            self.generate_word_cloud(
                distressed_texts,
                save_path=os.path.join(save_dir, 'distressed_wordcloud.png')
            )
        
        # Analyze sentiment for each group
        normal_sentiment = self.batch_analyze_sentiment(normal_texts)
        distressed_sentiment = self.batch_analyze_sentiment(distressed_texts)
        
        # Calculate average sentiment
        avg_normal_sentiment = {
            'neg': np.mean([s['neg'] for s in normal_sentiment]),
            'neu': np.mean([s['neu'] for s in normal_sentiment]),
            'pos': np.mean([s['pos'] for s in normal_sentiment]),
            'compound': np.mean([s['compound'] for s in normal_sentiment])
        }
        
        avg_distressed_sentiment = {
            'neg': np.mean([s['neg'] for s in distressed_sentiment]),
            'neu': np.mean([s['neu'] for s in distressed_sentiment]),
            'pos': np.mean([s['pos'] for s in distressed_sentiment]),
            'compound': np.mean([s['compound'] for s in distressed_sentiment])
        }
        
        # Plot sentiment comparison
        if save_dir:
            self.plot_sentiment_comparison(
                avg_normal_sentiment,
                avg_distressed_sentiment,
                save_path=os.path.join(save_dir, 'sentiment_comparison.png')
            )
        
        # Identify distinctive words (words that appear more frequently in one group)
        normal_word_dict = dict(normal_words)
        distressed_word_dict = dict(distressed_words)
        
        all_words = set(normal_word_dict.keys()) | set(distressed_word_dict.keys())
        
        distinctive_words = []
        for word in all_words:
            normal_freq = normal_word_dict.get(word, 0)
            distressed_freq = distressed_word_dict.get(word, 0)
            
            # Calculate ratio (with smoothing to avoid division by zero)
            ratio = (distressed_freq + 1) / (normal_freq + 1)
            
            distinctive_words.append({
                'word': word,
                'normal_freq': normal_freq,
                'distressed_freq': distressed_freq,
                'ratio': ratio
            })
        
        # Sort by ratio (descending)
        distinctive_words.sort(key=lambda x: x['ratio'], reverse=True)
        
        # Extract top distinctive words for each group
        top_distressed_words = [w for w in distinctive_words if w['ratio'] > 2][:20]
        
        # Sort by inverse ratio (ascending)
        distinctive_words.sort(key=lambda x: x['ratio'])
        
        # Extract top distinctive words for normal group
        top_normal_words = [w for w in distinctive_words if w['ratio'] < 0.5][:20]
        
        # Plot distinctive words
        if save_dir:
            self.plot_distinctive_words(
                top_normal_words,
                top_distressed_words,
                save_path=os.path.join(save_dir, 'distinctive_words.png')
            )
        
        # Perform topic modeling
        topics = self.extract_topics(texts, n_topics=5)
        
        # Save topics
        if save_dir:
            with open(os.path.join(save_dir, 'topics.json'), 'w') as f:
                json.dump(topics, f, indent=2)
        
        # Return analysis results
        return {
            'normal_words': normal_words,
            'distressed_words': distressed_words,
            'avg_normal_sentiment': avg_normal_sentiment,
            'avg_distressed_sentiment': avg_distressed_sentiment,
            'top_normal_words': top_normal_words,
            'top_distressed_words': top_distressed_words,
            'topics': topics
        }
    
    def plot_sentiment_comparison(self, normal_sentiment, distressed_sentiment, save_path=None):
        """
        Plot a comparison of sentiment between normal and distressed texts.
        
        Args:
            normal_sentiment (dict): Sentiment scores for normal texts
            distressed_sentiment (dict): Sentiment scores for distressed texts
            save_path (str): Path to save the plot
        """
        categories = ['Negative', 'Neutral', 'Positive', 'Compound']
        normal_values = [
            normal_sentiment['neg'],
            normal_sentiment['neu'],
            normal_sentiment['pos'],
            normal_sentiment['compound']
        ]
        distressed_values = [
            distressed_sentiment['neg'],
            distressed_sentiment['neu'],
            distressed_sentiment['pos'],
            distressed_sentiment['compound']
        ]
        
        x = np.arange(len(categories))
        width = 0.35
        
        fig, ax = plt.subplots(figsize=(10, 6))
        rects1 = ax.bar(x - width/2, normal_values, width, label='Normal')
        rects2 = ax.bar(x + width/2, distressed_values, width, label='Distressed')
        
        ax.set_ylabel('Score')
        ax.set_title('Sentiment Comparison: Normal vs. Distressed')
        ax.set_xticks(x)
        ax.set_xticklabels(categories)
        ax.legend()
        
        # Add value labels
        def autolabel(rects):
            for rect in rects:
                height = rect.get_height()
                ax.annotate(f'{height:.2f}',
                            xy=(rect.get_x() + rect.get_width() / 2, height),
                            xytext=(0, 3),
                            textcoords="offset points",
                            ha='center', va='bottom')
        
        autolabel(rects1)
        autolabel(rects2)
        
        fig.tight_layout()
        
        if save_path:
            plt.savefig(save_path)
            logger.info(f"Sentiment comparison plot saved to {save_path}")
        
        plt.close()
    
    def plot_distinctive_words(self, normal_words, distressed_words, save_path=None):
        """
        Plot distinctive words for normal and distressed texts.
        
        Args:
            normal_words (list): List of distinctive words for normal texts
            distressed_words (list): List of distinctive words for distressed texts
            save_path (str): Path to save the plot
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 8))
        
        # Plot distinctive words for normal texts
        normal_words = normal_words[:10]  # Limit to top 10
        normal_words_df = pd.DataFrame(normal_words)
        ax1.barh(normal_words_df['word'], 1 / normal_words_df['ratio'])
        ax1.set_title('Words More Common in Normal Texts')
        ax1.set_xlabel('Relative Frequency (Normal / Distressed)')
        
        # Plot distinctive words for distressed texts
        distressed_words = distressed_words[:10]  # Limit to top 10
        distressed_words_df = pd.DataFrame(distressed_words)
        ax2.barh(distressed_words_df['word'], distressed_words_df['ratio'])
        ax2.set_title('Words More Common in Distressed Texts')
        ax2.set_xlabel('Relative Frequency (Distressed / Normal)')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path)
            logger.info(f"Distinctive words plot saved to {save_path}")
        
        plt.close()
    
    def extract_topics(self, texts, n_topics=5, n_top_words=10):
        """
        Extract topics from texts using Latent Dirichlet Allocation.
        
        Args:
            texts (list): List of text strings
            n_topics (int): Number of topics to extract
            n_top_words (int): Number of top words per topic
            
        Returns:
            list: List of topics, each with top words
        """
        # Create a document-term matrix
        vectorizer = CountVectorizer(
            max_features=1000,
            min_df=2,
            max_df=0.85,
            stop_words='english'
        )
        
        # Filter out non-string texts
        valid_texts = [text for text in texts if isinstance(text, str)]
        
        try:
            dtm = vectorizer.fit_transform(valid_texts)
            
            # Create and fit the LDA model
            lda = LatentDirichletAllocation(
                n_components=n_topics,
                random_state=42,
                max_iter=10
            )
            
            lda.fit(dtm)
            
            # Get feature names
            feature_names = vectorizer.get_feature_names_out()
            
            # Extract topics
            topics = []
            for topic_idx, topic in enumerate(lda.components_):
                top_words_idx = topic.argsort()[:-n_top_words-1:-1]
                top_words = [feature_names[i] for i in top_words_idx]
                
                topics.append({
                    'id': topic_idx,
                    'words': top_words
                })
            
            return topics
        except Exception as e:
            logger.error(f"Error extracting topics: {str(e)}")
            return []
    
    def analyze_engagement_patterns(self, texts, timestamps, engagement_metrics=None):
        """
        Analyze patterns in user engagement over time.
        
        Args:
            texts (list): List of text strings
            timestamps (list): List of timestamp strings or datetime objects
            engagement_metrics (list): Optional list of engagement metrics (e.g., likes, shares)
            
        Returns:
            pd.DataFrame: DataFrame with engagement patterns
        """
        # Convert timestamps to datetime if they are strings
        if timestamps and isinstance(timestamps[0], str):
            timestamps = [pd.to_datetime(ts) for ts in timestamps]
        
        # Create a DataFrame
        df = pd.DataFrame({
            'text': texts,
            'timestamp': timestamps
        })
        
        # Add engagement metrics if provided
        if engagement_metrics:
            df['engagement'] = engagement_metrics
        
        # Add text length as a basic engagement metric
        df['text_length'] = df['text'].apply(lambda x: len(x) if isinstance(x, str) else 0)
        
        # Group by day
        df['date'] = df['timestamp'].dt.date
        
        # Aggregate by day
        agg_df = df.groupby('date').agg({
            'text': 'count',
            'text_length': 'mean'
        }).reset_index()
        
        # Rename columns
        agg_df = agg_df.rename(columns={
            'text': 'post_count',
            'text_length': 'avg_text_length'
        })
        
        # Add engagement metrics if provided
        if engagement_metrics:
            engagement_agg = df.groupby('date')['engagement'].mean().reset_index()
            agg_df = pd.merge(agg_df, engagement_agg, on='date')
        
        return agg_df
    
    def plot_engagement_trends(self, engagement_df, save_path=None):
        """
        Plot trends in user engagement over time.
        
        Args:
            engagement_df (pd.DataFrame): DataFrame with engagement patterns
            save_path (str): Path to save the plot
        """
        plt.figure(figsize=(12, 8))
        
        # Plot post count
        plt.subplot(2, 1, 1)
        plt.plot(engagement_df['date'], engagement_df['post_count'], 'b-')
        plt.ylabel('Post Count')
        plt.title('Engagement Trends Over Time')
        
        # Plot average text length
        plt.subplot(2, 1, 2)
        plt.plot(engagement_df['date'], engagement_df['avg_text_length'], 'g-')
        plt.ylabel('Average Text Length')
        plt.xlabel('Date')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path)
            logger.info(f"Engagement trends plot saved to {save_path}")
        
        plt.close()
    
    def run_full_analysis(self, texts, timestamps=None, labels=None, engagement_metrics=None):
        """
        Run a full sentiment and trend analysis.
        
        Args:
            texts (list): List of text strings
            timestamps (list): Optional list of timestamp strings or datetime objects
            labels (list): Optional list of labels (0 for normal, 1 for distressed)
            engagement_metrics (list): Optional list of engagement metrics
            
        Returns:
            dict: Dictionary of analysis results
        """
        # Create timestamp if not provided (using current time)
        if not timestamps:
            now = datetime.now()
            timestamps = [now - timedelta(days=i) for i in range(len(texts))]
        
        # Create output directory for this analysis
        timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        analysis_dir = os.path.join(self.output_dir, f"analysis_{timestamp_str}")
        os.makedirs(analysis_dir, exist_ok=True)
        
        # Analyze sentiment
        sentiment_scores = self.batch_analyze_sentiment(texts)
        
        # Generate word cloud
        wordcloud = self.generate_word_cloud(
            texts,
            save_path=os.path.join(analysis_dir, "wordcloud.png")
        )
        
        # Extract frequent words
        frequent_words = self.extract_frequent_words(texts, top_n=50)
        
        # Save frequent words
        with open(os.path.join(analysis_dir, "frequent_words.json"), "w") as f:
            json.dump(frequent_words, f, indent=2)
        
        # Extract topics
        topics = self.extract_topics(texts, n_topics=5)
        
        # Save topics
        with open(os.path.join(analysis_dir, "topics.json"), "w") as f:
            json.dump(topics, f, indent=2)
        
        # Analyze sentiment trends if timestamps are provided
        if timestamps:
            sentiment_trends = self.analyze_sentiment_trends(texts, timestamps)
            
            # Save sentiment trends
            sentiment_trends.to_csv(os.path.join(analysis_dir, "sentiment_trends.csv"), index=False)
            
            # Plot sentiment trends
            self.plot_sentiment_trends(
                sentiment_trends,
                save_path=os.path.join(analysis_dir, "sentiment_trends.png")
            )
        else:
            sentiment_trends = None
        
        # Analyze language patterns if labels are provided
        if labels:
            language_patterns = self.analyze_language_patterns(
                texts,
                labels,
                save_dir=os.path.join(analysis_dir, "language_patterns")
            )
        else:
            language_patterns = None
        
        # Analyze engagement patterns if timestamps are provided
        if timestamps:
            engagement_patterns = self.analyze_engagement_patterns(
                texts,
                timestamps,
                engagement_metrics
            )
            
            # Save engagement patterns
            engagement_patterns.to_csv(os.path.join(analysis_dir, "engagement_patterns.csv"), index=False)
            
            # Plot engagement trends
            self.plot_engagement_trends(
                engagement_patterns,
                save_path=os.path.join(analysis_dir, "engagement_trends.png")
            )
        else:
            engagement_patterns = None
        
        # Return analysis results
        return {
            'sentiment_scores': sentiment_scores,
            'frequent_words': frequent_words,
            'topics': topics,
            'sentiment_trends': sentiment_trends,
            'language_patterns': language_patterns,
            'engagement_patterns': engagement_patterns,
            'analysis_dir': analysis_dir
        }


# Example usage
if __name__ == "__main__":
    # Sample data
    data = pd.DataFrame({
        "text": [
            "I feel sad and empty inside",
            "I'm so depressed I can barely get out of bed",
            "Nothing brings me joy anymore",
            "I feel worthless and hopeless about the future",
            "I can't stop crying and I don't know why",
            "Life is good, I'm enjoying my day",
            "I had a productive meeting at work today",
            "Feeling great after my workout",
            "I'm excited about my upcoming vacation",
            "Just finished a good book and feeling satisfied"
        ],
        "label": [1, 1, 1, 1, 1, 0, 0, 0, 0, 0],
        "timestamp": [
            datetime.now() - timedelta(days=9),
            datetime.now() - timedelta(days=8),
            datetime.now() - timedelta(days=7),
            datetime.now() - timedelta(days=6),
            datetime.now() - timedelta(days=5),
            datetime.now() - timedelta(days=4),
            datetime.now() - timedelta(days=3),
            datetime.now() - timedelta(days=2),
            datetime.now() - timedelta(days=1),
            datetime.now()
        ]
    })
    
    # Initialize sentiment analyzer
    analyzer = SentimentAnalyzer()
    
    # Run full analysis
    results = analyzer.run_full_analysis(
        data["text"].tolist(),
        data["timestamp"].tolist(),
        data["label"].tolist()
    )
    
    print(f"Analysis results saved to {results['analysis_dir']}")