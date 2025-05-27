import tweepy
import praw
import pandas as pd
import json
import time
from datetime import datetime
import logging
from typing import List, Dict, Optional

class SocialMediaCollector:
    def __init__(self, config_path: str):
        """Initialize social media data collector with API credentials"""
        with open(config_path, 'r') as f:
            self.config = json.load(f)
        
        self.setup_logging()
        self.setup_apis()
    
    def setup_logging(self):
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('data_collection.log'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
    
    def setup_apis(self):
        """Setup API connections for different platforms"""
        # Twitter API v2 setup
        self.twitter_client = tweepy.Client(
            bearer_token=self.config['twitter']['bearer_token'],
            consumer_key=self.config['twitter']['api_key'],
            consumer_secret=self.config['twitter']['api_secret'],
            access_token=self.config['twitter']['access_token'],
            access_token_secret=self.config['twitter']['access_token_secret'],
            wait_on_rate_limit=True
        )
        
        # Reddit API setup
        self.reddit = praw.Reddit(
            client_id=self.config['reddit']['client_id'],
            client_secret=self.config['reddit']['client_secret'],
            user_agent=self.config['reddit']['user_agent']
        )
    
    def collect_twitter_data(self, keywords: List[str], max_results: int = 1000) -> pd.DataFrame:
        """Collect Twitter data based on keywords"""
        tweets_data = []
        
        try:
            for keyword in keywords:
                tweets = tweepy.Paginator(
                    self.twitter_client.search_recent_tweets,
                    query=keyword,
                    tweet_fields=['created_at', 'author_id', 'public_metrics', 'context_annotations'],
                    max_results=min(100, max_results)
                ).flatten(limit=max_results)
                
                for tweet in tweets:
                    tweets_data.append({
                        'id': tweet.id,
                        'text': tweet.text,
                        'created_at': tweet.created_at,
                        'author_id': tweet.author_id,
                        'retweet_count': tweet.public_metrics['retweet_count'],
                        'like_count': tweet.public_metrics['like_count'],
                        'platform': 'twitter',
                        'keyword': keyword
                    })
                
                time.sleep(1)  # Rate limiting
                
        except Exception as e:
            self.logger.error(f"Error collecting Twitter data: {e}")
        
        return pd.DataFrame(tweets_data)
    
    def collect_reddit_data(self, subreddits: List[str], max_posts: int = 1000) -> pd.DataFrame:
        """Collect Reddit data from specified subreddits"""
        reddit_data = []
        
        try:
            for subreddit_name in subreddits:
                subreddit = self.reddit.subreddit(subreddit_name)
                
                for post in subreddit.hot(limit=max_posts):
                    reddit_data.append({
                        'id': post.id,
                        'title': post.title,
                        'text': post.selftext,
                        'created_at': datetime.fromtimestamp(post.created_utc),
                        'author': str(post.author),
                        'score': post.score,
                        'num_comments': post.num_comments,
                        'platform': 'reddit',
                        'subreddit': subreddit_name
                    })
                
                time.sleep(1)  # Rate limiting
                
        except Exception as e:
            self.logger.error(f"Error collecting Reddit data: {e}")
        
        return pd.DataFrame(reddit_data)
    
    def save_data(self, data: pd.DataFrame, filename: str):
        """Save collected data to file"""
        data.to_csv(f"data/raw/{filename}", index=False)
        self.logger.info(f"Data saved to data/raw/{filename}")