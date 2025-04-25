"""
Data preparation module for the Mental Health Predictor.

This module downloads, processes, and prepares datasets for training the mental health prediction model.
It combines multiple datasets to create a comprehensive training dataset.
"""

import os
import pandas as pd
import numpy as np
from datasets import load_dataset
from sklearn.model_selection import train_test_split
import logging
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Download NLTK resources
try:
    nltk.download('punkt', quiet=True)
    nltk.download('stopwords', quiet=True)
except Exception as e:
    logger.warning(f"Failed to download NLTK resources: {e}")

# Create data directory if it doesn't exist
os.makedirs("app/data", exist_ok=True)

def clean_text(text):
    """
    Clean and preprocess text data.
    
    Args:
        text (str): The input text to clean
        
    Returns:
        str: Cleaned text
    """
    if not isinstance(text, str):
        return ""
    
    # Convert to lowercase
    text = text.lower()
    
    # Remove URLs
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    
    # Remove HTML tags
    text = re.sub(r'<.*?>', '', text)
    
    # Remove special characters and numbers
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\d+', '', text)
    
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text

def load_and_prepare_datasets():
    """
    Load and prepare multiple datasets for mental health prediction.
    
    Returns:
        tuple: (train_df, test_df) - Processed training and testing dataframes
    """
    logger.info("Loading and preparing datasets...")
    
    # Dataset 1: Load emotion dataset from Hugging Face
    logger.info("Loading emotion dataset...")
    try:
        emotion_dataset = load_dataset("emotion")
        
        # Convert to pandas DataFrame
        train_emotion = pd.DataFrame(emotion_dataset["train"])
        test_emotion = pd.DataFrame(emotion_dataset["test"])
        
        # Map emotions to binary labels (0: normal, 1: distressed)
        # Sadness (0), Fear (1), Anger (3) -> Distressed (1)
        # Joy (2), Love (4), Surprise (5) -> Normal (0)
        emotion_mapping = {0: 1, 1: 1, 2: 0, 3: 1, 4: 0, 5: 0}
        
        train_emotion['label'] = train_emotion['label'].map(emotion_mapping)
        test_emotion['label'] = test_emotion['label'].map(emotion_mapping)
        
        # Rename columns to match our schema
        train_emotion = train_emotion.rename(columns={'text': 'text_input'})
        test_emotion = test_emotion.rename(columns={'text': 'text_input'})
        
        logger.info(f"Emotion dataset loaded: {len(train_emotion)} training samples, {len(test_emotion)} test samples")
    except Exception as e:
        logger.error(f"Error loading emotion dataset: {e}")
        train_emotion = pd.DataFrame(columns=['text_input', 'label'])
        test_emotion = pd.DataFrame(columns=['text_input', 'label'])
    
    # Dataset 2: Load tweet emotion dataset
    logger.info("Loading tweet emotion dataset...")
    try:
        tweet_dataset = load_dataset("dair-ai/emotion")
        
        # Convert to pandas DataFrame
        tweet_df = pd.DataFrame(tweet_dataset["train"])
        
        # Map emotions to binary labels (0: normal, 1: distressed)
        # sadness, fear, anger -> Distressed (1)
        # joy, love, surprise -> Normal (0)
        tweet_mapping = {
            'sadness': 1, 'fear': 1, 'anger': 1,
            'joy': 0, 'love': 0, 'surprise': 0
        }
        
        tweet_df['label'] = tweet_df['label'].map(tweet_mapping)
        
        # Rename columns to match our schema
        tweet_df = tweet_df.rename(columns={'text': 'text_input'})
        
        # Split into train and test
        tweet_train, tweet_test = train_test_split(
            tweet_df, test_size=0.2, random_state=42, stratify=tweet_df["label"]
        )
        
        logger.info(f"Tweet emotion dataset loaded: {len(tweet_train)} training samples, {len(tweet_test)} test samples")
    except Exception as e:
        logger.error(f"Error loading tweet emotion dataset: {e}")
        tweet_train = pd.DataFrame(columns=['text_input', 'label'])
        tweet_test = pd.DataFrame(columns=['text_input', 'label'])
    
    # Dataset 3: Create a custom mental health dataset
    logger.info("Creating custom mental health dataset...")
    
    # Distressed examples (label 1)
    distressed_texts = [
        "I feel sad and empty inside all the time",
        "I'm so depressed I can barely get out of bed most days",
        "Nothing brings me joy anymore, everything feels pointless",
        "I feel worthless and hopeless about the future",
        "I can't stop crying and I don't know why",
        "I'm having thoughts about ending it all",
        "I feel like a burden to everyone around me",
        "I'm constantly anxious and can't relax for even a moment",
        "I haven't slept well in weeks and I'm exhausted",
        "I've lost interest in activities I used to enjoy",
        "I feel overwhelmed by simple daily tasks",
        "My mind is filled with negative thoughts I can't control",
        "I feel like I'm drowning in my own thoughts",
        "Everything feels like too much effort lately",
        "I'm constantly tired no matter how much I sleep",
        "I don't see any point in continuing like this",
        "I feel like nobody understands what I'm going through",
        "I'm struggling to find any reason to keep going",
        "My anxiety is making it hard to function normally",
        "I feel trapped in my own mind with no way out",
        "I can't concentrate on anything because of my anxiety",
        "I feel like I'm falling apart and can't hold myself together",
        "I'm having panic attacks almost every day now",
        "I feel completely alone even when I'm with other people",
        "I'm afraid I'll never feel happy again",
        "I've been isolating myself from friends and family",
        "I feel like a failure in every aspect of my life",
        "I'm constantly on edge and irritable",
        "I feel like I'm just going through the motions of living",
        "I can't see a future for myself anymore",
        "I feel like I'm watching my life from outside my body",
        "I'm exhausted from pretending to be okay",
        "I feel numb and disconnected from everything",
        "I'm having trouble making even simple decisions",
        "I feel like I'm a disappointment to everyone",
        "I'm consumed by feelings of guilt and shame",
        "I feel like I'm broken beyond repair",
        "I can't remember the last time I felt truly happy",
        "I'm afraid of being a burden so I keep everything inside",
        "I feel like I'm screaming inside but no one can hear me"
    ]
    
    # Normal examples (label 0)
    normal_texts = [
        "Life is good, I'm enjoying my day",
        "I had a productive meeting at work today",
        "Feeling great after my workout this morning",
        "I'm excited about my upcoming vacation next month",
        "Just finished a good book and feeling satisfied",
        "Had a nice dinner with friends tonight",
        "The weather is beautiful today, perfect for a walk",
        "I accomplished all my tasks for the day",
        "Looking forward to the weekend plans with family",
        "I learned something new today and it was interesting",
        "Feeling motivated to start this new project",
        "Had a good conversation with my family yesterday",
        "I'm proud of what I achieved today at work",
        "Taking time to relax and recharge this evening",
        "Feeling content with where I am in life right now",
        "I'm grateful for the support of my friends",
        "Today was challenging but I handled it well",
        "I'm making progress on my personal goals",
        "I enjoyed spending time in nature today",
        "I'm feeling optimistic about the future",
        "Had a great time at the concert last night",
        "Feeling refreshed after a good night's sleep",
        "I'm enjoying learning this new skill",
        "Spent quality time with my pet today",
        "I'm satisfied with the work I completed this week",
        "Had a good laugh with colleagues during lunch",
        "I'm looking forward to trying that new restaurant",
        "Feeling balanced and centered after meditation",
        "I'm appreciating the little things in life today",
        "Had a productive brainstorming session this afternoon",
        "I feel energized and ready for new challenges",
        "Enjoying this beautiful sunset from my window",
        "I'm happy with the progress I'm making in therapy",
        "Had a nice video call with an old friend",
        "I'm feeling creative and inspired today",
        "Enjoying my new hobby that I started last month",
        "I feel connected to my community after volunteering",
        "Had a peaceful morning with my coffee and book",
        "I'm excited about the positive changes in my life",
        "Feeling thankful for my health and wellbeing"
    ]
    
    # Create DataFrame
    custom_df = pd.DataFrame({
        "text_input": distressed_texts + normal_texts,
        "label": [1] * len(distressed_texts) + [0] * len(normal_texts)
    })
    
    # Split into train and test
    custom_train, custom_test = train_test_split(
        custom_df, test_size=0.2, random_state=42, stratify=custom_df["label"]
    )
    
    logger.info(f"Custom dataset created: {len(custom_train)} training samples, {len(custom_test)} test samples")
    
    # Combine all datasets
    train_df = pd.concat([train_emotion, tweet_train, custom_train], ignore_index=True)
    test_df = pd.concat([test_emotion, tweet_test, custom_test], ignore_index=True)
    
    # Clean text
    logger.info("Cleaning text data...")
    train_df['text_input'] = train_df['text_input'].apply(clean_text)
    test_df['text_input'] = test_df['text_input'].apply(clean_text)
    
    # Remove empty texts
    train_df = train_df[train_df['text_input'].str.len() > 0]
    test_df = test_df[test_df['text_input'].str.len() > 0]
    
    # Balance the dataset if needed
    if train_df['label'].mean() < 0.3 or train_df['label'].mean() > 0.7:
        logger.info("Balancing the training dataset...")
        # Separate by class
        train_normal = train_df[train_df['label'] == 0]
        train_distressed = train_df[train_df['label'] == 1]
        
        # Determine the target size (the smaller class size)
        target_size = min(len(train_normal), len(train_distressed))
        
        # Downsample the larger class
        if len(train_normal) > len(train_distressed):
            train_normal = train_normal.sample(target_size, random_state=42)
        else:
            train_distressed = train_distressed.sample(target_size, random_state=42)
        
        # Combine the balanced classes
        train_df = pd.concat([train_normal, train_distressed], ignore_index=True)
        
        # Shuffle the data
        train_df = train_df.sample(frac=1, random_state=42).reset_index(drop=True)
    
    # Save the processed datasets
    train_df.to_csv("app/data/train_data.csv", index=False)
    test_df.to_csv("app/data/test_data.csv", index=False)
    
    logger.info(f"Final dataset sizes: {len(train_df)} training samples, {len(test_df)} test samples")
    logger.info(f"Class distribution in training: {train_df['label'].value_counts(normalize=True)}")
    logger.info(f"Class distribution in testing: {test_df['label'].value_counts(normalize=True)}")
    
    return train_df, test_df

if __name__ == "__main__":
    # Load and prepare datasets
    train_df, test_df = load_and_prepare_datasets()
    
    print("\nDataset Preparation Complete!")
    print(f"Training samples: {len(train_df)}")
    print(f"Testing samples: {len(test_df)}")
    print(f"Training class distribution: {train_df['label'].value_counts()}")
    print(f"Testing class distribution: {test_df['label'].value_counts()}")