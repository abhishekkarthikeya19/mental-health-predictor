# Mental Health Predictor Datasets

This document describes the datasets used for training the Mental Health Predictor model and provides instructions for running the training pipeline.

## Dataset Overview

The Mental Health Predictor uses a combination of datasets to train a model that can identify signs of emotional distress in text:

1. **Emotion Dataset**: A dataset from Hugging Face containing text labeled with emotions (sadness, fear, anger, joy, love, surprise). We map these emotions to binary labels (distressed/normal).

2. **Tweet Emotion Dataset**: A dataset of tweets labeled with emotions, which we also map to binary labels.

3. **Custom Mental Health Dataset**: A manually curated dataset with examples of distressed and normal text specifically designed for mental health analysis.

## Data Preparation

The data preparation process includes:

1. Loading datasets from multiple sources
2. Cleaning and preprocessing text (removing URLs, HTML tags, special characters)
3. Mapping emotion labels to binary labels (distressed/normal)
4. Balancing the dataset to ensure equal representation of classes
5. Splitting into training and testing sets

## Training Pipeline

The training pipeline consists of the following steps:

1. **Data Preparation**: Processes and prepares the datasets
2. **Model Training**: Trains a transformer-based model (DistilBERT) on the prepared data
3. **Model Evaluation**: Evaluates the model on the test set
4. **Model Saving**: Saves the trained model and related artifacts

## Running the Training Pipeline

To run the complete training pipeline:

```bash
python app/run_training.py
```

This will:
1. Prepare the datasets
2. Train the model
3. Evaluate the model
4. Save the model and related artifacts

## Testing the Model

To test the trained model with example inputs:

```bash
python app/test_model.py
```

This will:
1. Load the trained model
2. Display model information and performance metrics
3. Test the model with example inputs
4. Save the test results

## Model Files

After training, the following files will be created:

- `app/model/mental_health_model.pkl`: The trained model wrapper
- `app/model/transformer_model/`: Directory containing the transformer model and tokenizer
- `app/model/model_info.json`: Information about the model and training process
- `app/model/evaluation_results.csv`: Detailed evaluation results on the test set
- `app/model/misclassified_examples.csv`: Examples that were misclassified by the model

## Dataset Statistics

After training, you can find detailed statistics about the datasets in:

- `app/data/train_data.csv`: The processed training dataset
- `app/data/test_data.csv`: The processed test dataset
- `app/training_log.txt`: Log file containing information about the training process

## Customizing the Training

You can customize the training process by modifying the following files:

- `app/data_preparation.py`: Modify to use different datasets or preprocessing steps
- `app/train_model.py`: Modify to change model architecture or training parameters

## Requirements

The training pipeline requires the following dependencies:

- transformers
- torch
- datasets
- pandas
- scikit-learn
- nltk
- joblib

These dependencies are listed in the main `requirements.txt` file.