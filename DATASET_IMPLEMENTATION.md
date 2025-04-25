# Mental Health Predictor - Dataset Implementation

This document provides an overview of the dataset implementation for the Mental Health Predictor project.

## Overview

We've implemented a comprehensive dataset solution for training the mental health prediction model. The implementation includes:

1. **Data Collection and Preparation**: Scripts to download, process, and prepare datasets from multiple sources
2. **Model Training**: Enhanced training pipeline with improved parameters and evaluation metrics
3. **Frontend Integration**: Components to display dataset information and model performance
4. **Testing and Evaluation**: Tools to test the model and evaluate its performance

## Implementation Details

### 1. Data Collection and Preparation

The `data_preparation.py` script:
- Downloads datasets from Hugging Face (emotion dataset, tweet emotion dataset)
- Combines them with a custom mental health dataset
- Cleans and preprocesses the text data
- Balances the dataset to ensure equal representation of classes
- Splits the data into training and testing sets

### 2. Model Training

The enhanced `train_model.py` script:
- Uses the prepared datasets to train a transformer-based model (DistilBERT)
- Implements improved training parameters for better performance
- Calculates detailed evaluation metrics (accuracy, precision, recall, F1 score)
- Saves the model and related artifacts for later use

### 3. Frontend Integration

We've added:
- `DatasetInfo.js`: A component to display dataset information and model performance
- `modelDemo.js`: Utility functions to demonstrate the model in the frontend
- CSS styles for the new components

### 4. Testing and Evaluation

The `test_model.py` script:
- Tests the trained model with example inputs
- Displays model information and performance metrics
- Saves test results for analysis

## Files Created/Modified

### New Files:
- `app/data_preparation.py`: Script to prepare datasets
- `app/run_training.py`: Script to run the training pipeline
- `app/test_model.py`: Script to test the model
- `app/README_DATASETS.md`: Documentation for the datasets
- `frontend/src/utils/modelDemo.js`: Frontend utility for model demonstration
- `frontend/src/components/DatasetInfo.js`: Component to display dataset information
- `run_implementation.py`: Script to run the complete implementation

### Modified Files:
- `app/train_model.py`: Enhanced with improved training parameters and evaluation metrics
- `frontend/src/styles.css`: Added styles for the new components
- `requirements.txt`: Added new dependencies

## How to Use

### Training the Model

To prepare the datasets and train the model:

```bash
python app/run_training.py
```

### Testing the Model

To test the trained model:

```bash
python app/test_model.py
```

### Running the Complete Implementation

To run the complete implementation:

```bash
python run_implementation.py
```

Options:
- `--skip-training`: Skip dataset preparation and model training
- `--backend-only`: Start only the backend API server
- `--frontend-only`: Start only the frontend development server

## Dataset Sources

1. **Emotion Dataset**: A dataset from Hugging Face containing text labeled with emotions
2. **Tweet Emotion Dataset**: A dataset of tweets labeled with emotions
3. **Custom Mental Health Dataset**: A manually curated dataset with examples of distressed and normal text

## Model Performance

The model achieves:
- High accuracy in distinguishing between normal and distressed text
- Good precision and recall for both classes
- Robust performance across different types of input text

## Future Improvements

Potential improvements for the dataset implementation:
- Incorporate more domain-specific mental health datasets
- Implement data augmentation techniques to increase dataset size
- Add support for multi-language datasets
- Implement active learning to continuously improve the model