# RNN-based Question Answering System

## Overview
This project implements a simple Question Answering system using Recurrent Neural Networks (RNN) in PyTorch. It demonstrates how to build, train, and use a basic neural network model that can respond to user queries by learning patterns from a dataset of question-answer pairs.

## Features
- Custom tokenization and vocabulary building pipeline
- Text-to-numerical conversion for NLP processing
- PyTorch Dataset and DataLoader implementation for efficient batch processing
- Simple RNN architecture with embedding layer
- Training loop with Adam optimizer
- Prediction functionality with confidence thresholding

## Dataset
The system is trained on a dataset of 100 unique question-answer pairs covering various topics from general knowledge, science, geography, and more. The dataset provides a foundation for the model to learn patterns between questions and appropriate answers.

## Model Architecture
- **Embedding Layer**: Converts token indices to dense vectors of fixed size (dimension: 50)
- **RNN Layer**: Processes the sequence of embeddings (hidden size: 64)
- **Fully Connected Layer**: Maps RNN output to vocabulary size for classification

## Setup and Usage
1. **Install Dependencies**:
     pip install torch pandas


 
3. **Prepare Dataset**:
- Use the provided `100_Unique_QA_Dataset.csv` or create your own dataset with similar format
- The dataset should have 'question' and 'answer' columns

3. **Training**:
- Run the script to train the model on your dataset
- Default parameters: learning rate = 0.001, epochs = 20

4. **Prediction**:
- Use the `predict()` function to ask questions to the trained model
- Example: `predict(model, "What is the largest planet in our solar system?")`

## Future Improvements
- Implement more sophisticated tokenization techniques
- Explore more complex architectures (LSTM, GRU, Transformer)
- Expand the training dataset for better generalization
- Add support for longer and more complex questions
- Implement attention mechanisms for improved performance

## About the Author
I'm a B.Tech student in Artificial Intelligence & Data Science at IIITDM with experience in Machine Learning, Data Science, and Web Development. This project demonstrates my skills in implementing neural network architectures for NLP tasks.
