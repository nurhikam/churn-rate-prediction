# Churn Rate Prediction using ANN

## Overview

This project aims to predict whether a bank's customers will continue using the bank's services or leave (churn) using Artificial Neural Networks (ANN).

The model is trained on a dataset containing information about 10,000 customers including:

- Demographic info (age, gender, geography, etc.)
- Financial info (credit score, balance, salary, etc.)
- Engagement info (tenure, number of products used, activity status, etc.)

The target variable is a binary variable indicating whether the customer exited (churned) or not.

## The project is structured as follows:
1. Exploratory Data Analysis
2. Data Preprocessing
3. Model Creation and Evaluation

## Requirements

- Python 3.x
- NumPy
- Pandas
- Matplotlib
- Seaborn
- Scikit-learn
- TensorFlow

## Data

The original dataset can be found at [Churn Modelling Dataset](https://www.kaggle.com/datasets/shrutimechlearn/churn-modelling) on Kaggle. 

The dataset contains 10,000 rows and 14 columns.

Overview Table:
![image](https://github.com/nurhikam/churn-rate-prediction/assets/92198564/e3e060f1-5679-4c39-b146-f825ac9bb704)

## Model

An ANN model with the following structure is built and trained:

- 2 hidden layers with 6 nodes each
- Input layer with 11 input features
- Output layer with sigmoid activation and binary classification

The model is compiled with adam optimizer and binary crossentropy loss.

## Usage

The Jupyter notebook `churn_prediction.ipynb` contains the full code to load data, preprocess, build model, train, evaluate and make predictions. 

To use:

1. Clone this repo
2. Install requirements
3. Run the notebook

## Performance

The model achieves an accuracy of 85.8% on the test set.

## Future Work

Some ways to improve the model performance:

- Try different model architectures (more layers, nodes, etc)  
- Experiment with different activation functions and optimizers
- Use regularization techniques like dropout to reduce overfitting
- Get more data for training
- Feature engineering to extract more predictive features
- Tune Hyperparameter

## References

- [Kaggle - Churn Modelling Dataset](https://www.kaggle.com/datasets/shrutimechlearn/churn-modelling)
- [Artificial Neural Network (ANN)](https://machinelearningmastery.com/neural-networks-crash-course/)
- [Keras Sequential Model](https://keras.io/api/models/sequential/)

## Articles to Read
- [Activation Functions and Optimizers for Deep Learning Models](https://becominghuman.ai/activation-functions-and-optimizers-for-deep-learning-models-5a1181649d6b?gi=d75a2b572f7a)
- [A Simple Introduction to Dropout Regularization (With Code!)](https://medium.com/analytics-vidhya/a-simple-introduction-to-dropout-regularization-with-code-5279489dda1e)
- [Discover Feature Engineering, How to Engineer Features and How to Get Good at It](https://machinelearningmastery.com/discover-feature-engineering-how-to-engineer-features-and-how-to-get-good-at-it/)
