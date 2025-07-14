# Financial Transaction Fraud Detection

This project implements a machine learning-based fraud detection system for financial transactions. The system uses a Random Forest classifier to identify potentially fraudulent transactions with high precision.

## Project Overview

The project consists of two main components:
1. A machine learning model trained on financial transaction data
2. A Streamlit web application for real-time fraud prediction

### Model Performance
- F1 Score for fraud detection: 0.87
- Precision: 97% (97% of predicted frauds were actual frauds)
- Recall: 79% (79% of actual frauds were detected)

## Features

The model takes into account several transaction features:
- Transaction Type (PAYMENT, TRANSFER, CASH_OUT)
- Transaction Amount
- Origin Account Balance (before and after transaction)
- Destination Account Balance (before and after transaction)

## Project Structure
```
Fraud_Detection/
├── data/
│   └── Fraud_Dataset.csv
├── analysis_model.ipynb
├── fraud_detection.py
├── fraud_detection_pipeline.pkl
└── requirements.txt
```

## Installation

1. Clone the repository:
```bash
git clone https://github.com/PH0EN1X1509/Fraud_Detection.git
cd Fraud_Detection
```

2. Install required packages:
```bash
pip install -r requirements.txt
```

## Usage

### Running the Web Application

To start the Streamlit web interface:
```bash
streamlit run fraud_detection.py
```

The application will open in your default web browser where you can:
- Select transaction type
- Enter transaction amount
- Input sender's and receiver's balance details
- Get instant fraud prediction results

### Model Development

The `analysis_model.ipynb` notebook contains:
- Exploratory Data Analysis (EDA)
- Model training and evaluation
- Feature importance analysis
- Model performance metrics

## Technical Details

### Model Selection
Multiple models were evaluated using 5-fold cross-validation:
- Logistic Regression
- Random Forest
- Gradient Boosting
- Decision Tree

Random Forest was selected as the final model due to its superior performance.

### Data Processing
- Numerical features are standardized using StandardScaler
- Categorical features are encoded using OneHotEncoder
- Model pipeline ensures consistent preprocessing for training and prediction

## Requirements

- Python 3.7+
- pandas
- numpy
- scikit-learn
- streamlit
- seaborn
- matplotlib
- joblib
