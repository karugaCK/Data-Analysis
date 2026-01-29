# LSTM Based Traffic Volume Prediction Using the I-94 Dataset

This project implements a Long Short-Term Memory (LSTM) neural network to predict
hourly traffic volume using historical traffic data from the I-94 Traffic Volume Dataset.

## Project Overview
Urban traffic congestion is a major challenge in modern cities.
This system predicts next-hour traffic volume based on the previous 24 hours of traffic data.

## Dataset
- I-94 Traffic Volume Dataset (Kaggle)
- Hourly traffic data (2015–2017)

## Methodology
- Data preprocessing and normalization
- Time-series sequence generation (24-hour sliding window)
- LSTM model training and evaluation
- Performance metrics: RMSE and MAE
- Web-based deployment using Streamlit

## Results
- RMSE ≈ 5.9 vehicles
- MAE ≈ 4.5 vehicles
- Model accurately captures daily traffic patterns

## Web Application
The Streamlit app allows users to input traffic volumes for the previous 24 hours
and predicts traffic volume for the next hour.

## How to Run
1. Clone the repository
```bash
git clone https://github.com/your-username/LSTM-Traffic-Volume-Prediction.git
cd LSTM-Traffic-Volume-Prediction
