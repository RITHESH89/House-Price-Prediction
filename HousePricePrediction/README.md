# 🏠 House Price Prediction - Advanced Regression

## 📌 Project Overview

This project predicts house sale prices using advanced regression
techniques. The dataset used is the Ames Housing Dataset, which contains
79 explanatory variables describing residential homes.

The goal is to build a machine learning model that accurately predicts
property sale prices based on housing features.

------------------------------------------------------------------------

## 📊 Dataset Information

-   Dataset: Ames Housing Dataset\
-   Target Variable: SalePrice\
-   Total Features: 79\
-   Problem Type: Regression

------------------------------------------------------------------------

## ⚙️ Technologies Used

-   Python
-   Pandas
-   NumPy
-   Scikit-learn
-   Streamlit
-   Joblib

------------------------------------------------------------------------

## 🔍 Feature Engineering

-   Handled missing values using median and categorical replacement
-   Converted categorical variables using encoding
-   Selected important numerical housing features
-   Applied model training with proper train-test split

------------------------------------------------------------------------

## 🤖 Model Used

Random Forest Regressor

Why Random Forest? - Handles non-linearity well - Reduces overfitting
using ensemble learning - Works effectively on structured datasets

------------------------------------------------------------------------

## 📈 Results

Evaluation Metric: RMSE (Root Mean Squared Error)

The model achieved strong predictive performance on test data. Feature
engineering significantly improved model accuracy.

------------------------------------------------------------------------

## 🚀 How to Run the Project

1.  Install dependencies:

        pip install -r requirements.txt

2.  Run the Streamlit app:

        streamlit run app.py

3.  Enter house details and get predicted price.

------------------------------------------------------------------------

## 📂 Project Structure

    house-price-prediction/
    │
    ├── data/
    ├── models/
    │   └── model.pkl
    ├── app.py
    ├── requirements.txt
    └── README.md

------------------------------------------------------------------------

## 🔮 Future Improvements

-   Use XGBoost for better performance
-   Perform hyperparameter tuning
-   Add cross-validation
-   Deploy using cloud platforms

------------------------------------------------------------------------

## 👨‍💻 Author

Rithesh

B.Com Graduate \| Aspiring AI Engineer
