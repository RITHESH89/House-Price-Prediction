# House-Price-Prediction

# Project Overview

The Ames Housing dataset, compiled by Dean De Cock, is a comprehensive and modernized alternative to the Boston Housing dataset. It contains detailed information about residential properties, including structural features, quality ratings, and location-based attributes.

The objective of this project is to predict property sale prices using advanced regression techniques. Due to the dataset’s high dimensionality and complex feature relationships, extensive feature engineering and preprocessing are required to achieve high predictive accuracy.

# Problem Statement
To build a machine learning model that accurately predicts the sale price of houses based on multiple numerical and categorical features.

# Dataset Information
- Source: Ames Housing Dataset (Dean De Cock)
- Number of Features: 70+ explanatory variables
- Target Variable: SalePrice
- Data Type: Numerical & Categorical
- Challenges:
   - Missing values
   - High cardinality categorical variables
   - Non-linear relationships
   - Skewed target distribution
 
# Technologies & Libraries Used

- Programming Language: Python
- Libraries:
  - NumPy
  - Pandas
  - Matplotlib
  - Seaborn
  - Scikit-learn
  - XGBoost

# Project Workflow
1️⃣ Data Loading & Exploration
- Loaded training and test datasets
- Performed exploratory data analysis (EDA)
- Identified missing values and outliers

2️⃣ Data Preprocessing
- Handled missing numerical features using median imputation
- Filled missing categorical values with meaningful placeholders
- Removed irrelevant or redundant features

3️⃣ Feature Engineering
- Created new meaningful features such as:
- Total square footage
- Applied log transformation to the target variable to reduce skewness
- Encoded categorical variables using label encoding

4️⃣ Model Building
- Implemented and compared multiple regression models:
- Linear Regression (Baseline)
- Random Forest Regressor
- XGBoost Regressor (Final Model)

5️⃣ Model Evaluation
- Used Root Mean Squared Error (RMSE) as the evaluation metric
- Compared model performance on validation data
- Selected the best-performing model based on RMSE

# Project Structure
  
   House-Price-Prediction/
│
├── data/
│   └── train.csv
│
├── src/
│   ├── preprocessing.py
│   └── train_model.py
│
├── model/
│   └── model.pkl
│
├── requirements.txt
└── README.md


# Results

- XGBoost regression achieved the best performance
- Feature engineering significantly improved prediction accuracy
- Log transformation of the target variable reduced prediction error
