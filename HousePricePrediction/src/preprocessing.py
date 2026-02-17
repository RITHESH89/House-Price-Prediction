
import pandas as pd

def load_data(path):
    return pd.read_csv(path)

def preprocess_data(df):
    # Handle missing values
    df = df.fillna(df.median(numeric_only=True))
    df = df.fillna("None")
    
    # Convert categorical variables
    df = pd.get_dummies(df, drop_first=True)
    
    return df
