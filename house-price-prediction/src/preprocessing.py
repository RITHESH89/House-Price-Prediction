import pandas as pd
import numpy as np

def preprocess_data(path):
    df = pd.read_csv(path)

    # Log transform target (important for skewed prices)
    y = np.log1p(df["SalePrice"])

    # Drop target from dataframe
    df = df.drop("SalePrice", axis=1)

    # Fill missing values
    df.fillna(df.median(numeric_only=True), inplace=True)
    df.fillna("None", inplace=True)

    # Convert categorical features
    df = pd.get_dummies(df, drop_first=True)

    return df, y
