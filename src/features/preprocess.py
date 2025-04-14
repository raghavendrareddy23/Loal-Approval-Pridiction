import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
import joblib

def preprocess(df):
    # Drop duplicates
    df = df.drop_duplicates()

    # Separate features and target
    # target = 'Loan_Status'
    # if target in df.columns:
    #     y = df[target].map({'N': 0, 'Y': 1})
    #     df = df.drop(columns=[target])
    # else:
    #     y = None

    y = df['loan_status']

    df = df.drop('loan_status', axis = 1)

    # Identify column types
    numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
    categorical_cols = df.select_dtypes(include=['object']).columns.tolist()

    # Define numeric pipeline
    numeric_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    # Define categorical pipeline
    categorical_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('encoder', OneHotEncoder(handle_unknown='ignore'))
    ])

    # Combine into ColumnTransformer
    preprocessor = ColumnTransformer([
        ('num', numeric_pipeline, numeric_cols),
        ('cat', categorical_pipeline, categorical_cols)
    ])

    # Fit and transform
    X = preprocessor.fit_transform(df)

    joblib.dump(preprocessor, "model/preprocessor.pkl")

    return X, y, preprocessor
