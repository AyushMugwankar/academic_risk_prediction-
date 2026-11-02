# academic_risk_prediction-
AN AI PREDICTION MODULE
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, roc_auc_score
from sklearn.cluster import KMeans

def load_and_prepare_data(filepath="student_data.csv"):
    """Loads dataset from CSV and inspects initial state."""
    print("Loading dataset from:", filepath)
    df = pd.read_csv(filepath)
    return df

def preprocess_data(df):
    """Executes Unit II Data Preparation tasks: Imputation, Encoding, and Scaling."""
    df_clean = df.copy()
    
    # Median Imputation for missing values
    df_clean['Attendance'] = df_clean['Attendance'].fillna(df_clean['Attendance'].median())
    df_clean['Assignment_Score'] = df_clean['Assignment_Score'].fillna(df_clean['Assignment_Score'].median())
    
    X = df_clean.drop(columns=['High_Risk'])
    y = df_clean['High_Risk']
    
    num_cols = ['Study_Hours', 'Attendance', 'Assignment_Score', 'LMS_Logins_Per_Week']
    cat_cols = ['Department']
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), num_cols),
            ('cat', OneHotEncoder(drop='first', sparse_output=False), cat_cols)
        ]
    )
