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
    try:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.25, random_state=42, stratify=y
        )
    except ValueError:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.25, random_state=42
        )
    
    X_train_processed = preprocessor.fit_transform(X_train)
    X_test_processed = preprocessor.transform(X_test)
    
    return X_train, X_test, X_train_processed, X_test_processed, y_train, y_test, preprocessor

def train_and_evaluate_supervised(X_train_proc, X_test_proc, y_train, y_test):
    """Executes Unit III Supervised Classification."""
    print("\n" + "="*50)
    print(" UNIT III: SUPERVISED LEARNING (RISK CLASSIFICATION)")
    print("="*50)
    
    # Model 1: Logistic Regression
    lr = LogisticRegression(random_state=42)
    lr.fit(X_train_proc, y_train)
    lr_preds = lr.predict(X_test_proc)
    print("\n--- Logistic Regression Model Performance ---")
    print(f"Accuracy: {accuracy_score(y_test, lr_preds):.4f}")
    
    # Model 2: Random Forest Classifier
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X_train_proc, y_train)
    rf_preds = rf.predict(X_test_proc)
    print("\n--- Random Forest Model Performance ---")
    print(f"Accuracy: {accuracy_score(y_test, rf_preds):.4f}")
    print("\nConfusion Matrix:\n", confusion_matrix(y_test, rf_preds))
    print("\nClassification Report:\n", classification_report(y_test, rf_preds, zero_division=0))
    
    return rf
