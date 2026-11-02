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

def perform_unsupervised_clustering(X_train_proc, df_train_orig):
    """Executes Unit IV Unsupervised Learning (K-Means Clustering)."""
    print("\n" + "="*50)
    print(" UNIT IV: UNSUPERVISED LEARNING (STUDENT SEGMENTATION)")
    print("="*50)
    
    n_clusters = min(3, len(X_train_proc))
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    clusters = kmeans.fit_predict(X_train_proc)
    
    df_clustered = df_train_orig.copy()
    df_clustered['Risk_Cluster'] = clusters
    
    print("\nCluster Profiles (Mean Feature Values per Cluster):")
    summary = df_clustered.groupby('Risk_Cluster')[['Study_Hours', 'Attendance', 'Assignment_Score', 'LMS_Logins_Per_Week']].mean()
    print(summary)
    
    return kmeans
    def run_intervention_engine(rf_model, X_test_orig, X_test_proc):
    """Computes Priority Scores and generates the Intervention Queue."""
    print("\n" + "="*50)
    print(" INTERVENTION PRIORITIZATION ENGINE")
    print("="*50)
    
    probabilities = rf_model.predict_proba(X_test_proc)[:, 1]
    results = X_test_orig.copy()
    results['Predicted_Risk_Probability'] = probabilities
    
    # Intervention Priority Calculation Formula
    results['Priority_Score'] = (
        results['Predicted_Risk_Probability'] *
        (1 + (100 - results['Attendance']) / 100) *
        (1 + (100 - results['Assignment_Score']) / 100)
    )
    
    prioritized_queue = results.sort_values(by='Priority_Score', ascending=False)
    print("\nPrioritized Students Queue for Advisor Intervention:")
    display_cols = ['Study_Hours', 'Attendance', 'Assignment_Score', 'Predicted_Risk_Probability', 'Priority_Score']
    print(prioritized_queue[display_cols].to_string())
    
    return prioritized_queue

if __name__ == "__main__":
    df = load_and_prepare_data("student_data.csv")
    X_train_orig, X_test_orig, X_tr_proc, X_te_proc, y_train, y_test, preprocessor = preprocess_data(df)
    trained_rf = train_and_evaluate_supervised(X_tr_proc, X_te_proc, y_train, y_test)
    kmeans_model = perform_unsupervised_clustering(X_tr_proc, X_train_orig)
    intervention_queue = run_intervention_engine(trained_rf, X_test_orig, X_te_proc)
