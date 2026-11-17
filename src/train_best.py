import pandas as pd
import numpy as np
import os
import time
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
from sklearn.ensemble import ExtraTreesClassifier, VotingClassifier, StackingClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
import warnings
warnings.filterwarnings('ignore')

def explore_and_preprocess(filepath='data/streaming_data.csv'):
    print("================ DATA ANALYSIS ================")
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Data file not found: {filepath}")
    
    df = pd.read_csv(filepath)
    print(f"Original shape: {df.shape}")
    
    # Check what columns exist
    print(f"Columns: {df.columns.tolist()}")
    
    # Rename for consistency
    if 'heart_rate' in df.columns:
        df = df.rename(columns={
            'heart_rate': 'hr',
            'systolic_bp': 'sysbp',
            'diastolic_bp': 'diabp',
            'mean_bp': 'meanbp'
        })
    
    # Print missing values summary
    print("\nMissing Values before processing:")
    print(df[['hr', 'spo2', 'sysbp', 'diabp', 'meanbp']].isnull().sum())
    
    # Sort and forward-fill by subject_id
    if 'timestamp' in df.columns and 'subject_id' in df.columns:
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df = df.sort_values(by=['subject_id', 'timestamp'])
        for col in ['hr', 'spo2', 'sysbp', 'diabp', 'meanbp']:
            if col in df.columns:
                df[col] = df.groupby('subject_id')[col].ffill()
    
    # Use ALL 5 vital signs this time (previously diabp was ignored)
    feature_cols = ['hr', 'spo2', 'sysbp', 'diabp', 'meanbp']
    df = df.dropna(subset=feature_cols)
    print(f"\nShape after dropping remaining NaNs: {df.shape}")
    
    print("\nFeature Correlations with target (warning):")
    if 'warning' in df.columns:
        corrs = df[feature_cols + ['warning']].corr()['warning'].sort_values(ascending=False)
        print(corrs)
        
    return df, feature_cols

def generate_labels(df):
    if 'warning' in df.columns:
        df['current_risk'] = df['warning']
    else:
        raise ValueError("No ground-truth target label ('warning') found in the dataset.")

    if 'subject_id' in df.columns:
        df['target_label'] = df.groupby('subject_id')['current_risk'].shift(-1)
    else:
        df['target_label'] = df['current_risk']
        
    df = df.dropna(subset=['target_label'])
    return df

def evaluate_model(model, X_test, y_test, name, latency):
    if hasattr(model, "predict_proba"):
        y_pred_proba = model.predict_proba(X_test)[:, 1]
    else:
        y_pred_proba = model.predict(X_test)
        
    y_pred = model.predict(X_test)
    
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    roc_auc = roc_auc_score(y_test, y_pred_proba)
    
    print(f"\n--- Metrics for {name} ---")
    print(f"Accuracy:      {acc:.4f}")
    print(f"ROC-AUC:       {roc_auc:.4f}")
    print(f"Precision:     {prec:.4f}")
    print(f"Recall:        {rec:.4f}")
    print(f"F1-score:      {f1:.4f}")
    print(f"Training Time: {latency:.4f} seconds")
    return {'Accuracy': acc, 'ROC-AUC': roc_auc, 'Precision': prec, 'Recall': rec, 'F1-score': f1, 'Training Time': latency}

def train_best_model():
    df, feature_cols = explore_and_preprocess('data/streaming_data.csv')
    df = generate_labels(df)
    
    X = df[feature_cols].values
    y = df['target_label'].values
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    results = {}
    print("\n================ MODEL TRAINING ================")
    
    # Model 1: Extra Trees (A different tree ensemble)
    print("Training ExtraTreesClassifier...")
    et = ExtraTreesClassifier(n_estimators=200, max_depth=15, class_weight='balanced', random_state=42)
    start = time.time()
    et.fit(X_train_scaled, y_train)
    et_time = time.time() - start
    results['Extra Trees'] = evaluate_model(et, X_test_scaled, y_test, 'Extra Trees', et_time)
    
    # Model 2: Voting Classifier (Combines XGBoost + LightGBM + Extra Trees)
    print("Training Voting Ensemble (XGBoost + LightGBM + ExtraTrees)...")
    xgb = XGBClassifier(use_label_encoder=False, eval_metric='logloss', max_depth=5, learning_rate=0.05, random_state=42)
    lgbm = LGBMClassifier(max_depth=5, learning_rate=0.05, random_state=42)
    voting = VotingClassifier(estimators=[('xgb', xgb), ('lgbm', lgbm), ('et', et)], voting='soft')
    start = time.time()
    voting.fit(X_train_scaled, y_train)
    voting_time = time.time() - start
    results['Voting Ensemble'] = evaluate_model(voting, X_test_scaled, y_test, 'Voting Ensemble', voting_time)
    
    # Model 3: Stacking Classifier (The ultimate way to combine models)
    print("Training Stacking Ensemble...")
    stacking = StackingClassifier(
        estimators=[('xgb', xgb), ('lgbm', lgbm), ('et', et)],
        final_estimator=XGBClassifier(use_label_encoder=False, eval_metric='logloss', max_depth=3, random_state=42),
        cv=3
    )
    start = time.time()
    stacking.fit(X_train_scaled, y_train)
    stacking_time = time.time() - start
    results['Stacking Ensemble'] = evaluate_model(stacking, X_test_scaled, y_test, 'Stacking Ensemble', stacking_time)
    
    print("\n================ FINAL COMPARISON ================")
    df_results = pd.DataFrame(results).T
    print(df_results.to_string())
    
    # Determine best model by F1-score
    best_model_name = df_results['F1-score'].idxmax()
    print(f"\n🏆 The Best Possible Model based on F1-Score is: {best_model_name}")

if __name__ == '__main__':
    train_best_model()
