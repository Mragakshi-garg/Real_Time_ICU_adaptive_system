import os
import time
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

FEATURE_COLS = ['hr', 'spo2', 'sysbp', 'meanbp']

df = pd.read_csv('data/streaming_data.csv')
df = df.rename(columns={'heart_rate': 'hr', 'systolic_bp': 'sysbp', 'diastolic_bp': 'diabp', 'mean_bp': 'meanbp'})
df['timestamp'] = pd.to_datetime(df['timestamp'])
df = df.sort_values(by=['subject_id', 'timestamp'])
for col in FEATURE_COLS:
    df[col] = df.groupby('subject_id')[col].ffill()
df = df.dropna(subset=FEATURE_COLS)
df['target_label'] = df.groupby('subject_id')['warning'].shift(-1)
df = df.dropna(subset=['target_label'])

X = df[FEATURE_COLS]
y = df['target_label'].astype(int)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Test 1: Random Forest deeper search for accuracy
rf_grid = {
    'n_estimators': [150, 200, 300],
    'max_depth': [9, 12, 15, 18],
    'min_samples_split': [2, 4, 6],
    'min_samples_leaf': [1, 2, 3],
    'max_features': ['sqrt', 'log2', None],
    'criterion': ['gini', 'entropy'],
    'class_weight': [None]
}

# Test 2: XGBoost search for accuracy
xgb_grid = {
    'n_estimators': [150, 250, 350],
    'learning_rate': [0.03, 0.05, 0.08, 0.12],
    'max_depth': [3, 4, 5, 6, 7],
    'min_child_weight': [1, 2, 4],
    'gamma': [0.0, 0.1],
    'subsample': [0.8, 0.9, 1.0],
    'colsample_bytree': [0.8, 1.0],
    'reg_alpha': [0.0, 0.05],
    'reg_lambda': [0.5, 1.0, 2.0]
}

# Test 3: LightGBM search for accuracy
lgb_grid = {
    'n_estimators': [150, 200, 300],
    'learning_rate': [0.03, 0.05, 0.08],
    'max_depth': [5, 7, 9, -1],
    'num_leaves': [15, 25, 31, 50],
    'min_child_samples': [10, 20, 30],
    'subsample': [0.85, 1.0],
    'colsample_bytree': [0.85, 1.0]
}

print("Running exploratory accuracy search...")
# Run quick Randomized or focused Grid search to compare
from sklearn.model_selection import RandomizedSearchCV

search_rf = RandomizedSearchCV(RandomForestClassifier(random_state=42), rf_grid, n_iter=25, cv=cv, scoring='accuracy', n_jobs=-1, random_state=42)
search_rf.fit(X_train, y_train)
acc_rf = accuracy_score(y_test, search_rf.predict(X_test))
print(f"RF Tuned Accuracy: {acc_rf:.4f} with params: {search_rf.best_params_}")

search_xgb = RandomizedSearchCV(XGBClassifier(eval_metric='logloss', random_state=42), xgb_grid, n_iter=25, cv=cv, scoring='accuracy', n_jobs=-1, random_state=42)
search_xgb.fit(X_train, y_train)
acc_xgb = accuracy_score(y_test, search_xgb.predict(X_test))
print(f"XGB Tuned Accuracy: {acc_xgb:.4f} with params: {search_xgb.best_params_}")

search_lgb = RandomizedSearchCV(LGBMClassifier(random_state=42, verbose=-1), lgb_grid, n_iter=25, cv=cv, scoring='accuracy', n_jobs=-1, random_state=42)
search_lgb.fit(X_train, y_train)
acc_lgb = accuracy_score(y_test, search_lgb.predict(X_test))
print(f"LGB Tuned Accuracy: {acc_lgb:.4f} with params: {search_lgb.best_params_}")
