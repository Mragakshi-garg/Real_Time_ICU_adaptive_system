import pandas as pd
import numpy as np
import os
import time
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.utils import class_weight
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv1D, Flatten, MaxPooling1D, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import warnings
warnings.filterwarnings('ignore')

def preprocess_data(filepath='data/streaming_data.csv'):
    print("Loading data...")
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Data file not found: {filepath}")
    
    df = pd.read_csv(filepath)
    if 'heart_rate' in df.columns:
        df = df.rename(columns={
            'heart_rate': 'hr',
            'systolic_bp': 'sysbp',
            'diastolic_bp': 'diabp',
            'mean_bp': 'meanbp'
        })
        
        if 'timestamp' in df.columns and 'subject_id' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df = df.sort_values(by=['subject_id', 'timestamp'])
            for col in ['hr', 'spo2', 'sysbp', 'diabp', 'meanbp']:
                if col in df.columns:
                    df[col] = df.groupby('subject_id')[col].ffill()
                    
        required_cols = ['hr', 'spo2', 'sysbp', 'meanbp']
        for col in required_cols:
            if col not in df.columns:
                df[col] = np.nan
        df = df.dropna(subset=required_cols)
        return df
    
    return df

def generate_labels(df):
    if 'warning' in df.columns:
        df['current_risk'] = df['warning']
    else:
        raise ValueError("No ground-truth target label ('warning') found in the dataset.")

    if 'subject_id' in df.columns:
        df['target_label'] = df.groupby('subject_id')['current_risk'].shift(-1)
    elif 'stay_id' in df.columns:
        df['target_label'] = df.groupby('stay_id')['current_risk'].shift(-1)
    else:
        df['target_label'] = df['current_risk']
        
    df = df.dropna(subset=['target_label'])
    return df

def build_optimized_mlp(input_dim):
    model = Sequential([
        Dense(128, input_shape=(input_dim,)),
        BatchNormalization(),
        Dense(128, activation='relu'),
        Dropout(0.3),
        Dense(64, activation='relu'),
        Dropout(0.3),
        Dense(32, activation='relu'),
        Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer=Adam(learning_rate=0.001), loss='binary_crossentropy', metrics=['accuracy'])
    return model

def build_optimized_cnn(input_dim):
    model = Sequential([
        Conv1D(filters=32, kernel_size=2, padding='same', input_shape=(input_dim, 1)),
        BatchNormalization(),
        Conv1D(filters=32, kernel_size=2, activation='relu', padding='same'),
        MaxPooling1D(pool_size=2),
        Dropout(0.3),
        Flatten(),
        Dense(32, activation='relu'),
        Dropout(0.2),
        Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer=Adam(learning_rate=0.001), loss='binary_crossentropy', metrics=['accuracy'])
    return model

def evaluate_model(y_true, y_pred, y_pred_proba, name, latency):
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    roc_auc = roc_auc_score(y_true, y_pred_proba)
    
    print(f"\n--- Metrics for {name} ---")
    print(f"Accuracy:      {acc:.4f}")
    print(f"ROC-AUC:       {roc_auc:.4f}")
    print(f"Precision:     {prec:.4f}")
    print(f"Recall:        {rec:.4f}")
    print(f"F1-score:      {f1:.4f}")
    print(f"Confusion Mat:\n{confusion_matrix(y_true, y_pred)}")
    print(f"Training Time: {latency:.4f} seconds")
    return {'Accuracy': acc, 'ROC-AUC': roc_auc, 'Precision': prec, 'Recall': rec, 'F1-score': f1, 'Training Time': latency}

def train_and_compare():
    df = preprocess_data('data/streaming_data.csv')
    df = generate_labels(df)
    
    feature_cols = ['hr', 'spo2', 'sysbp', 'meanbp']
    X = df[feature_cols].values
    y = df['target_label'].values
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # Scale features for Neural Networks
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Compute class weights for imbalanced dataset
    class_weights = class_weight.compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
    class_weight_dict = {i: weight for i, weight in enumerate(class_weights)}
    print(f"Class Weights applied: {class_weight_dict}")
    
    # Callbacks for optimization
    early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-5)
    callbacks = [early_stopping, reduce_lr]
    
    # Reshape for CNN
    X_train_cnn = X_train_scaled.reshape(X_train_scaled.shape[0], X_train_scaled.shape[1], 1)
    X_test_cnn = X_test_scaled.reshape(X_test_scaled.shape[0], X_test_scaled.shape[1], 1)
    
    results = {}
    
    print("\nTraining Deep Learning Models with Optimizations (Class weights, Early Stopping, Deeper Architecture)...")
    
    # 1. Train Optimized MLP (Neural Network)
    mlp_model = build_optimized_mlp(len(feature_cols))
    start_time = time.time()
    mlp_model.fit(X_train_scaled, y_train, epochs=100, batch_size=32, verbose=0, validation_split=0.2, 
                  class_weight=class_weight_dict, callbacks=callbacks)
    mlp_latency = time.time() - start_time
    
    y_pred_proba_mlp = mlp_model.predict(X_test_scaled, verbose=0).flatten()
    y_pred_mlp = (y_pred_proba_mlp > 0.5).astype(int)
    results['Optimized Neural Network (MLP)'] = evaluate_model(y_test, y_pred_mlp, y_pred_proba_mlp, 'Optimized Neural Network (MLP)', mlp_latency)
    
    # 2. Train Optimized 1D CNN
    cnn_model = build_optimized_cnn(len(feature_cols))
    start_time = time.time()
    cnn_model.fit(X_train_cnn, y_train, epochs=100, batch_size=32, verbose=0, validation_split=0.2,
                  class_weight=class_weight_dict, callbacks=callbacks)
    cnn_latency = time.time() - start_time
    
    y_pred_proba_cnn = cnn_model.predict(X_test_cnn, verbose=0).flatten()
    y_pred_cnn = (y_pred_proba_cnn > 0.5).astype(int)
    results['Optimized 1D CNN'] = evaluate_model(y_test, y_pred_cnn, y_pred_proba_cnn, 'Optimized 1D CNN', cnn_latency)
    
    # Save DL models
    os.makedirs('data', exist_ok=True)
    mlp_model.save('data/icu_risk_model_nn.keras')
    cnn_model.save('data/icu_risk_model_cnn.keras')
    print("\nSaved Neural Network models in data/ as .keras files.")
    
    # 3. Evaluate the existing baseline model (from train.py)
    existing_model_path = 'data/icu_risk_model.pkl'
    if os.path.exists(existing_model_path):
        existing_model = joblib.load(existing_model_path)
        start_time = time.time()
        
        if hasattr(existing_model, "predict_proba"):
            y_pred_proba_base = existing_model.predict_proba(X_test)[:, 1]
        else:
            y_pred_proba_base = existing_model.predict(X_test)
            
        y_pred_base = existing_model.predict(X_test)
        base_latency = 0.0
        
        results['Existing Model (.pkl)'] = evaluate_model(y_test, y_pred_base, y_pred_proba_base, 'Existing Model (.pkl)', base_latency)
    else:
        print(f"\nExisting model {existing_model_path} not found.")
        
    print("\n================ COMPARATIVE ANALYSIS ================")
    df_results = pd.DataFrame(results).T
    print(df_results.to_string())
    
    # Write analysis to a markdown file
    with open('data/comparative_analysis.md', 'w') as f:
        f.write("# Model Comparative Analysis (Optimized)\n\n")
        f.write("This report compares the performance of the **optimized** Deep Learning models against the classical baseline.\n")
        f.write("Optimizations included: Architecture improvements (BatchNormalization, deeper layers), Early Stopping, Learning Rate Reduction, and Class Weights.\n\n")
        f.write("### Model Metrics\n\n")
        f.write(df_results.to_string())

if __name__ == '__main__':
    train_and_compare()
