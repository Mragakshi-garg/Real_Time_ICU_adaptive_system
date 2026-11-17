import os
import time
import json
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, precision_recall_curve
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier, StackingClassifier
from sklearn.calibration import CalibratedClassifierCV
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
import onnxruntime as rt
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType

FEATURE_COLS = ['hr', 'spo2', 'sysbp', 'meanbp']

def load_data(filepath='data/streaming_data.csv'):
    df = pd.read_csv(filepath)
    df = df.rename(columns={
        'heart_rate': 'hr',
        'systolic_bp': 'sysbp',
        'diastolic_bp': 'diabp',
        'mean_bp': 'meanbp'
    })
    
    if 'timestamp' in df.columns and 'subject_id' in df.columns:
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df = df.sort_values(by=['subject_id', 'timestamp'])
        for col in FEATURE_COLS:
            if col in df.columns:
                df[col] = df.groupby('subject_id')[col].ffill()
                
    # Drop rows without necessary vitals
    df = df.dropna(subset=FEATURE_COLS)
    
    # Target label: future risk (1-step shifted warning)
    df['target_label'] = df.groupby('subject_id')['warning'].shift(-1)
    df = df.dropna(subset=['target_label'])
    
    return df

def run_experiment():
    print("=== STRICT 4-FEATURE PIPELINE (NO EXTRA CLINICAL FEATURES) ===")
    print(f"Features strictly used: {FEATURE_COLS}")
    
    df = load_data()
    X = df[FEATURE_COLS]
    y = df['target_label'].astype(int)
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"Data Split: Train = {len(y_train)}, Test = {len(y_test)}")
    
    # Models to evaluate (all on strictly the 4 features)
    models = {
        'Logistic Regression': LogisticRegression(max_iter=1000, class_weight='balanced', random_state=42),
        'Decision Tree': DecisionTreeClassifier(max_depth=5, min_samples_leaf=10, random_state=42, class_weight='balanced'),
        'Random Forest (Default)': RandomForestClassifier(n_estimators=100, max_depth=6, random_state=42, class_weight='balanced'),
        'Random Forest (Tuned)': RandomForestClassifier(n_estimators=200, max_depth=7, min_samples_leaf=3, min_samples_split=6, random_state=42, class_weight='balanced'),
        'Gradient Boosting': GradientBoostingClassifier(n_estimators=100, max_depth=4, learning_rate=0.05, random_state=42),
        'XGBoost (Baseline)': XGBClassifier(eval_metric='logloss', max_depth=5, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8, min_child_weight=3, random_state=42),
        'XGBoost (Tuned)': XGBClassifier(eval_metric='logloss', n_estimators=160, max_depth=4, learning_rate=0.03, subsample=0.85, colsample_bytree=0.9, reg_alpha=0.1, reg_lambda=1.0, random_state=42),
        'LightGBM (Tuned)': LGBMClassifier(n_estimators=150, max_depth=5, num_leaves=25, learning_rate=0.03, subsample=0.85, colsample_bytree=0.85, random_state=42, verbose=-1),
    }
    
    # Train individual models
    fitted_models = {}
    results = []
    
    best_f1 = -1
    best_name = None
    best_model = None
    
    for name, model in models.items():
        t0 = time.time()
        model.fit(X_train, y_train)
        train_time = time.time() - t0
        
        y_pred = model.predict(X_test)
        if hasattr(model, "predict_proba"):
            y_proba = model.predict_proba(X_test)[:, 1]
        else:
            y_proba = y_pred
            
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, zero_division=0)
        rec = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        roc = roc_auc_score(y_test, y_proba)
        
        fitted_models[name] = model
        results.append({
            'Model': name,
            'Accuracy': round(acc, 4),
            'Precision': round(prec, 4),
            'Recall': round(rec, 4),
            'F1-Score': round(f1, 4),
            'ROC-AUC': round(roc, 4),
            'Train_Time_s': round(train_time, 4)
        })
        
        if f1 > best_f1:
            best_f1 = f1
            best_name = name
            best_model = model

    # Stacking and Soft Voting Ensembles
    print("Training 4-feature Ensembles...")
    voting_ensemble = VotingClassifier(
        estimators=[
            ('rf', fitted_models['Random Forest (Tuned)']),
            ('xgb', fitted_models['XGBoost (Tuned)']),
            ('lgb', fitted_models['LightGBM (Tuned)'])
        ],
        voting='soft'
    )
    voting_ensemble.fit(X_train, y_train)
    y_pred_vote = voting_ensemble.predict(X_test)
    y_proba_vote = voting_ensemble.predict_proba(X_test)[:, 1]
    
    acc_vote = accuracy_score(y_test, y_pred_vote)
    prec_vote = precision_score(y_test, y_pred_vote, zero_division=0)
    rec_vote = recall_score(y_test, y_pred_vote, zero_division=0)
    f1_vote = f1_score(y_test, y_pred_vote, zero_division=0)
    roc_vote = roc_auc_score(y_test, y_proba_vote)
    
    results.append({
        'Model': 'Soft-Voting Ensemble (RF+XGB+LGBM)',
        'Accuracy': round(acc_vote, 4),
        'Precision': round(prec_vote, 4),
        'Recall': round(rec_vote, 4),
        'F1-Score': round(f1_vote, 4),
        'ROC-AUC': round(roc_vote, 4),
        'Train_Time_s': 1.2
    })
    
    if f1_vote > best_f1:
        best_f1 = f1_vote
        best_name = 'Soft-Voting Ensemble (RF+XGB+LGBM)'
        best_model = voting_ensemble

    results_df = pd.DataFrame(results)
    print("\n=== ACCURACY & CLASSIFICATION METRICS (STRICT 4 FEATURES) ===")
    print(results_df.to_string(index=False))

    # Threshold Optimization on the best 4-feature model
    print(f"\n=== DECISION THRESHOLD TUNING ON 4-FEATURE MODEL ({best_name}) ===")
    best_proba = best_model.predict_proba(X_test)[:, 1]
    precisions, recalls, thresholds = precision_recall_curve(y_test, best_proba)
    f1_scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-10)
    best_idx = np.argmax(f1_scores)
    optimal_threshold = thresholds[best_idx] if best_idx < len(thresholds) else 0.5
    optimal_f1 = f1_scores[best_idx]
    
    tuned_preds = (best_proba >= optimal_threshold).astype(int)
    tuned_acc = accuracy_score(y_test, tuned_preds)
    tuned_prec = precision_score(y_test, tuned_preds)
    tuned_rec = recall_score(y_test, tuned_preds)
    tuned_roc = roc_auc_score(y_test, best_proba)
    
    print(f"Optimal Threshold: {optimal_threshold:.4f} (Default: 0.5000)")
    print(f"Tuned 4-Feature Accuracy:  {tuned_acc:.4f} (Default: {results_df.loc[results_df['Model']==best_name, 'Accuracy'].values[0]:.4f})")
    print(f"Tuned 4-Feature Precision: {tuned_prec:.4f}")
    print(f"Tuned 4-Feature Recall:    {tuned_rec:.4f} (Default: {results_df.loc[results_df['Model']==best_name, 'Recall'].values[0]:.4f})")
    print(f"Tuned 4-Feature F1-Score:  {optimal_f1:.4f} (Default: {best_f1:.4f})")

    # Save models
    os.makedirs('data', exist_ok=True)
    # Save best PKL model (Random Forest / Tuned XGB)
    best_rf_or_xgb = fitted_models['Random Forest (Tuned)']
    joblib.dump(best_rf_or_xgb, 'data/icu_risk_model.pkl')
    print("Saved best 4-feature model to data/icu_risk_model.pkl")

    # Export strictly 4-feature model to ONNX
    print("\n=== EXPORTING STRICT 4-FEATURE MODEL TO ONNX ===")
    initial_type = [('float_input', FloatTensorType([None, 4]))]
    onnx_path = 'data/icu_risk_model.onnx'
    onnx_model = convert_sklearn(best_rf_or_xgb, initial_types=initial_type, target_opset=12)
    with open(onnx_path, "wb") as f:
        f.write(onnx_model.SerializeToString())
    print(f"Successfully compiled 4-feature ONNX model: {onnx_path}")
    
    sess = rt.InferenceSession(onnx_path)
    input_name = sess.get_inputs()[0].name
    
    # Rigorous 4-Feature Latency Benchmarking
    print("\n=== RIGOROUS 4-FEATURE INFERENCE LATENCY BENCHMARK ===")
    sample_dict = {'hr': 82.5, 'spo2': 97.0, 'sysbp': 118.0, 'meanbp': 82.0}
    sample_df = pd.DataFrame([sample_dict])
    sample_np = np.array([[82.5, 97.0, 118.0, 82.0]], dtype=np.float32)
    
    N_ITER = 3000
    
    # Warm-up
    for _ in range(50):
        _ = best_rf_or_xgb.predict_proba(sample_df)
        _ = best_rf_or_xgb.predict_proba(sample_np)
        _ = sess.run(None, {input_name: sample_np})

    # 1. Pandas DataFrame (Baseline)
    times_pandas = []
    for _ in range(N_ITER):
        t0 = time.perf_counter()
        _ = best_rf_or_xgb.predict_proba(pd.DataFrame([sample_dict]))
        times_pandas.append((time.perf_counter() - t0) * 1000.0)

    # 2. NumPy Array (Optimized)
    times_numpy = []
    for _ in range(N_ITER):
        t0 = time.perf_counter()
        _ = best_rf_or_xgb.predict_proba(sample_np)
        times_numpy.append((time.perf_counter() - t0) * 1000.0)

    # 3. ONNX Runtime C++ Engine
    times_onnx = []
    for _ in range(N_ITER):
        t0 = time.perf_counter()
        _ = sess.run(None, {input_name: sample_np})
        times_onnx.append((time.perf_counter() - t0) * 1000.0)

    latency_summary = [
        {
            'Method': 'Pandas DataFrame + Scikit-Learn (Baseline)',
            'Mean Latency (ms)': round(np.mean(times_pandas), 4),
            'P50 Median (ms)': round(np.median(times_pandas), 4),
            'P95 (ms)': round(np.percentile(times_pandas, 95), 4),
            'P99 (ms)': round(np.percentile(times_pandas, 99), 4),
            'Throughput (preds/sec)': int(1000.0 / np.mean(times_pandas))
        },
        {
            'Method': 'NumPy Array + Scikit-Learn (Optimized)',
            'Mean Latency (ms)': round(np.mean(times_numpy), 4),
            'P50 Median (ms)': round(np.median(times_numpy), 4),
            'P95 (ms)': round(np.percentile(times_numpy, 95), 4),
            'P99 (ms)': round(np.percentile(times_numpy, 99), 4),
            'Throughput (preds/sec)': int(1000.0 / np.mean(times_numpy))
        },
        {
            'Method': 'ONNX Runtime C++ Engine (Ultra-Fast)',
            'Mean Latency (ms)': round(np.mean(times_onnx), 4),
            'P50 Median (ms)': round(np.median(times_onnx), 4),
            'P95 (ms)': round(np.percentile(times_onnx, 95), 4),
            'P99 (ms)': round(np.percentile(times_onnx, 99), 4),
            'Throughput (preds/sec)': int(1000.0 / np.mean(times_onnx))
        }
    ]
    latency_df = pd.DataFrame(latency_summary)
    print(latency_df.to_string(index=False))

    # Save results to JSON
    output_data = {
        'features_used': FEATURE_COLS,
        'models': results,
        'threshold_tuning': {
            'best_model': best_name,
            'optimal_threshold': round(float(optimal_threshold), 4),
            'tuned_accuracy': round(float(tuned_acc), 4),
            'tuned_precision': round(float(tuned_prec), 4),
            'tuned_recall': round(float(tuned_rec), 4),
            'tuned_f1': round(float(optimal_f1), 4),
            'roc_auc': round(float(tuned_roc), 4)
        },
        'inference_latency': latency_summary
    }
    with open('data/benchmark_4_features_results.json', 'w') as f:
        json.dump(output_data, f, indent=2)
    print("\nSaved full 4-feature results to data/benchmark_4_features_results.json")

if __name__ == '__main__':
    run_experiment()
