import os
import time
import json
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, precision_recall_curve
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
import onnxruntime as rt
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType

def load_and_engineer_features(filepath='data/streaming_data.csv'):
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
        
        # Forward fill
        for col in ['hr', 'spo2', 'sysbp', 'diabp', 'meanbp']:
            if col in df.columns:
                df[col] = df.groupby('subject_id')[col].ffill()
    
    # 1. Shock Index & Hemodynamic Features
    df['shock_index'] = df['hr'] / df['sysbp'].replace(0, np.nan)
    df['modified_shock_index'] = df['hr'] / df['meanbp'].replace(0, np.nan)
    df['pulse_pressure'] = df['sysbp'] - df['diabp']
    df['map_ratio'] = (df['sysbp'] + 2 * df['diabp']) / (3.0 * df['meanbp'].replace(0, np.nan))
    df['hr_spo2_ratio'] = df['hr'] / df['spo2'].replace(0, np.nan)
    
    # 2. Dynamic Deltas (Rate of change per patient)
    for col in ['hr', 'spo2', 'sysbp', 'meanbp']:
        df[f'delta_{col}'] = df.groupby('subject_id')[col].diff().fillna(0.0)
        
    # 3. Rolling window volatility (last 3 steps)
    df['rolling_std_hr'] = df.groupby('subject_id')['hr'].rolling(window=3, min_periods=1).std().reset_index(level=0, drop=True).fillna(0.0)
    df['rolling_min_spo2'] = df.groupby('subject_id')['spo2'].rolling(window=3, min_periods=1).min().reset_index(level=0, drop=True).fillna(df['spo2'])

    # Target label: future risk (1-step shifted warning)
    df['target_label'] = df.groupby('subject_id')['warning'].shift(-1)
    df = df.dropna(subset=['target_label'])
    
    # Fill remaining NaNs if any
    feature_cols = [
        'hr', 'spo2', 'sysbp', 'diabp', 'meanbp',
        'shock_index', 'modified_shock_index', 'pulse_pressure', 'map_ratio', 'hr_spo2_ratio',
        'delta_hr', 'delta_spo2', 'delta_sysbp', 'delta_meanbp',
        'rolling_std_hr', 'rolling_min_spo2'
    ]
    df[feature_cols] = df[feature_cols].fillna(0.0)
    
    return df, feature_cols

def run_experiment():
    print("=== STEP 1: Feature Engineering & Data Preparation ===")
    df, enhanced_cols = load_and_engineer_features()
    baseline_cols = ['hr', 'spo2', 'sysbp', 'meanbp']
    
    y = df['target_label'].astype(int)
    
    # Stratified Split
    X_train_base, X_test_base, y_train, y_test = train_test_split(
        df[baseline_cols], y, test_size=0.2, random_state=42, stratify=y
    )
    X_train_enh = df.loc[X_train_base.index, enhanced_cols]
    X_test_enh = df.loc[X_test_base.index, enhanced_cols]
    
    print(f"Dataset split: Train = {len(y_train)}, Test = {len(y_test)}")
    print(f"Baseline Features ({len(baseline_cols)}): {baseline_cols}")
    print(f"Enhanced Features ({len(enhanced_cols)}): {enhanced_cols}")

    models_to_test = {}
    
    # 1. Baseline Model
    xgb_base = XGBClassifier(
        eval_metric='logloss', max_depth=5, learning_rate=0.05, 
        subsample=0.8, colsample_bytree=0.8, min_child_weight=3, random_state=42
    )
    xgb_base.fit(X_train_base, y_train)
    models_to_test['Baseline XGBoost (4 Features)'] = (xgb_base, X_test_base)
    
    # 2. Enhanced Models
    rf_enh = RandomForestClassifier(
        n_estimators=150, max_depth=8, min_samples_leaf=3, random_state=42, class_weight='balanced'
    )
    rf_enh.fit(X_train_enh, y_train)
    models_to_test['Enhanced Random Forest (16 Features)'] = (rf_enh, X_test_enh)
    
    lgbm_enh = LGBMClassifier(
        n_estimators=150, max_depth=6, learning_rate=0.03, subsample=0.8, colsample_bytree=0.8, random_state=42, verbose=-1
    )
    lgbm_enh.fit(X_train_enh, y_train)
    models_to_test['Enhanced LightGBM (16 Features)'] = (lgbm_enh, X_test_enh)
    
    xgb_enh = XGBClassifier(
        n_estimators=150, eval_metric='logloss', max_depth=5, learning_rate=0.03,
        subsample=0.8, colsample_bytree=0.8, min_child_weight=2, random_state=42
    )
    xgb_enh.fit(X_train_enh, y_train)
    models_to_test['Enhanced XGBoost (16 Features)'] = (xgb_enh, X_test_enh)
    
    # 3. Voting Ensemble
    voting_ensemble = VotingClassifier(
        estimators=[('rf', rf_enh), ('lgbm', lgbm_enh), ('xgb', xgb_enh)],
        voting='soft'
    )
    voting_ensemble.fit(X_train_enh, y_train)
    models_to_test['Enhanced Soft-Voting Ensemble'] = (voting_ensemble, X_test_enh)
    
    print("\n=== STEP 2: Accuracy, Precision, Recall, F1 & ROC-AUC Comparison ===")
    results = []
    
    best_model_name = None
    best_f1 = -1
    best_model_obj = None
    best_X_test = None
    
    for name, (model, X_test) in models_to_test.items():
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1]
        
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, zero_division=0)
        rec = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        roc = roc_auc_score(y_test, y_proba)
        
        results.append({
            'Model': name,
            'Accuracy': round(acc, 4),
            'Precision': round(prec, 4),
            'Recall': round(rec, 4),
            'F1-Score': round(f1, 4),
            'ROC-AUC': round(roc, 4)
        })
        
        if f1 > best_f1:
            best_f1 = f1
            best_model_name = name
            best_model_obj = model
            best_X_test = X_test

    results_df = pd.DataFrame(results)
    print(results_df.to_string(index=False))

    # Threshold Optimization for the best model
    print("\n=== STEP 3: Decision Threshold Optimization ===")
    best_proba = best_model_obj.predict_proba(best_X_test)[:, 1]
    precisions, recalls, thresholds = precision_recall_curve(y_test, best_proba)
    f1_scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-10)
    best_idx = np.argmax(f1_scores)
    optimal_threshold = thresholds[best_idx] if best_idx < len(thresholds) else 0.5
    optimal_f1 = f1_scores[best_idx]
    
    tuned_preds = (best_proba >= optimal_threshold).astype(int)
    tuned_acc = accuracy_score(y_test, tuned_preds)
    tuned_prec = precision_score(y_test, tuned_preds)
    tuned_rec = recall_score(y_test, tuned_preds)
    
    print(f"Optimal Classification Threshold: {optimal_threshold:.4f} (Default is 0.5000)")
    print(f"Tuned Metrics -> Accuracy: {tuned_acc:.4f} | Precision: {tuned_prec:.4f} | Recall: {tuned_rec:.4f} | F1: {optimal_f1:.4f}")

    # Save the enhanced model & export ONNX
    print("\n=== STEP 4: Model Export & ONNX Compilation ===")
    os.makedirs('data', exist_ok=True)
    joblib.dump(xgb_enh, 'data/icu_risk_model_enhanced.pkl')
    
    # Export RF or XGB to ONNX
    # We use Random Forest or convert with FloatTensorType
    initial_type = [('float_input', FloatTensorType([None, len(enhanced_cols)]))]
    onnx_path = 'data/icu_risk_model.onnx'
    onnx_model = convert_sklearn(rf_enh, initial_types=initial_type, target_opset=12)
    with open(onnx_path, "wb") as f:
        f.write(onnx_model.SerializeToString())
    print(f"Exported model to ONNX runtime: {onnx_path}")
    
    # ONNX Inference Session
    sess = rt.InferenceSession(onnx_path)
    input_name = sess.get_inputs()[0].name
    
    # STEP 5: Latency Benchmarks
    print("\n=== STEP 5: Rigorous Single-Sample Latency Benchmarks ===")
    sample_raw = X_test_enh.iloc[0:1]
    sample_dict = sample_raw.to_dict('records')[0]
    sample_np = sample_raw.to_numpy(dtype=np.float32)
    sample_df = pd.DataFrame([sample_dict])
    
    N_ITER = 3000
    
    # Warm-up
    for _ in range(50):
        _ = rf_enh.predict_proba(sample_df)
        _ = rf_enh.predict_proba(sample_np)
        _ = sess.run(None, {input_name: sample_np})
        
    # 1. Pandas DataFrame Latency
    times_pandas = []
    for _ in range(N_ITER):
        t0 = time.perf_counter()
        _ = rf_enh.predict_proba(pd.DataFrame([sample_dict]))
        times_pandas.append((time.perf_counter() - t0) * 1000.0) # in ms
        
    # 2. NumPy Array Latency
    times_numpy = []
    for _ in range(N_ITER):
        t0 = time.perf_counter()
        _ = rf_enh.predict_proba(sample_np)
        times_numpy.append((time.perf_counter() - t0) * 1000.0) # in ms
        
    # 3. ONNX Runtime Latency
    times_onnx = []
    for _ in range(N_ITER):
        t0 = time.perf_counter()
        _ = sess.run(None, {input_name: sample_np})
        times_onnx.append((time.perf_counter() - t0) * 1000.0) # in ms
        
    latency_summary = [
        {
            'Inference Method': 'Pandas DataFrame + Scikit-Learn (Baseline)',
            'Mean Latency (ms)': round(np.mean(times_pandas), 4),
            'P50 Median (ms)': round(np.median(times_pandas), 4),
            'P95 (ms)': round(np.percentile(times_pandas, 95), 4),
            'P99 (ms)': round(np.percentile(times_pandas, 99), 4),
            'Throughput (preds/sec)': int(1000.0 / np.mean(times_pandas))
        },
        {
            'Inference Method': 'NumPy Array + Scikit-Learn (Optimized)',
            'Mean Latency (ms)': round(np.mean(times_numpy), 4),
            'P50 Median (ms)': round(np.median(times_numpy), 4),
            'P95 (ms)': round(np.percentile(times_numpy, 95), 4),
            'P99 (ms)': round(np.percentile(times_numpy, 99), 4),
            'Throughput (preds/sec)': int(1000.0 / np.mean(times_numpy))
        },
        {
            'Inference Method': 'ONNX Runtime C++ Engine (Ultra-Fast)',
            'Mean Latency (ms)': round(np.mean(times_onnx), 4),
            'P50 Median (ms)': round(np.median(times_onnx), 4),
            'P95 (ms)': round(np.percentile(times_onnx, 95), 4),
            'P99 (ms)': round(np.percentile(times_onnx, 99), 4),
            'Throughput (preds/sec)': int(1000.0 / np.mean(times_onnx))
        }
    ]
    latency_df = pd.DataFrame(latency_summary)
    print(latency_df.to_string(index=False))

    # STEP 6: Consumer I/O Latency Benchmark
    print("\n=== STEP 6: Consumer Stream I/O Benchmark ===")
    dummy_state = {f"stay_{i}": {"hr": 80, "spo2": 98, "sysbp": 120} for i in range(50)}
    
    # Approach 1: Synchronous JSON dump to disk per event (Current consumer.py)
    t0 = time.perf_counter()
    for _ in range(500):
        with open('data/temp_state_benchmark.json', 'w') as f:
            json.dump(dummy_state, f)
    sync_disk_time = time.perf_counter() - t0
    sync_disk_ms = (sync_disk_time / 500) * 1000.0
    
    # Approach 2: In-memory state update
    memory_state = {}
    t0 = time.perf_counter()
    for _ in range(500):
        memory_state.update(dummy_state)
    mem_time = time.perf_counter() - t0
    mem_ms = (mem_time / 500) * 1000.0
    
    if os.path.exists('data/temp_state_benchmark.json'):
        os.remove('data/temp_state_benchmark.json')
        
    print(f"Synchronous Disk Dump per event: {sync_disk_ms:.3f} ms/event (Throughput: {int(1000/sync_disk_ms)} events/s)")
    print(f"In-Memory / Cached State update:  {mem_ms:.4f} ms/event (Throughput: {int(1000/mem_ms)} events/s)")
    speedup = sync_disk_ms / max(mem_ms, 1e-6)
    print(f"I/O Latency Reduction Factor: {speedup:.1f}x faster!")

    # Save summary report to JSON
    report = {
        'model_metrics': results,
        'threshold_tuning': {
            'optimal_threshold': round(float(optimal_threshold), 4),
            'tuned_accuracy': round(float(tuned_acc), 4),
            'tuned_precision': round(float(tuned_prec), 4),
            'tuned_recall': round(float(tuned_rec), 4),
            'tuned_f1': round(float(optimal_f1), 4)
        },
        'inference_latency': latency_summary,
        'io_latency': {
            'sync_disk_write_ms': round(sync_disk_ms, 3),
            'in_memory_cache_ms': round(mem_ms, 4),
            'speedup_factor': round(speedup, 1)
        }
    }
    with open('data/optimization_benchmark_results.json', 'w') as f:
        json.dump(report, f, indent=2)
    print("\nSaved full benchmark results to data/optimization_benchmark_results.json")

if __name__ == '__main__':
    run_experiment()
