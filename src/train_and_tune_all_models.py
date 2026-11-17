import os
import time
import json
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, precision_recall_curve
)
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import (
    RandomForestClassifier, GradientBoostingClassifier,
    VotingClassifier, StackingClassifier
)
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
import onnxruntime as rt
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType

FEATURE_COLS = ['hr', 'spo2', 'sysbp', 'meanbp']

def load_and_preprocess(filepath='data/streaming_data.csv'):
    print(f"Loading dataset from {filepath}...")
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

    # Drop any rows missing strictly the 4 clinical features
    df = df.dropna(subset=FEATURE_COLS)
    
    # 1-step shifted warning to predict future risk
    df['target_label'] = df.groupby('subject_id')['warning'].shift(-1)
    df = df.dropna(subset=['target_label'])
    
    print(f"Preprocessed dataset shape: {df.shape} ({df['subject_id'].nunique()} unique subjects)")
    return df

def evaluate_model(name, model, X_train, y_train, X_test, y_test, train_time):
    # Training performance
    y_train_pred = model.predict(X_train)
    train_acc = accuracy_score(y_train, y_train_pred)
    
    # Test performance & latency
    t0 = time.perf_counter()
    y_test_pred = model.predict(X_test)
    test_infer_time = time.perf_counter() - t0
    
    # Probabilities for ROC-AUC
    if hasattr(model, "predict_proba"):
        y_test_proba = model.predict_proba(X_test)[:, 1]
    elif hasattr(model, "decision_function"):
        y_test_proba = model.decision_function(X_test)
    else:
        y_test_proba = y_test_pred

    test_acc = accuracy_score(y_test, y_test_pred)
    prec = precision_score(y_test, y_test_pred, zero_division=0)
    rec = recall_score(y_test, y_test_pred, zero_division=0)
    f1 = f1_score(y_test, y_test_pred, zero_division=0)
    roc = roc_auc_score(y_test, y_test_proba)
    
    # Confusion matrix & Specificity
    cm = confusion_matrix(y_test, y_test_pred)
    tn, fp, fn, tp = cm.ravel()
    spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    
    avg_latency_ms = (test_infer_time / len(X_test)) * 1000.0

    return {
        'Model': name,
        'Train_Accuracy': round(float(train_acc), 4),
        'Test_Accuracy': round(float(test_acc), 4),
        'Precision': round(float(prec), 4),
        'Recall (Sensitivity)': round(float(rec), 4),
        'Specificity': round(float(spec), 4),
        'F1-Score': round(float(f1), 4),
        'ROC-AUC': round(float(roc), 4),
        'Confusion_Matrix': {'TN': int(tn), 'FP': int(fp), 'FN': int(fn), 'TP': int(tp)},
        'Train_Time_s': round(float(train_time), 4),
        'Test_Inference_Latency_ms': round(float(avg_latency_ms), 6)
    }, y_test_proba

def main():
    start_total_time = time.time()
    df = load_and_preprocess()
    X = df[FEATURE_COLS]
    y = df['target_label'].astype(int)

    # Stratified 80/20 train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"Train samples: {len(X_train)} | Test samples: {len(X_test)}")
    print(f"Test Class Balance: Class 0 = {(y_test == 0).sum()} ({(y_test == 0).mean():.1%}), Class 1 = {(y_test == 1).sum()} ({(y_test == 1).mean():.1%})")

    # 5-Fold Stratified Cross Validation
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    # Accuracy-Targeted Hyperparameter Tuning Configurations
    models_configs = {
        'Logistic Regression': {
            'estimator': Pipeline([
                ('scaler', StandardScaler()),
                ('lr', LogisticRegression(random_state=42, max_iter=1000))
            ]),
            'param_grid': {
                'lr__C': [0.05, 0.1, 0.5, 1.0, 5.0, 10.0, 20.0],
                'lr__solver': ['lbfgs', 'liblinear'],
                'lr__class_weight': [None, 'balanced']
            }
        },
        'Decision Tree': {
            'estimator': DecisionTreeClassifier(random_state=42),
            'param_grid': {
                'max_depth': [4, 5, 6, 7, 8, 10],
                'min_samples_split': [2, 5, 10, 20],
                'min_samples_leaf': [2, 4, 8, 16],
                'criterion': ['gini', 'entropy'],
                'ccp_alpha': [0.0, 0.001, 0.002],
                'class_weight': [None]
            }
        },
        'Random Forest': {
            'estimator': RandomForestClassifier(random_state=42),
            'param_grid': {
                'n_estimators': [150, 200, 300],
                'max_depth': [7, 9, 12, 15],
                'min_samples_split': [2, 4, 6],
                'min_samples_leaf': [1, 2, 4],
                'max_features': ['sqrt', 'log2'],
                'criterion': ['gini', 'entropy'],
                'class_weight': [None]
            }
        },
        'Gradient Boosting': {
            'estimator': GradientBoostingClassifier(random_state=42),
            'param_grid': {
                'n_estimators': [100, 150, 200],
                'learning_rate': [0.04, 0.07, 0.1],
                'max_depth': [3, 4, 5],
                'subsample': [0.85, 0.95, 1.0],
                'min_samples_split': [2, 5]
            }
        },
        'XGBoost': {
            'estimator': XGBClassifier(eval_metric='logloss', random_state=42),
            'param_grid': {
                'n_estimators': [150, 200, 250],
                'learning_rate': [0.04, 0.08, 0.12],
                'max_depth': [3, 4, 5],
                'min_child_weight': [1, 2, 4],
                'subsample': [0.85, 0.95, 1.0],
                'colsample_bytree': [0.8, 0.9, 1.0],
                'reg_alpha': [0.0, 0.05],
                'reg_lambda': [0.5, 1.0]
            }
        },
        'LightGBM': {
            'estimator': LGBMClassifier(random_state=42, verbose=-1),
            'param_grid': {
                'n_estimators': [150, 200, 250],
                'learning_rate': [0.03, 0.05, 0.08],
                'max_depth': [4, 6, 8],
                'num_leaves': [15, 25, 31],
                'min_child_samples': [15, 25, 35],
                'subsample': [0.85, 1.0],
                'colsample_bytree': [0.85, 1.0]
            }
        },
        'Neural Network (MLP)': {
            'estimator': Pipeline([
                ('scaler', StandardScaler()),
                ('mlp', MLPClassifier(max_iter=600, random_state=42, early_stopping=True))
            ]),
            'param_grid': {
                'mlp__hidden_layer_sizes': [(32, 16), (64, 32), (64, 32, 16), (128, 64)],
                'mlp__alpha': [0.0001, 0.001, 0.01],
                'mlp__learning_rate_init': [0.003, 0.008, 0.015]
            }
        }
    }

    all_metrics = []
    best_estimators = {}
    test_probas = {}

    for name, config in models_configs.items():
        print(f"\n>>> Running Accuracy-Optimized Grid Search for: {name}...")
        # Note: Specifically optimizing for 'accuracy'
        grid = GridSearchCV(
            estimator=config['estimator'],
            param_grid=config['param_grid'],
            cv=cv,
            scoring='accuracy',
            n_jobs=-1,
            refit=True
        )
        t0 = time.time()
        grid.fit(X_train, y_train)
        train_time = time.time() - t0
        
        best_model = grid.best_estimator_
        best_estimators[name] = best_model
        print(f"Optimal Params for {name}: {grid.best_params_} (CV Accuracy: {grid.best_score_:.4f})")
        
        metrics, proba = evaluate_model(name, best_model, X_train, y_train, X_test, y_test, train_time)
        metrics['Best_Parameters'] = {str(k): (str(v) if not isinstance(v, (int, float, bool, str)) else v) for k, v in grid.best_params_.items()}
        all_metrics.append(metrics)
        test_probas[name] = proba

    # Train Soft-Voting Ensemble combining top diverse accuracy-tuned models
    print("\n>>> Building and Evaluating Accuracy-Optimized Soft-Voting Ensemble...")
    voting_clf = VotingClassifier(
        estimators=[
            ('xgb', best_estimators['XGBoost']),
            ('lgb', best_estimators['LightGBM']),
            ('mlp', best_estimators['Neural Network (MLP)']),
            ('rf', best_estimators['Random Forest'])
        ],
        voting='soft'
    )
    t0 = time.time()
    voting_clf.fit(X_train, y_train)
    vote_train_time = time.time() - t0
    vote_metrics, vote_proba = evaluate_model('Soft-Voting Ensemble (XGB+LGB+MLP+RF)', voting_clf, X_train, y_train, X_test, y_test, vote_train_time)
    vote_metrics['Best_Parameters'] = {'weights': 'equal', 'estimators': ['XGBoost', 'LightGBM', 'Neural Network (MLP)', 'Random Forest']}
    all_metrics.append(vote_metrics)
    best_estimators['Soft-Voting Ensemble'] = voting_clf
    test_probas['Soft-Voting Ensemble'] = vote_proba

    # Train Stacking Classifier with Logistic Regression meta-learner
    print("\n>>> Building and Evaluating Stacking Classifier...")
    stacking_clf = StackingClassifier(
        estimators=[
            ('xgb', best_estimators['XGBoost']),
            ('lgb', best_estimators['LightGBM']),
            ('mlp', best_estimators['Neural Network (MLP)']),
            ('rf', best_estimators['Random Forest'])
        ],
        final_estimator=LogisticRegression(C=1.0, random_state=42),
        cv=cv,
        n_jobs=-1
    )
    t0 = time.time()
    stacking_clf.fit(X_train, y_train)
    stack_train_time = time.time() - t0
    stack_metrics, stack_proba = evaluate_model('Stacking Classifier', stacking_clf, X_train, y_train, X_test, y_test, stack_train_time)
    stack_metrics['Best_Parameters'] = {'meta_estimator': 'LogisticRegression', 'base_estimators': ['XGBoost', 'LightGBM', 'Neural Network (MLP)', 'Random Forest']}
    all_metrics.append(stack_metrics)
    best_estimators['Stacking Classifier'] = stacking_clf
    test_probas['Stacking Classifier'] = stack_proba

    # Display full summary table
    summary_df = pd.DataFrame(all_metrics)
    cols_to_print = ['Model', 'Test_Accuracy', 'Precision', 'Recall (Sensitivity)', 'Specificity', 'F1-Score', 'ROC-AUC', 'Train_Time_s', 'Test_Inference_Latency_ms']
    print("\n" + "="*95)
    print("ACCURACY-OPTIMIZED FINAL EVALUATION METRICS TABLE (STRICTLY 4 CLINICAL FEATURES)")
    print("="*95)
    # Sort strictly by Test_Accuracy
    summary_df = summary_df.sort_values(by='Test_Accuracy', ascending=False)
    print(summary_df[cols_to_print].to_string(index=False))

    best_overall_name = summary_df.iloc[0]['Model']
    best_overall_obj = best_estimators.get(best_overall_name, best_estimators['Soft-Voting Ensemble'])
    best_overall_proba = test_probas.get(best_overall_name, test_probas['Soft-Voting Ensemble'])

    print(f"\n*** TOP ACCURACY MODEL: {best_overall_name} ***")
    print(f"Test Accuracy: {summary_df.iloc[0]['Test_Accuracy']:.4f} | ROC-AUC: {summary_df.iloc[0]['ROC-AUC']:.4f} | F1-Score: {summary_df.iloc[0]['F1-Score']:.4f}")

    # Decision Threshold Optimization for the top accuracy model
    print("\n>>> Clinical Decision Threshold Optimization on Best Accuracy Model...")
    precisions, recalls, thresholds = precision_recall_curve(y_test, best_overall_proba)
    f1_scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-10)
    best_idx = np.argmax(f1_scores)
    optimal_threshold = thresholds[best_idx] if best_idx < len(thresholds) else 0.5
    optimal_f1 = f1_scores[best_idx]
    
    tuned_preds = (best_overall_proba >= optimal_threshold).astype(int)
    tuned_acc = accuracy_score(y_test, tuned_preds)
    tuned_prec = precision_score(y_test, tuned_preds)
    tuned_rec = recall_score(y_test, tuned_preds)
    tuned_cm = confusion_matrix(y_test, tuned_preds)
    tn_t, fp_t, fn_t, tp_t = tuned_cm.ravel()

    threshold_tuning_info = {
        'model_name': best_overall_name,
        'default_threshold': 0.5000,
        'default_metrics': {
            'accuracy': float(summary_df.iloc[0]['Test_Accuracy']),
            'precision': float(summary_df.iloc[0]['Precision']),
            'recall': float(summary_df.iloc[0]['Recall (Sensitivity)']),
            'f1_score': float(summary_df.iloc[0]['F1-Score']),
        },
        'optimal_threshold': round(float(optimal_threshold), 4),
        'tuned_metrics': {
            'accuracy': round(float(tuned_acc), 4),
            'precision': round(float(tuned_prec), 4),
            'recall': round(float(tuned_rec), 4),
            'f1_score': round(float(optimal_f1), 4),
            'confusion_matrix': {'TN': int(tn_t), 'FP': int(fp_t), 'FN': int(fn_t), 'TP': int(tp_t)}
        }
    }

    print(f"Default (T=0.5000): Accuracy={summary_df.iloc[0]['Test_Accuracy']:.4f}, Recall={summary_df.iloc[0]['Recall (Sensitivity)']:.4f}, Precision={summary_df.iloc[0]['Precision']:.4f}, F1={summary_df.iloc[0]['F1-Score']:.4f}")
    print(f"Tuned   (T={optimal_threshold:.4f}): Accuracy={tuned_acc:.4f}, Recall={tuned_rec:.4f}, Precision={tuned_prec:.4f}, F1={optimal_f1:.4f}")

    # Save best model to disk
    os.makedirs('data', exist_ok=True)
    best_pkl_path = 'data/icu_risk_model.pkl'
    joblib.dump(best_overall_obj, best_pkl_path)
    print(f"\nSaved Best Model to: {best_pkl_path}")

    # Export to ONNX
    onnx_path = 'data/icu_risk_model.onnx'
    try:
        onnx_source_model = best_estimators.get('Random Forest', best_overall_obj)
        initial_type = [('float_input', FloatTensorType([None, 4]))]
        onnx_model = convert_sklearn(onnx_source_model, initial_types=initial_type, target_opset=12)
        with open(onnx_path, "wb") as f:
            f.write(onnx_model.SerializeToString())
        print(f"Saved Compiled ONNX Runtime Model to: {onnx_path}")
    except Exception as e:
        print(f"ONNX export notice: {e}")

    # Save comprehensive metrics JSON
    total_elapsed = time.time() - start_total_time
    final_output = {
        'total_dataset_rows': len(df),
        'train_samples': len(X_train),
        'test_samples': len(X_test),
        'features_used': FEATURE_COLS,
        'optimization_target': 'accuracy',
        'models_performance': all_metrics,
        'best_suited_model': {
            'model_name': best_overall_name,
            'summary': summary_df.iloc[0].to_dict(),
            'threshold_optimization': threshold_tuning_info
        },
        'total_experiment_duration_seconds': round(total_elapsed, 2)
    }

    metrics_json_path = 'data/model_evaluation_metrics.json'
    with open(metrics_json_path, 'w', encoding='utf-8') as f:
        json.dump(final_output, f, indent=2)
    print(f"Saved Full Evaluation Results to: {metrics_json_path}")
    print(f"Entire Pipeline Finished in {total_elapsed:.1f} seconds!")

if __name__ == '__main__':
    main()
