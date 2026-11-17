import numpy as np
import pandas as pd
import joblib
import os

FEATURE_COLS = ['hr', 'spo2', 'sysbp', 'meanbp']

class RiskPredictor:
    def __init__(self, model_path='data/icu_risk_model.pkl', onnx_path='data/icu_risk_model.onnx'):
        self.model_path = model_path
        self.onnx_path = onnx_path
        self.model = None
        self.onnx_session = None
        self.input_name = None

        # Try loading high-performance ONNX model first (< 0.03ms latency)
        if os.path.exists(self.onnx_path):
            try:
                import onnxruntime as rt
                self.onnx_session = rt.InferenceSession(self.onnx_path)
                self.input_name = self.onnx_session.get_inputs()[0].name
                print(f"High-performance ONNX model loaded from {self.onnx_path}")
            except Exception as e:
                print(f"Failed to load ONNX model: {e}")

        # Fallback to Scikit-Learn / Joblib model
        if self.onnx_session is None and os.path.exists(self.model_path):
            try:
                self.model = joblib.load(self.model_path)
                print(f"ML model loaded from {self.model_path}")
            except Exception as e:
                print(f"Failed to load PKL model: {e}. Using heuristic fallback.")

        if self.onnx_session is None and self.model is None:
            print(f"No ML model available. Using clinical heuristic fallback.")

    def predict_risk(self, heart_rate=None, spo2=None, systolic_bp=None, mean_bp=None, state=None):
        """
        Takes either state dictionary or individual vital values.
        Returns risk probability from 0.0 to 1.0.
        """
        # Handle state dictionary passed from consumer
        if isinstance(heart_rate, dict):
            state = heart_rate
            heart_rate = None

        if state is not None and isinstance(state, dict):
            v = state.get('vitals', {})
            heart_rate = v.get('hr', 75.0) or 75.0
            spo2 = v.get('spo2', 98.0) or 98.0
            systolic_bp = v.get('sysbp', 120.0) or 120.0
            mean_bp = v.get('meanbp', 85.0) or 85.0

        heart_rate = float(heart_rate or 75.0)
        spo2 = float(spo2 or 98.0)
        systolic_bp = float(systolic_bp or 120.0)
        mean_bp = float(mean_bp or 85.0)

        # 1. High-Performance ONNX C++ Execution (~0.024ms)
        if self.onnx_session is not None:
            try:
                features_4 = np.array([[heart_rate, spo2, systolic_bp, mean_bp]], dtype=np.float32)
                outputs = self.onnx_session.run(None, {self.input_name: features_4})
                # outputs[1] is probabilities dict/array in ONNX classifier
                if len(outputs) > 1 and isinstance(outputs[1], list):
                    prob = float(outputs[1][0].get(1, outputs[1][0].get(1.0, 0.5)))
                elif len(outputs) > 1 and hasattr(outputs[1], 'shape'):
                    prob = float(outputs[1][0, 1])
                else:
                    prob = float(outputs[0][0])
                return round(prob, 3)
            except Exception as e:
                pass

        # 2. Scikit-Learn Model Execution
        if self.model is not None:
            try:
                # Use fast NumPy array instead of slow DataFrame
                arr = np.array([[heart_rate, spo2, systolic_bp, mean_bp]], dtype=np.float32)
                if hasattr(self.model, "predict_proba"):
                    prob = float(self.model.predict_proba(arr)[0][1])
                else:
                    prob = float(self.model.predict(arr)[0])
                return round(prob, 3)
            except Exception:
                try:
                    df = pd.DataFrame([[heart_rate, spo2, systolic_bp, mean_bp]], columns=FEATURE_COLS)
                    prob = float(self.model.predict_proba(df)[0][1])
                    return round(prob, 3)
                except Exception:
                    pass

        # 3. Clinical Heuristic Fallback
        risk = 0.05
        if heart_rate > 110: risk += 0.20
        if heart_rate > 130: risk += 0.20
        if heart_rate < 50:  risk += 0.30
        if systolic_bp < 100: risk += 0.20
        if systolic_bp < 90:  risk += 0.30
        if mean_bp < 65: risk += 0.20
        if spo2 < 92: risk += 0.20
        if spo2 < 85: risk += 0.20

        return round(min(0.99, risk), 3)