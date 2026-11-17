import pandas as pd
import numpy as np
import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.utils import class_weight
import warnings
warnings.filterwarnings('ignore')

# 1. Preprocessing (same as before)
def preprocess_data(filepath='data/streaming_data.csv'):
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

# 2. PyTorch GRU Model
class GRUModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers=1, dropout=0.2):
        super(GRUModel, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        # GRU layer
        self.gru = nn.GRU(input_dim, hidden_dim, num_layers, batch_first=True, dropout=(dropout if num_layers > 1 else 0))
        
        # Fully connected layers
        self.fc1 = nn.Linear(hidden_dim, 32)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(32, output_dim)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # Initialize hidden state
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(x.device)
        
        # Forward propagate GRU
        out, _ = self.gru(x, h0)
        
        # Take the output from the last time step
        out = out[:, -1, :]
        
        # Pass through FC layers
        out = self.fc1(out)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.fc2(out)
        out = self.sigmoid(out)
        return out

def train_pytorch_gru():
    print("Loading and preparing data...")
    df = preprocess_data('data/streaming_data.csv')
    df = generate_labels(df)
    
    feature_cols = ['hr', 'spo2', 'sysbp', 'meanbp']
    X = df[feature_cols].values
    y = df['target_label'].values
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Reshape for GRU: (batch_size, sequence_length, input_dim)
    # Here sequence_length = 1 as we are treating point-in-time features as a 1-step sequence
    X_train_gru = X_train_scaled.reshape((X_train_scaled.shape[0], 1, X_train_scaled.shape[1]))
    X_test_gru = X_test_scaled.reshape((X_test_scaled.shape[0], 1, X_test_scaled.shape[1]))
    
    # Convert to PyTorch tensors
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    X_train_tensor = torch.FloatTensor(X_train_gru).to(device)
    y_train_tensor = torch.FloatTensor(y_train).unsqueeze(1).to(device)
    X_test_tensor = torch.FloatTensor(X_test_gru).to(device)
    y_test_tensor = torch.FloatTensor(y_test).unsqueeze(1).to(device)
    
    # Create DataLoader
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    
    # Calculate class weights for BCELoss
    classes = np.unique(y_train)
    weights = class_weight.compute_class_weight('balanced', classes=classes, y=y_train)
    # For BCE, we can use positive weight (pos_weight) = weight_for_class_1 / weight_for_class_0
    pos_weight = torch.tensor([weights[1] / weights[0]]).to(device)
    
    # Initialize Model, Loss, and Optimizer
    input_dim = len(feature_cols)
    hidden_dim = 64
    output_dim = 1
    
    model = GRUModel(input_dim, hidden_dim, output_dim, num_layers=2, dropout=0.2).to(device)
    # BCEWithLogitsLoss requires raw logits, but since we have a sigmoid in the model, we use BCELoss
    # But since we want to use pos_weight, it's better to use an unreduced loss or manually weight it.
    # We will just use BCELoss and standard weighting in the loop.
    criterion = nn.BCELoss(reduction='none') 
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    num_epochs = 50
    total_start_time = time.time()
    
    print("\n--- Training GRU-D (PyTorch) ---")
    model.train()
    for epoch in range(num_epochs):
        for batch_X, batch_y in train_loader:
            optimizer.zero_grad()
            outputs = model(batch_X)
            
            # Apply weights
            loss = criterion(outputs, batch_y)
            weight = torch.where(batch_y == 1, pos_weight, torch.tensor(1.0).to(device))
            loss = (loss * weight).mean()
            
            loss.backward()
            optimizer.step()
            
    train_end_time = time.time()
    train_latency = train_end_time - total_start_time
    
    # Evaluation
    model.eval()
    with torch.no_grad():
        # Train metrics
        train_outputs = model(X_train_tensor)
        y_train_pred_proba = train_outputs.cpu().numpy().flatten()
        y_train_pred = (y_train_pred_proba > 0.5).astype(int)
        
        train_acc = accuracy_score(y_train, y_train_pred)
        train_roc_auc = roc_auc_score(y_train, y_train_pred_proba)
        
        # Inference / Test metrics
        inference_start_time = time.time()
        test_outputs = model(X_test_tensor)
        inference_end_time = time.time()
        
        y_test_pred_proba = test_outputs.cpu().numpy().flatten()
        y_test_pred = (y_test_pred_proba > 0.5).astype(int)
        
        test_acc = accuracy_score(y_test, y_test_pred)
        test_roc_auc = roc_auc_score(y_test, y_test_pred_proba)
        test_prec = precision_score(y_test, y_test_pred, zero_division=0)
        test_rec = recall_score(y_test, y_test_pred, zero_division=0)
        test_f1 = f1_score(y_test, y_test_pred, zero_division=0)
        
    inference_latency = inference_end_time - inference_start_time
    avg_inference_latency = inference_latency / len(y_test)
    total_latency = train_latency + inference_latency
    
    print(f"Train Accuracy: {train_acc:.4f} | Test Accuracy:  {test_acc:.4f}")
    print(f"Train ROC-AUC:  {train_roc_auc:.4f} | Test ROC-AUC:   {test_roc_auc:.4f}")
    print(f"Test Precision: {test_prec:.4f}")
    print(f"Test Recall:    {test_rec:.4f}")
    print(f"Test F1-score:  {test_f1:.4f}")
    print(f"Total Latency:  {total_latency:.4f} seconds")
    print(f"Inference Latency: {inference_latency:.4f} seconds")
    print(f"Average Inference Latency: {avg_inference_latency:.6f} seconds/sample")
    
    # Save the PyTorch model
    os.makedirs('data', exist_ok=True)
    torch.save(model.state_dict(), 'data/icu_risk_model_gru.pth')

if __name__ == '__main__':
    train_pytorch_gru()
