import pandas as pd
import numpy as np
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import StandardScaler
import json
import os

class ReliabilitySentinel:
    def __init__(self, data_path):
        self.data_path = data_path
        self.model = GaussianNB()
        self.scaler = StandardScaler()
        self.overload_threshold = 0.75  # Probability threshold for triggering load balance request
        self.feature_columns = ['Global_active_power', 'Global_reactive_power', 'Voltage', 'Global_intensity']
        
    def load_and_preprocess(self):
        """Load the CSV data and preprocess for classification"""
        # Read the semicolon-delimited file
        df = pd.read_csv(self.data_path, delimiter=';', low_memory=False)
        
        # Handle missing values represented by '?'
        df.replace('?', np.nan, inplace=True)
        
        # Convert columns to numeric
        for col in self.feature_columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # Drop rows where essential data is missing
        df_clean = df.dropna(subset=self.feature_columns).copy()
        
        return df_clean
    
    def create_overload_labels(self, df):
        """Create overload labels based on Global_active_power exceeding a threshold"""
        # Define overload as power consumption above the 95th percentile
        power_threshold = df['Global_active_power'].quantile(0.95)
        df['Overload'] = (df['Global_active_power'] > power_threshold).astype(int)
        return df
    
    def train_model(self):
        """Train the Naive Bayes classifier"""
        # Load and preprocess data
        df = self.load_and_preprocess()
        
        # Create overload labels
        df = self.create_overload_labels(df)
        
        # Prepare features and target
        X = df[self.feature_columns].values
        y = df['Overload'].values
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        # Train the model
        self.model.fit(X_scaled, y)
        
        print(f"Reliability Sentinel (The Predictor) trained on {len(df)} samples")
        print(f"Overload threshold (95th percentile): {df['Global_active_power'].quantile(0.95):.2f} kW")
        
    def predict_overload_probability(self, current_load_data):
        """Predict probability of overload given current load measurements"""
        # Ensure data is in the right format
        if isinstance(current_load_data, dict):
            # Convert single measurement to DataFrame
            df_input = pd.DataFrame([current_load_data])
        else:
            df_input = current_load_data.copy()
        
        # Handle missing values
        df_input.replace('?', np.nan, inplace=True)
        
        # Convert to numeric
        for col in self.feature_columns:
            if col in df_input.columns:
                df_input[col] = pd.to_numeric(df_input[col], errors='coerce')
        
        # Fill missing values with column means (for simplicity)
        for col in self.feature_columns:
            if col in df_input.columns:
                df_input[col].fillna(df_input[col].mean(), inplace=True)
        
        # Prepare features
        X = df_input[self.feature_columns].values
        
        # Scale features
        X_scaled = self.scaler.transform(X)
        
        # Get probability of overload (class 1)
        probabilities = self.model.predict_proba(X_scaled)
        overload_prob = probabilities[:, 1]  # Probability of class 1 (overload)
        
        return overload_prob
    
    def should_trigger_load_balance(self, current_load_data):
        """Check if overload probability exceeds threshold and trigger load balance request"""
        overload_prob = self.predict_overload_probability(current_load_data)
        
        # For single prediction, return boolean; for multiple, return array
        if isinstance(overload_prob, np.ndarray) and len(overload_prob) == 1:
            overload_prob = overload_prob[0]
        
        should_trigger = overload_prob > self.overload_threshold
        
        if should_trigger:
            print(f"WARNING: Reliability Sentinel: Overload probability {overload_prob:.3f} > {self.overload_threshold}")
            print("   Triggering Load Balance Request to The Searcher (Load Optimizer)")
            
            # Save request for the Optimizer to pick up
            with open(r'C:\volt_mesh_output\load_balance_request.json', 'w') as f:
                json.dump({
                    'triggered': True,
                    'overload_probability': float(overload_prob),
                    'timestamp': pd.Timestamp.now().isoformat(),
                    'current_load': current_load_data
                }, f)
        
        return should_trigger, overload_prob

if __name__ == "__main__":
    # Initialize sentinel with the data file
    sentinel = ReliabilitySentinel(r"C:\data\power_usage.csv")
    
    # Train the model
    sentinel.train_model()
    
    # Example usage: predict overload for a high voltage spike scenario
    print("\n--- Testing with High Voltage Spike Scenario ---")
    high_voltage_spike = {
        'Global_active_power': 6.0,  # High power consumption
        'Global_reactive_power': 0.5,
        'Voltage': 250.0,           # High voltage spike
        'Global_intensity': 24.0
    }
    
    should_trigger, prob = sentinel.should_trigger_load_balance(high_voltage_spike)
    print(f"Overload probability: {prob:.3f}")
    print(f"Load balance requested: {should_trigger}")