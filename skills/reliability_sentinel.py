import pandas as pd
import numpy as np
from sklearn.naive_bayes import GaussianNB
import joblib
import os

def train_reliability_sentinel():
    """
    Train the Reliability Sentinel agent using Naive Bayes to predict overload probability.
    Saves the trained model for later use.
    """
    # Load the data
    data_path = r'C:\data\power_usage.csv'
    df = pd.read_csv(data_path, sep=';', na_values='?')
    
    # Combine Date and Time into a datetime column
    df['datetime'] = pd.to_datetime(df['Date'] + ' ' + df['Time'], format='%d/%m/%Y %H:%M:%S')
    
    # Set datetime as index
    df.set_index('datetime', inplace=True)
    
    # Feature: Global_active_power (current load)
    # Target: Whether the power consumption indicates potential overload
    # We'll define overload as when power consumption is above the 95th percentile
    threshold = df['Global_active_power'].quantile(0.95)
    df['overload'] = (df['Global_active_power'] > threshold).astype(int)
    
    # Drop rows with missing values
    df_clean = df[['Global_active_power', 'Voltage', 'Global_intensity', 'overload']].dropna()
    
    # Features and target - using multivariate features
    X = df_clean[['Global_active_power', 'Voltage', 'Global_intensity']].values
    y = df_clean['overload'].values
    
    # Train Naive Bayes classifier
    nb_classifier = GaussianNB()
    nb_classifier.fit(X, y)
    
    # Save the model
    skills_dir = r'C:\.opencode\skills'
    os.makedirs(skills_dir, exist_ok=True)
    model_path = os.path.join(skills_dir, 'reliability_sentinel_model.joblib')
    joblib.dump(nb_classifier, model_path)
    
    print(f"Reliability Sentinel trained. Model saved to {model_path}")
    print(f"Overload threshold (95th percentile): {threshold:.2f} kW")
    
    # Test the model with some example values (using average values for Voltage and Intensity)
    avg_voltage = df_clean['Voltage'].mean()
    avg_intensity = df_clean['Global_intensity'].mean()
    test_loads = [0.5, 1.0, 2.0, 3.0, 4.0, 5.0]
    print("\nTesting overload probability for different loads (with avg Voltage & Intensity):")
    for load in test_loads:
        features = [[load, avg_voltage, avg_intensity]]
        prob = nb_classifier.predict_proba(features)[0][1]  # Probability of overload (class 1)
        print(f"  Load {load:4.1f} kW -> P(Overload) = {prob:.4f}")
        if prob > 0.75:
            print(f"    -> TRIGGER LOAD BALANCE REQUEST (prob > 0.75)")
    
    return nb_classifier

def predict_overload_probability(current_load, voltage=None, intensity=None):
    """
    Predict P(Overload | Current_Load, Voltage, Intensity) using the trained model.
    Returns True if probability > 0.75 (should trigger load balance request).
    If voltage or intensity not provided, uses average values from training data.
    """
    skills_dir = r'C:\.opencode\skills'
    model_path = os.path.join(skills_dir, 'reliability_sentinel_model.joblib')
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found at {model_path}. Please train first.")
    
    nb_classifier = joblib.load(model_path)
    
    # If voltage/intensity not provided, we need to compute averages from data
    # For simplicity in this function, we'll load the data to get averages
    # In a production system, these would be stored with the model
    if voltage is None or intensity is None:
        data_path = r'C:\data\power_usage.csv'
        df = pd.read_csv(data_path, sep=';', na_values='?')
        df['datetime'] = pd.to_datetime(df['Date'] + ' ' + df['Time'], format='%d/%m/%Y %H:%M:%S')
        df.set_index('datetime', inplace=True)
        threshold = df['Global_active_power'].quantile(0.95)
        df['overload'] = (df['Global_active_power'] > threshold).astype(int)
        df_clean = df[['Global_active_power', 'Voltage', 'Global_intensity', 'overload']].dropna()
        avg_voltage = df_clean['Voltage'].mean()
        avg_intensity = df_clean['Global_intensity'].mean()
        
        voltage = voltage if voltage is not None else avg_voltage
        intensity = intensity if intensity is not None else avg_intensity
    
    features = [[current_load, voltage, intensity]]
    prob = nb_classifier.predict_proba(features)[0][1]  # Probability of overload
    
    should_trigger = prob > 0.75
    return prob, should_trigger

if __name__ == "__main__":
    train_reliability_sentinel()