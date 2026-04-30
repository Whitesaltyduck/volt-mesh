import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import json
import os

class UsageProfiler:
    def __init__(self, data_path):
        self.data_path = data_path
        self.cluster_map = {}
        self.scaler = StandardScaler()
        self.kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
        
    def load_and_preprocess(self):
        """Load the CSV data and preprocess for clustering"""
        # Read the semicolon-delimited file
        df = pd.read_csv(self.data_path, delimiter=';', low_memory=False)
        
        # Handle missing values represented by '?'
        df.replace('?', np.nan, inplace=True)
        
        # Convert columns to numeric, errors='coerce' will turn problematic values to NaN
        numeric_columns = ['Global_active_power', 'Global_reactive_power', 'Voltage', 
                          'Global_intensity', 'Sub_metering_1', 'Sub_metering_2', 'Sub_metering_3']
        for col in numeric_columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # For simplicity, we'll use Global_active_power as our primary usage metric
        # and extract hour from the time column
        df['Time'] = pd.to_datetime(df['Time'], format='%H:%M:%S', errors='coerce')
        df['Hour'] = df['Time'].dt.hour
        
        # Drop rows where essential data is missing
        df_clean = df.dropna(subset=['Global_active_power', 'Hour']).copy()
        
        return df_clean
    
    def create_24h_usage_profile(self, df):
        """Create average usage profile for each hour of the day"""
        hourly_avg = df.groupby('Hour')['Global_active_power'].mean().reset_index()
        hourly_avg.columns = ['Hour', 'Avg_Usage']
        
        # Ensure we have all 24 hours (fill missing hours with 0)
        all_hours = pd.DataFrame({'Hour': range(24)})
        hourly_complete = all_hours.merge(hourly_avg, on='Hour', how='left').fillna(0)
        
        return hourly_complete
    
    def cluster_usage_zones(self):
        """Perform K-Means clustering to identify Eco, Moderate, and Peak zones"""
        # Load and preprocess data
        df = self.load_and_preprocess()
        
        # Create 24-hour usage profile
        usage_profile = self.create_24h_usage_profile(df)
        
        # Prepare features for clustering (using normalized usage)
        X = usage_profile[['Avg_Usage']].values
        X_scaled = self.scaler.fit_transform(X)
        
        # Perform K-Means clustering
        clusters = self.kmeans.fit_predict(X_scaled)
        
        # Create cluster map
        usage_profile['Cluster'] = clusters
        usage_profile['Zone'] = usage_profile['Cluster'].map({
            0: 'Eco',      # Lowest usage cluster
            1: 'Moderate', # Middle usage cluster
            2: 'Peak'      # Highest usage cluster
        })
        
        # Reorder clusters to ensure Eco < Moderate < Peak based on usage levels
        cluster_centers = self.kmeans.cluster_centers_.flatten()
        cluster_order = np.argsort(cluster_centers)  # Order from lowest to highest usage
        
        # Remap cluster labels
        remap_dict = {cluster_order[i]: i for i in range(len(cluster_order))}
        usage_profile['Cluster'] = usage_profile['Cluster'].map(remap_dict)
        usage_profile['Zone'] = usage_profile['Cluster'].map({
            0: 'Eco',
            1: 'Moderate',
            2: 'Peak'
        })
        
        # Save cluster map for other agents to reference
        self.cluster_map = usage_profile.set_index('Hour')['Zone'].to_dict()
        
        # Save to file for persistence
        with open(r'C:\volt_mesh_output\cluster_map.json', 'w') as f:
            json.dump(self.cluster_map, f)
            
        return self.cluster_map
    
    def get_zone_for_hour(self, hour):
        """Get the usage zone for a specific hour"""
        return self.cluster_map.get(hour % 24, 'Moderate')  # Default to Moderate if not found

if __name__ == "__main__":
    # Initialize profiler with the data file
    profiler = UsageProfiler(r"C:\data\power_usage.csv")
    
    # Run clustering and save results
    cluster_map = profiler.cluster_usage_zones()
    
    print("Usage Profiler ('Clusternaut') completed.")
    print(f"Cluster map saved with {len(cluster_map)} hourly zones:")
    for hour, zone in sorted(cluster_map.items()):
        print(f"  Hour {hour:2d}: {zone}")