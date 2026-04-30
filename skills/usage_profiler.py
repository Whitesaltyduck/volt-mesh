import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
import json
import os

def train_usage_profiler():
    """
    Train the Usage Profiler agent using K-Means clustering on hourly power usage.
    Saves a cluster map mapping hour of day (0-23) to usage zone: 'Eco', 'Moderate', 'Peak'.
    """
    # Load the data
    data_path = r'C:\data\power_usage.csv'
    df = pd.read_csv(data_path, sep=';', na_values='?')
    
    # Combine Date and Time into a datetime column
    df['datetime'] = pd.to_datetime(df['Date'] + ' ' + df['Time'], format='%d/%m/%Y %H:%M:%S')
    
    # Set datetime as index
    df.set_index('datetime', inplace=True)
    
    # Resample to hourly frequency (mean of Global_active_power)
    hourly_power = df['Global_active_power'].resample('H').mean()
    
    # Extract hour of day (0-23) from the index
    hourly_power_hour = hourly_power.index.hour
    
    # Group by hour and compute the mean power for that hour across all days
    hourly_avg = hourly_power.groupby(hourly_power_hour).mean()
    
    # Ensure we have 24 hours by reindexing and filling missing values (imputation)
    # This preserves the 24-hour structure for proper mapping
    hourly_avg = hourly_avg.reindex(range(24)).ffill().bfill()
    
    # Safety check: ensure we have exactly 24 hours
    if len(hourly_avg) != 24:
        raise ValueError(f"Expected 24 hours of data after imputation, got {len(hourly_avg)}")
    
    # Prepare data for KMeans: 24 points, each with one feature (average power)
    X = hourly_avg.values.reshape(-1, 1)
    
    # Apply KMeans with 3 clusters
    kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
    kmeans.fit(X)
    
    # Get labels for each hour (0-23)
    labels = kmeans.labels_
    
    # Map clusters to zones based on cluster centers (lowest -> Eco, middle -> Moderate, highest -> Peak)
    centers = kmeans.cluster_centers_.flatten()
    sorted_indices = np.argsort(centers)
    zone_map = {}
    zone_map[sorted_indices[0]] = 'Eco'
    zone_map[sorted_indices[1]] = 'Moderate'
    zone_map[sorted_indices[2]] = 'Peak'
    
    # Create hour-to-zone mapping
    hour_to_zone = {hour: zone_map[label] for hour, label in zip(range(24), labels)}
    
    # Save the cluster map (memory) for other agents
    skills_dir = r'C:\.opencode\skills'
    os.makedirs(skills_dir, exist_ok=True)
    cluster_map_path = os.path.join(skills_dir, 'cluster_map.json')
    with open(cluster_map_path, 'w') as f:
        json.dump(hour_to_zone, f)
    
    print(f"Usage Profiler trained. Cluster map saved to {cluster_map_path}")
    print("Hour to zone mapping:")
    for hour in range(24):
        print(f"  Hour {hour:2d}: {hour_to_zone[hour]}")
    
    return hour_to_zone

if __name__ == "__main__":
    train_usage_profiler()