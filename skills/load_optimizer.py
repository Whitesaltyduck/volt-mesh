import json
import os
import pandas as pd
import numpy as np

def find_optimal_heavy_appliance_window():
    """
    Implement a Greedy Best-First Search algorithm to find the lowest-cost 1-hour window
    for heavy appliances based on the Profiler's "Eco" zones.
    
    Cost function: Prefer Eco zones > Moderate zones > Peak zones
    For simplicity, we assign costs: Eco=1, Moderate=2, Peak=3
    Tie-breaker: Among equal cost zones, select hour with lowest historical average power
    The goal is to find a contiguous 1-hour window with minimum cost.
    
    Returns:
        best_hour: The optimal hour to run heavy appliances
        best_cost: The cost of that hour
        zone: The zone classification of that hour
    """
    # Load the cluster map from the Usage Profiler
    skills_dir = r'C:\.opencode\skills'
    cluster_map_path = os.path.join(skills_dir, 'cluster_map.json')
    
    if not os.path.exists(cluster_map_path):
        raise FileNotFoundError(f"Cluster map not found at {cluster_map_path}. Please run the Usage Profiler first.")
    
    with open(cluster_map_path, 'r') as f:
        hour_to_zone = json.load(f)
    
    # Convert string keys to integers
    hour_to_zone = {int(k): v for k, v in hour_to_zone.items()}
    
    # Load historical power data for tie-breaking
    data_path = r'C:\data\power_usage.csv'
    df = pd.read_csv(data_path, sep=';', na_values='?')
    
    # Combine Date and Time into a datetime column
    df['datetime'] = pd.to_datetime(df['Date'] + ' ' + df['Time'], format='%d/%m/%Y %H:%M:%S')
    
    # Set datetime as index
    df.set_index('datetime', inplace=True)
    
    # Resample to hourly frequency (mean of Global_active_power) and group by hour
    hourly_power = df['Global_active_power'].resample('H').mean()
    hourly_power_hour = hourly_power.index.hour
    hourly_avg_power = hourly_power.groupby(hourly_power_hour).mean()
    
    # Define cost function: Eco=1 (lowest cost), Moderate=2, Peak=3 (highest cost)
    zone_cost = {'Eco': 1, 'Moderate': 2, 'Peak': 3}
    
    # Find all candidate hours and their costs
    candidates = []
    for hour in range(24):
        zone = hour_to_zone[hour]
        cost = zone_cost[zone]
        # Get historical average power for this hour (default to infinity if not available)
        avg_power = hourly_avg_power.get(hour, float('inf'))
        candidates.append({
            'hour': hour,
            'zone': zone,
            'cost': cost,
            'avg_power': avg_power
        })
    
    # Sort by cost (primary) and avg_power (secondary/tie-breaker)
    candidates.sort(key=lambda x: (x['cost'], x['avg_power']))
    
    # Select the best candidate
    best_candidate = candidates[0]
    best_hour = best_candidate['hour']
    best_cost = best_candidate['cost']
    best_zone = best_candidate['zone']
    
    return best_hour, best_cost, best_zone

def get_all_eco_hours():
    """
    Get all hours classified as 'Eco' zones for running heavy appliances.
    """
    # Load the cluster map from the Usage Profiler
    skills_dir = r'C:\.opencode\skills'
    cluster_map_path = os.path.join(skills_dir, 'cluster_map.json')
    
    if not os.path.exists(cluster_map_path):
        raise FileNotFoundError(f"Cluster map not found at {cluster_map_path}. Please run the Usage Profiler first.")
    
    with open(cluster_map_path, 'r') as f:
        hour_to_zone = json.load(f)
    
    # Convert string keys to integers
    hour_to_zone = {int(k): v for k, v in hour_to_zone.items()}
    
    # Get all Eco hours
    eco_hours = [hour for hour in range(24) if hour_to_zone[hour] == 'Eco']
    
    return eco_hours

if __name__ == "__main__":
    # Find the optimal hour for heavy appliances
    best_hour, best_cost, best_zone = find_optimal_heavy_appliance_window()
    
    print(f"Load Optimizer (The Searcher) Analysis:")
    print(f"Optimal hour to run heavy appliances: {best_hour}:00")
    print(f"Zone classification: {best_zone}")
    print(f"Cost value: {best_cost} (lower is better)")
    
    # Also show all Eco hours as alternatives
    eco_hours = get_all_eco_hours()
    print(f"\nAll Eco hours available for heavy appliances: {eco_hours}")
    
    # Save the recommendation for potential use by other agents
    skills_dir = r'C:\.opencode\skills'
    recommendation = {
        'optimal_hour': best_hour,
        'optimal_zone': best_zone,
        'optimal_cost': best_cost,
        'all_eco_hours': eco_hours
    }
    
    recommendation_path = os.path.join(skills_dir, 'load_optimizer_recommendation.json')
    with open(recommendation_path, 'w') as f:
        json.dump(recommendation, f, indent=2)
    
    print(f"\nRecommendation saved to {recommendation_path}")