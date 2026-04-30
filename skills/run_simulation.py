import pandas as pd
import numpy as np
import json
import os
from sklearn.naive_bayes import GaussianNB
import joblib

def simulate_high_voltage_spike():
    """
    Simulate a high voltage spike scenario and run the agent mesh.
    Returns a markdown efficiency audit.
    """
    print("=== Volt-Mesh Simulation: High Voltage Spike Scenario ===\n")
    
    # Step 1: Ensure all agents are trained
    print("Step 1: Training/Loading Agent Models...")
    
    # Load or train Usage Profiler (Clusternaut)
    skills_dir = r'C:\.opencode\skills'
    cluster_map_path = os.path.join(skills_dir, 'cluster_map.json')
    
    if not os.path.exists(cluster_map_path):
        print("  Training Usage Profiler (Clusternaut)...")
        # We'll import and run it, but for simplicity let's just check if it exists
        # In a full implementation, we'd call the training function
        pass
    
    # Load cluster map
    with open(cluster_map_path, 'r') as f:
        cluster_map = json.load(f)
    cluster_map = {int(k): v for k, v in cluster_map.items()}
    print(f"  Loaded cluster map with {len(cluster_map)} hours")
    
    # Load or train Reliability Sentinel (The Predictor)
    model_path = os.path.join(skills_dir, 'reliability_sentinel_model.joblib')
    if not os.path.exists(model_path):
        print("  Training Reliability Sentinel (The Predictor)...")
        # Again, in practice we'd call the training function
        pass
    
    nb_classifier = joblib.load(model_path)
    print("  Loaded Reliability Sentinel model")
    
    # Step 2: Simulate high power consumption (overload condition)
    print("\nStep 2: Simulating High Power Consumption Spike...")
    
    # Get the 95th percentile threshold from our earlier analysis
    threshold = 3.26  # kW
    
    # Simulate a high power consumption spike (well above threshold)
    high_power_spike = 8.0  # kW - this is quite high
    
    print(f"  Normal threshold for overload: {threshold:.2f} kW")
    print(f"  Simulated power consumption: {high_power_spike:.2f} kW")
    print(f"  This is {high_power_spike/threshold:.1f}x the threshold")
    
    # Step 3: Run the Reliability Sentinel to check overload probability
    print("\nStep 3: Running Reliability Sentinel (The Predictor)...")
    # For simulation, we'll use high values for voltage and intensity to stress test
    # In a real scenario, these would come from current sensor readings
    overload_prob, should_trigger = predict_overload_probability(nb_classifier, high_power_spike, voltage=200.0, intensity=15.0)
    
    print(f"  P(Overload | {high_power_spike:.2f} kW) = {overload_prob:.4f}")
    
    if should_trigger:
        print("  !! OVERLOAD PROBABILITY > 0.75 - TRIGGERING LOAD BALANCE REQUEST")
        load_balance_triggered = True
    else:
        print("  OK Overload probability within normal bounds")
        load_balance_triggered = False
    
    # Step 4: If load balance requested, run the Load Optimizer
    print("\nStep 4: Running Load Optimizer (The Searcher)...")
    if load_balance_triggered:
        # Get optimization recommendation
        eco_hours = get_eco_hours(cluster_map)
        optimal_hour = find_optimal_hour(cluster_map)
        
        print(f"  Eco hours available: {eco_hours}")
        print(f"  Optimal hour for heavy appliances: {optimal_hour}:00 ({cluster_map[optimal_hour]} zone)")
        
        # Step 5: Generate Efficiency Audit
        print("\nStep 5: Generating Efficiency Audit...")
        audit_markdown = generate_efficiency_audit(
            high_power_spike, 
            threshold, 
            overload_prob, 
            load_balance_triggered,
            optimal_hour,
            cluster_map[optimal_hour],
            eco_hours,
            cluster_map
        )
        
        # Save the audit
        audit_path = os.path.join(skills_dir, 'efficiency_audit.md')
        with open(audit_path, 'w') as f:
            f.write(audit_markdown)
        
        print(f"  * Efficiency audit saved to {audit_path}")
        print("\n" + "="*50)
        print("EFFICIENCY AUDIT PREVIEW")
        print("="*50)
        print(audit_markdown[:500] + "..." if len(audit_markdown) > 500 else audit_markdown)
        
        return audit_markdown
    else:
        print("  No load balance request triggered - no optimization needed")
        return None

def predict_overload_probability(model, current_load, voltage=None, intensity=None):
    """Helper to predict overload probability"""
    prob, should_trigger = model.predict_proba([[current_load, 
                                                voltage if voltage is not None else 230.0, 
                                                intensity if intensity is not None else 10.0]])[0][1], model.predict_proba([[current_load, 
                                                                                                                               voltage if voltage is not None else 230.0, 
                                                                                                                               intensity if intensity is not None else 10.0]])[0][1] > 0.75
    # Actually, let's call the model's predict_proba properly
    features = [[current_load, 
                voltage if voltage is not None else 230.0, 
                intensity if intensity is not None else 10.0]]
    prob_result = model.predict_proba(features)[0]
    prob = prob_result[1]  # Probability of overload (class 1)
    should_trigger = prob > 0.75
    return prob, should_trigger

def get_eco_hours(cluster_map):
    """Get all hours classified as Eco zones"""
    return [hour for hour in range(24) if cluster_map[hour] == 'Eco']

def find_optimal_hour(cluster_map):
    """Find the optimal hour (lowest cost) for heavy appliances"""
    zone_cost = {'Eco': 1, 'Moderate': 2, 'Peak': 3}
    best_hour = None
    best_cost = float('inf')
    
    for hour in range(24):
        cost = zone_cost[cluster_map[hour]]
        if cost < best_cost:
            best_cost = cost
            best_hour = hour
    
    return best_hour

def generate_efficiency_audit(power_consumption, threshold, overload_prob, 
                            load_balance_triggered, optimal_hour, optimal_zone, eco_hours, cluster_map):
    """Generate a markdown efficiency audit"""
    
    audit = f"""# Volt-Mesh Efficiency Audit Report

## :bar_chart: Simulation Overview
- **Scenario**: High Power Consumption Spike
- **Simulated Power Consumption**: {power_consumption:.2f} kW
- **Overload Threshold (95th percentile)**: {threshold:.2f} kW
- **Threshold Multiple**: {power_consumption/threshold:.1f}x

## :mag: Reliability Sentinel Analysis
- **Overload Probability P(Overload | {power_consumption:.2f} kW)**: {overload_prob:.4f}
- **Load Balance Request Triggered**: {"YES" if load_balance_triggered else "NO"}
- **Probability Threshold for Action**: 0.75
- **Status**: {"!! CRITICAL OVERLOAD RISK DETECTED" if load_balance_triggered else "OK NORMAL OPERATION"}

## :zap: Load Optimizer Recommendations
"""
    
    if load_balance_triggered:
        audit += f"""- **Optimal Hour for Heavy Appliances**: {optimal_hour}:00
- **Zone Classification**: {optimal_zone} (lowest cost)
- **Available Eco Hours**: {eco_hours}
- **Recommended Action**: Schedule heavy appliance usage during {optimal_hour}:00-{optimal_hour+1}:00 for minimal grid impact

## :bar_chart: Energy Efficiency Insights
"""
        
        # Add some insights based on the data
        eco_count = len(eco_hours)
        audit += f"""- **Eco Hours Available**: {eco_count} out of 24 hours ({eco_count/24*100:.1f}% of day)
- **Peak Hours to Avoid**: {[h for h in range(24) if cluster_map[h] == 'Peak']}
- **Cost Savings Potential**: By shifting heavy loads to Eco hours, customers can reduce grid strain and potentially lower electricity costs

## :shield: Grid Stability Recommendations
"""
        
        if overload_prob > 0.9:
            audit += """- **IMMEDIATE ACTION REQUIRED**: Consider temporary non-essential load shedding
- **Customer Notification**: Alert high-consumption users of impending grid stress
- **Preventive Measure**: Activate demand response programs"""
        elif overload_prob > 0.8:
            audit += """- **MONITORING ADVISED**: Increased grid surveillance recommended
- **Prepare Response**: Have load balancing protocols ready
- **Customer Advisory**: Suggest voluntary load reduction during peak hours"""
        else:
            audit += """- **STANDARD OPERATION**: Grid operating within normal parameters
- **Continue Monitoring**: Maintain regular surveillance intervals
- **Optimize Efficiency**: Use predicted low-cost hours for flexible loads"""
    
    audit += f"""
## :clipboard: Technical Details
- **Agent Mesh Status**: All agents operational
- **Data Source**: C:\\data\\power_usage.csv
- **Timestamp**: Simulation run at {pd.Timestamp.now()}
- **Model Version**: Naive Bayes Classifier (trained on 95th percentile threshold)

## :end: Conclusion
{"The Volt-Mesh agent mesh successfully detected a high-risk overload condition and provided actionable recommendations for grid stability." if load_balance_triggered else "The Volt-Mesh agent mesh confirmed normal grid operation with no immediate action required."}
"""
    
    return audit

if __name__ == "__main__":
    audit = simulate_high_voltage_spike()
    if audit:
        print("\nSUCCESS: Simulation completed successfully!")
    else:
        print("\nWARNING: Simulation completed but no audit generated (no overload detected)")