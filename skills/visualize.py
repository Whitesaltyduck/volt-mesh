import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import os

def generate_correlation_heatmap():
    """Generate a correlation heatmap of the power usage dataset"""
    # Read the semicolon-delimited file
    df = pd.read_csv(r"C:\data\power_usage.csv", delimiter=';', low_memory=False)
    
    # Handle missing values represented by '?'
    df.replace('?', np.nan, inplace=True)
    
    # Convert columns to numeric
    numeric_columns = ['Global_active_power', 'Global_reactive_power', 'Voltage', 
                      'Global_intensity', 'Sub_metering_1', 'Sub_metering_2', 'Sub_metering_3']
    for col in numeric_columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # Select only numeric columns for correlation
    df_numeric = df[numeric_columns].copy()
    
    # Drop rows with missing values for correlation calculation
    df_clean = df_numeric.dropna()
    
    # Calculate correlation matrix
    correlation_matrix = df_clean.corr()
    
    # Create heatmap
    plt.figure(figsize=(10, 8))
    sns.heatmap(correlation_matrix, 
                annot=True, 
                fmt='.2f', 
                cmap='coolwarm', 
                center=0,
                square=True,
                linewidths=0.5,
                cbar_kws={"shrink": 0.8})
    
    plt.title('Correlation Heatmap of Power Usage Features\nVolt-Mesh Smart Grid Analysis', 
              fontsize=16, fontweight='bold', pad=20)
    plt.tight_layout()
    
    # Save the plot to skills directory
    skills_dir = r'C:\.opencode\skills'
    os.makedirs(skills_dir, exist_ok=True)
    heatmap_path = os.path.join(skills_dir, 'correlation_heatmap.png')
    plt.savefig(heatmap_path, dpi=300, bbox_inches='tight')
    print(f"SUCCESS: Correlation heatmap saved as '{heatmap_path}'")
    
    # Also save correlation data
    corr_csv_path = os.path.join(skills_dir, 'correlation_matrix.csv')
    correlation_matrix.to_csv(corr_csv_path)
    print(f"SUCCESS: Correlation matrix saved as '{corr_csv_path}'")
    
    # Show some statistics
    print("\nDATASET STATISTICS:")
    print(f"   Total rows: {len(df)}")
    print(f"   Clean rows (no missing values): {len(df_clean)}")
    print(f"   Missing values: {len(df) - len(df_clean)} ({((len(df) - len(df_clean)) / len(df) * 100):.1f}%)")
    
    print("\nSTRONGEST CORRELATIONS:")
    # Get upper triangle of correlation matrix
    upper_tri = correlation_matrix.where(
        np.triu(np.ones(correlation_matrix.shape), k=1).astype(bool)
    )
    
    # Find strongest correlations (absolute value)
    strongest_corr = upper_tri.unstack().sort_values(key=abs, ascending=False)
    strongest_corr = strongest_corr.dropna().head(5)
    
     for (var1, var2), corr_value in strongest_corr.items():
         print(f"   {var1} <-> {var2}: {corr_value:.3f}")

if __name__ == "__main__":
    generate_correlation_heatmap()