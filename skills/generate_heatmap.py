import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import os

def generate_correlation_heatmap():
    """
    Generate a correlation heatmap using Seaborn to visualize the data features.
    Saves the heatmap as a PNG file in the skills directory.
    """
    # Load the data
    data_path = r'C:\data\power_usage.csv'
    df = pd.read_csv(data_path, sep=';', na_values='?')
    
    # Select only numeric columns for correlation
    numeric_df = df.select_dtypes(include=[np.number])
    
    # Calculate correlation matrix
    corr_matrix = numeric_df.corr()
    
    # Set up the matplotlib figure
    plt.figure(figsize=(12, 10))
    
    # Generate heatmap
    sns.heatmap(
        corr_matrix,
        annot=True,
        fmt=".2f",
        cmap='coolwarm',
        center=0,
        square=True,
        linewidths=0.5,
        cbar_kws={"shrink": 0.8}
    )
    
    plt.title('Correlation Heatmap of Power Usage Features', fontsize=16, pad=20)
    plt.tight_layout()
    
    # Save the figure
    skills_dir = r'C:\.opencode\skills'
    os.makedirs(skills_dir, exist_ok=True)
    heatmap_path = os.path.join(skills_dir, 'correlation_heatmap.png')
    plt.savefig(heatmap_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Correlation heatmap saved to {heatmap_path}")
    
    # Also save the correlation matrix as CSV for reference
    corr_csv_path = os.path.join(skills_dir, 'correlation_matrix.csv')
    corr_matrix.to_csv(corr_csv_path)
    print(f"Correlation matrix saved to {corr_csv_path}")

if __name__ == "__main__":
    generate_correlation_heatmap()