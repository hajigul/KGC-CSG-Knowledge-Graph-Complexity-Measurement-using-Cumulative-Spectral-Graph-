import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.lines import Line2D
from matplotlib import rcParams
import numpy as np
import itertools
import uuid


# Load the dataset
file_path = r"D:\dataset_complexity\Multi_Model_Mrr_vs_Features.csv"

df = pd.read_csv(file_path)

# Clean numerical columns - now only need to exclude 'Datasets'
numeric_cols = df.columns.drop(['Datasets'])  # Removed 'Bipartiteness'
for col in numeric_cols:
    df[col] = pd.to_numeric(df[col].astype(str).str.replace(',', ''), errors='coerce')

# Select numerical columns (excluding Datasets and Total CSG column)
numerical_columns = df.select_dtypes(include=['number']).columns.tolist()
numerical_columns = [col for col in numerical_columns if col not in ['TransE_MRR', 'RESCAL_MRR', 'RotatE_MRR', 'ConvE_MRR', 'TuckER_MRR', 'ComplEx_MRR']]

# Set up visualization
sns.set(style="whitegrid")
rcParams.update({'font.size': 10})

# Define a list of markers and colors for dataset styling
markers = ['o', 's', '^', 'D', 'P', 'H', 'X', '*']  # List of markers
colors = ['blue', 'green', 'red', 'orange', 'purple', 'brown', 'pink', 'cyan']  # List of colors

# Mapping each dataset to a marker and color
datasets = df['Datasets'].unique()  # Get all unique datasets
dataset_style_map = {dataset: {'marker': markers[i % len(markers)], 'color': colors[i % len(colors)]} for i, dataset in enumerate(datasets)}

# Collect handles for the global legend
legend_handles = []

# List of MRR columns
mrr_columns = ['TransE_MRR', 'RESCAL_MRR', 'RotatE_MRR', 'ConvE_MRR', 'TuckER_MRR', 'ComplEx_MRR']

# List of features to compare with MRR
features = [col for col in numerical_columns if col not in mrr_columns]

# Create the subplots
for mrr_col in mrr_columns:
    # Sort the dataframe based on the current MRR column
    df_sorted = df.sort_values(by=mrr_col)
    
    # Create a grid with one extra subplot for the legend
    num_plots = len(features)
    num_cols = 6
    num_rows = (num_plots + 1) // num_cols + 1
    
    plt.figure(figsize=(20, 20))
    
    for i, feature in enumerate(features, 1):
        ax = plt.subplot(num_rows, num_cols, i)
        
        temp_df = df_sorted[['Datasets', mrr_col, feature]].dropna()
        
        # Check if the column contains enough data to plot
        if temp_df[feature].dropna().empty:
            print(f"Warning: No valid data to plot for {feature}. Skipping plot.")
            continue  # Skip this iteration and move to the next column
        
        # Calculate dynamic vertical offset based on data range
        y_range = temp_df[feature].max() - temp_df[feature].min()
        
        # Check if y_range is NaN or Inf, and handle it gracefully
        if np.isnan(y_range) or np.isinf(y_range):
            y_range = 1  # Default value in case of NaN or Inf (prevents the error)
            print(f"Warning: Invalid y_range detected for {feature}. Defaulting to 1.")
        
        offset = 0.05 * y_range
        
        # Scatter points for datasets with different marker styles and colors
        for idx, row in temp_df.iterrows():
            # Get the style (marker and color) for the current dataset
            style = dataset_style_map[row['Datasets']]
            
            scatter = ax.scatter(
                x=row[mrr_col],
                y=row[feature],
                s=60,
                color=style['color'],
                marker=style['marker'],
                label=row['Datasets'] if idx == 0 else "",  # Add label for first point of each dataset
            )
        
        # Line plot with increased line width
        sns.lineplot(
            x=mrr_col,
            y=feature,
            data=temp_df,
            color='red',
            linestyle='--',
            linewidth=2.5,  # Increased line width
            ax=ax,
            legend=False
        )
        
        # Add dataset names below each point with improved positioning
        for _, row in temp_df.iterrows():
            ax.text(
                x=row[mrr_col],
                y=row[feature] - offset,
                s=row['Datasets'],
                fontsize=8,
                ha='center',
                va='top',
                rotation=45,
                bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', pad=1)
            )
        
        # Calculate correlation with emphasis on inverse relationships
        correlation = temp_df[[mrr_col, feature]].corr().iloc[0, 1]
        
        # Determine relationship type and strength
        if abs(correlation) >= 0.7:
            strength = "Strong"
        elif abs(correlation) >= 0.3:
            strength = "Moderate"
        else:
            strength = "Weak"
        
        relationship = "inverse" if correlation < 0 else "direct"
        
        # Format correlation display
        corr_display = f"{abs(correlation):.2f}"
        if correlation < 0:
            corr_display = f"-{abs(correlation):.2f}"
        
        ax.set_title(
            f"{feature}\n{strength} {relationship} relationship\nCorrelation: {corr_display}", 
            fontsize=12, 
            pad=10,
            fontweight='bold'  # Make title bold
        )
        
        ax.set_xlabel(mrr_col, fontsize=10, fontweight='bold')  # Make label bold
        ax.set_ylabel(feature, fontsize=10, fontweight='bold')  # Set y-axis label to the column name
        ax.tick_params(axis='both', labelsize=8)
        
        # Adjust y-axis limits to accommodate labels
        y_min, y_max = ax.get_ylim()
        ax.set_ylim(y_min - 0.15 * y_range, y_max + 0.05 * y_range)
    
    # Create a new subplot for the legend (empty subplot)
    ax_legend = plt.subplot(num_rows, num_cols, num_plots + 1)
    
    # Remove axis
    ax_legend.axis('off')
    
    # Create a global legend with the dataset markers and colors
    legend_labels = [Line2D([0], [0], marker=dataset_style_map[dataset]['marker'], color='w', markerfacecolor=dataset_style_map[dataset]['color'], markersize=8, label=dataset) for dataset in datasets]
    
    # Add the global legend to this new subplot with 2 columns
    ax_legend.legend(handles=legend_labels, title='Datasets', loc='center', fontsize=12, title_fontsize=14, frameon=False, ncol=2)
    
    plt.tight_layout()
    plt.subplots_adjust(hspace=0.6, wspace=0.4)
    plt.show()