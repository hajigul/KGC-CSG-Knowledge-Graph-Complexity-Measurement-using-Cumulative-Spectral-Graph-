import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import rcParams
import numpy as np

# Load dataset
file_path = r"D:\dataset_complexity\Multi_Model_MRR_vs_Features.csv"
df = pd.read_csv(file_path)

# Normalize column names
df.columns = df.columns.str.strip().str.replace('\s+', '_', regex=True)

# Clean numeric data
df.replace(',', '', regex=True, inplace=True)
for col in df.columns:
    if col != 'Datasets':
        df[col] = pd.to_numeric(df[col], errors='coerce')

# MRR columns
mrr_columns = ["TransE_MRR", "RESCAL_MRR", "RotatE_MRR", "ConvE_MRR", "TuckER_MRR", "ComplEx_MRR"]

# Feature columns
start_feature = "Average_Degree"
if start_feature not in df.columns:
    raise KeyError(f"'{start_feature}' not found in columns. Available: {list(df.columns)}")

feature_start_index = df.columns.get_loc(start_feature)
feature_columns = df.columns[feature_start_index:].tolist()

# Style setup
sns.set(style="whitegrid")
rcParams.update({'font.size': 10})

# MRR visual styles: fixed line (dashed), unique marker
mrr_styles = {
    "TransE_MRR":  {'color': 'blue',   'marker': 'o'},
    "RESCAL_MRR":  {'color': 'green',  'marker': 's'},
    "RotatE_MRR":  {'color': 'red',    'marker': '^'},
    "ConvE_MRR":   {'color': 'orange', 'marker': 'D'},
    "TuckER_MRR":  {'color': 'purple', 'marker': 'P'},
    "ComplEx_MRR": {'color': 'brown',  'marker': '*'}
}

# Layout config
num_plots = len(feature_columns)
num_cols = 4
num_rows = (num_plots + num_cols - 1) // num_cols

fig = plt.figure(figsize=(5 * num_cols, 4 * num_rows))
fig.suptitle("MRR vs Network Features", fontsize=18, fontweight='bold', y=1.02)

for i, feature in enumerate(feature_columns, 1):
    ax = plt.subplot(num_rows, num_cols, i)

    for mrr in mrr_columns:
        if mrr not in df.columns:
            continue

        temp_df = df[['Datasets', mrr, feature]].dropna()
        if temp_df.empty or temp_df[feature].dropna().empty:
            continue

        temp_df = temp_df.sort_values(by=mrr)
        correlation = temp_df[[mrr, feature]].corr().iloc[0, 1]
        corr_str = f"{correlation:.2f}" if not np.isnan(correlation) else "NaN"
        label = f"{mrr.replace('_MRR', '')} ({corr_str})"

        style = mrr_styles[mrr]

        sns.lineplot(
            x=mrr,
            y=feature,
            data=temp_df,
            label=label,
            color=style['color'],
            linestyle='--',
            marker=style['marker'],
            markersize=6,
            ax=ax
        )

        if mrr == mrr_columns[0]:
            strength = (
                "Strong" if abs(correlation) >= 0.7 else
                "Moderate" if abs(correlation) >= 0.3 else "Weak"
            )
            relationship = "inverse" if correlation < 0 else "direct"
            ax.set_title(
                f"{feature}\n{strength} {relationship} (Corr: {corr_str})",
                fontsize=11,
                pad=10,
                fontweight='bold'
            )

    # Bold axis labels
    ax.set_xlabel("MRR", fontsize=9, fontweight='bold')
    ax.set_ylabel(feature, fontsize=9, fontweight='bold')

    #  Bold tick labels
    ax.tick_params(axis='both', labelsize=8)
    for tick in ax.get_xticklabels():
        tick.set_fontweight('bold')
    for tick in ax.get_yticklabels():
        tick.set_fontweight('bold')

    # Bold legend
    leg = ax.legend(title="MRR Models (Correlation)", fontsize=9, title_fontsize=10, loc='best')
    for text in leg.get_texts():
        text.set_fontweight('bold')
    if leg.get_title():
        leg.get_title().set_fontweight('bold')

# Final layout
plt.tight_layout()
plt.subplots_adjust(hspace=0.6, wspace=0.4, top=0.92)
plt.show()