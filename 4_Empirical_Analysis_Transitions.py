import matplotlib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns
matplotlib.use('TkAgg')

# Read the data
df = pd.read_csv(r"C:\Users\60848\OneDrive - Bain\Desktop\Project_Genome\USA_Journeys_summary.csv")

# Desired sectors and date range
unique_sectors = df["Sector"].unique()
desired_sectors = unique_sectors # ["Technology"]
plot_label = "market"
start_year = 2014
end_year = 2023

# 1. Filter by sector
df_filtered = df[df['Sector'].isin(desired_sectors)]

# 2. Filter by date range
df_filtered = df_filtered[(df_filtered['Year_beginning'] >= start_year) & (df_filtered['Year_final'] <= end_year)]

# 3. Remove rows with '0' in Genome_classification_beginning or Genome_classification_end
df_filtered = df_filtered[(df_filtered['Genome_classification_beginning'] != '0') & (df_filtered['Genome_classification_end'] != '0')]

# 4. Get unique genome classes
genome_classes = sorted(set(df['Genome_classification_beginning']).union(df['Genome_classification_end']))
genome_classes = [gc for gc in genome_classes if gc != '0']

# 5. Create the frequency matrix with all unique classes included
frequency_matrix = pd.crosstab(df_filtered['Genome_classification_beginning'], df_filtered['Genome_classification_end'])

# 6. Reindex the frequency matrix to include all genome classes
frequency_matrix = frequency_matrix.reindex(index=genome_classes, columns=genome_classes, fill_value=0)

# 7. Normalize the frequency matrix to create the transition matrix
transition_matrix = frequency_matrix.div(frequency_matrix.sum(axis=1), axis=0).fillna(0)

# 8. Plot the heatmap using matplotlib
fig, ax = plt.subplots(figsize=(12, 10))
cax = ax.matshow(transition_matrix, cmap="Blues")

# Add colorbar
fig.colorbar(cax)

# Set x and y axis labels
ax.set_xticks(np.arange(len(genome_classes)))
ax.set_yticks(np.arange(len(genome_classes)))
ax.set_xticklabels(genome_classes, rotation=45, ha="right")
ax.set_yticklabels(genome_classes)

# Annotate each cell with the percentage
for i in range(len(genome_classes)):
    for j in range(len(genome_classes)):
        text = ax.text(j, i, f"{transition_matrix.iloc[i, j]:.2%}",
                       ha="center", va="center", color="black")

# Adjust the layout to ensure the x-axis labels are visible
plt.subplots_adjust(bottom=0.25)

# Set title and labels
plt.title(f'{plot_label}_Transition_Matrix_' + str(start_year) + "-" + str(end_year))
plt.xlabel('To Genome Class')
plt.ylabel('From Genome Class')

# Save and show the plot
plt.savefig("Transition_matrix_" + plot_label + "_" + str(start_year) + "-" + str(end_year) + ".png")
plt.show()