import matplotlib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns

matplotlib.use('TkAgg')

Genome_filter = True

# Read the data
df = pd.read_csv(r"C:\Users\60848\OneDrive - Bain\Desktop\Project_Genome\Journeys_summary.csv")

if Genome_filter:
    df = df.loc[(df["EP/FE_end"] >= -.3) & (df["EP/FE_end"] <= .5) &
                (df["Revenue_growth_end"] >= -.3) & (df["Revenue_growth_end"] <= 1.5) &
                (df["Annualized_TSR_Capiq"] >= -.4) & (df["Annualized_TSR_Capiq"] <= 1) &
                (df["Price_to_book"] > -200)]

# Desired sectors and date range
unique_sectors = df["Sector"].unique()
desired_sectors = unique_sectors  # ["Technology"]
plot_label = "market"
start_year = 2014
end_year = 2023

# 1. Filter by sector
df_filtered = df[df['Sector'].isin(desired_sectors)]

# 2. Filter by date range
df_filtered = df_filtered[(df_filtered['Year_beginning'] >= start_year) & (df_filtered['Year_final'] <= end_year)]

# 3. Remove rows with '0' in Genome_classification_beginning or Genome_classification_end
df_filtered = df_filtered[
    (df_filtered['Genome_classification_beginning'] != '0') & (df_filtered['Genome_classification_end'] != '0')]

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
ax.set_xticklabels(genome_classes, rotation=90, ha="right")
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

# Define the list of metrics you want to analyze
metrics = ["PE_end", "Annualized_TSR_Capiq"]

for metric in metrics:
    # Create an empty matrix to store the median values
    metric_matrix = pd.DataFrame(index=genome_classes, columns=genome_classes)

    # Iterate through all unique combinations of start and end Genome segments
    for start_class in genome_classes:
        for end_class in genome_classes:
            # Filter the data for the specific transition
            df_transition = df.loc[(df["Genome_classification_beginning"] == start_class) &
                                   (df["Genome_classification_end"] == end_class)]

            # Calculate the median value for this transition
            median_value = df_transition[metric].median()

            # Replace values less than 0 with 'N/A' and limit decimal places to 2
            if pd.isna(median_value) or median_value < 0:
                metric_matrix.loc[start_class, end_class] = 'N/A'
            else:
                if metric == "Annualized_TSR_Capiq":
                    metric_matrix.loc[start_class, end_class] = f"{median_value * 100:.2f}%"  # Convert to percentage
                else:
                    metric_matrix.loc[start_class, end_class] = f"{median_value:.2f}"

    # Save the matrix to a CSV file
    metric_matrix.to_csv(rf"C:\Users\60848\OneDrive - Bain\Desktop\Project_Genome\{metric}_transition_matrix.csv")

    # Convert 'N/A' to NaN and remove percentage signs for visualization purposes
    metric_matrix_viz = metric_matrix.replace('N/A', np.nan).replace('%', '', regex=True).astype(float)

    # Plot the heatmap using matplotlib
    fig, ax = plt.subplots(figsize=(12, 10))
    cax = ax.matshow(metric_matrix_viz, cmap="Blues", vmin=metric_matrix_viz.min().min(),
                     vmax=metric_matrix_viz.max().max())

    # Add colorbar
    fig.colorbar(cax)

    # Set x and y axis labels
    ax.set_xticks(np.arange(len(genome_classes)))
    ax.set_yticks(np.arange(len(genome_classes)))
    ax.set_xticklabels(genome_classes, rotation=90, ha="right")
    ax.set_yticklabels(genome_classes)

    # Annotate each cell with the value
    for i in range(len(genome_classes)):
        for j in range(len(genome_classes)):
            value = metric_matrix.iloc[i, j]
            text = ax.text(j, i, value, ha="center", va="center", color="black")

    # Adjust the layout to ensure the x-axis labels are visible
    plt.subplots_adjust(bottom=0.25)

    # Set title and labels
    plt.title(f'Median {metric} Transition Matrix')
    plt.xlabel('To Genome Class')
    plt.ylabel('From Genome Class')

    # Save and show the plot
    plt.savefig(f"{metric}_Transition_Matrix.png")
    plt.show()

print("Transition matrices for all metrics saved and visualized successfully.")
