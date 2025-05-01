import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris
import os

# Create a directory to save plots
output_dir = "outputs"
os.makedirs(output_dir, exist_ok=True)

# Load Iris dataset
try:
    iris = load_iris()
    df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
    df['species'] = iris.target
    df['species'] = df['species'].map(dict(zip(range(3), iris.target_names)))
except Exception as e:
    print(f"Error loading dataset: {e}")
    exit()

# Task 1: Explore Dataset
print("First 5 rows of dataset:")
print(df.head())

print("\nDataset Info:")
print(df.info())

print("\nMissing values:")
print(df.isnull().sum())

# Clean Data: (Iris dataset is clean but check just in case)
df.dropna(inplace=True)

# Task 2: Basic Statistics
print("\nBasic Statistics:")
print(df.describe())

# Group by species and compute mean
grouped = df.groupby('species').mean()
print("\nMean values by species:")
print(grouped)

# Task 3: Visualizations

# 1. Line Plot - Mean values per species
grouped.plot(kind='line', marker='o', title='Mean Feature Values per Species')
plt.ylabel("Mean Value")
plt.xlabel("Feature")
plt.grid(True)
plt.savefig(os.path.join(output_dir, "line_plot.png"))
plt.show()

# 2. Bar Chart - Average petal length per species
sns.barplot(x='species', y='petal length (cm)', data=df)
plt.title("Average Petal Length per Species")
plt.savefig(os.path.join(output_dir, "bar_chart.png"))
plt.show()

# 3. Histogram - Sepal Length Distribution
plt.hist(df['sepal length (cm)'], bins=15, color='skyblue', edgecolor='black')
plt.title("Distribution of Sepal Length")
plt.xlabel("Sepal Length (cm)")
plt.ylabel("Frequency")
plt.savefig(os.path.join(output_dir, "histogram.png"))
plt.show()

# 4. Scatter Plot - Sepal Length vs Petal Length
sns.scatterplot(x='sepal length (cm)', y='petal length (cm)', hue='species', data=df)
plt.title("Sepal Length vs Petal Length")
plt.savefig(os.path.join(output_dir, "scatter_plot.png"))
plt.show()

# Save cleaned data
df.to_csv(os.path.join(output_dir, "cleaned_iris.csv"), index=False)

print("\nAll plots and cleaned data have been saved in the 'outputs' directory.")
