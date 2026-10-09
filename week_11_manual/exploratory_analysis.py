import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# 1. Load the dataset from CSV file
# Replace 'Iris.csv' with the actual path to your CSV file if stored elsewhere
df = pd.read_csv('../Iris.csv')

# Drop 'Id' column if present in the Kaggle CSV format
if 'Id' in df.columns:
    df = df.drop(columns=['Id'])

print("--- First 5 Rows ---")
print(df.head())

print("\n--- Dataset Info ---")
print(df.info())

# 2. Calculate Summary Statistics
print("\n--- Summary Statistics (Numerical Features) ---")
print(df.describe())

print("\n--- Class / Species Counts ---")
print(df['Species'].value_counts())

# 3. Data Visualization
# Set style
sns.set_theme(style="whitegrid")

# Plot 1: Pairplot showing relationships between features grouped by species
sns.pairplot(df, hue="Species", markers=["o", "s", "D"])
plt.suptitle("Pairplot of Iris Features by Species", y=1.02)
plt.show()

# Plot 2: Boxplots to visualize feature distributions and outliers
plt.figure(figsize=(10, 6))
df.melt(id_vars="Species").pipe(
    (sns.boxplot, "data"), x="variable", y="value", hue="Species"
)
plt.title("Feature Distributions Across Species")
plt.xlabel("Features")
plt.ylabel("Measurement (cm)")
plt.show()

# Plot 3: Correlation Heatmap for numerical columns
plt.figure(figsize=(8, 6))
numeric_df = df.drop(columns=['Species'])
sns.heatmap(numeric_df.corr(), annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Correlation Heatmap of Iris Features")
plt.show()