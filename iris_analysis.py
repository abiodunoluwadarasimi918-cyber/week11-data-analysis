import pandas as pd
import matplotlib.pyplot as plt

# Load the Iris dataset
df = pd.read_csv("/storage/emulated/0/Download/Iris.csv")

# Basic exploration
print(df.head())
print()
df.info()
print()
print("Shape:", df.shape)
print("Columns:", df.columns)

# Summary statistics
print("\nSummary statistics:")
print(df.describe())

print("\nSpecies counts:")
print(df["Species"].value_counts())

print("\nMissing values:")
print(df.isnull().sum())

# Keep only the four measurement columns
features = df[["SepalLengthCm", "SepalWidthCm",
               "PetalLengthCm", "PetalWidthCm"]]

print("\nMean:")
print(features.mean())

print("\nMedian:")
print(features.median())

print("\nStandard deviation:")
print(features.std())

print("\nMean by species:")
print(df.groupby("Species")[list(features.columns)].mean())

print("\nCorrelation:")
print(features.corr())

# Create all plots in one figure
fig, axes = plt.subplots(4, 2, figsize=(12, 16))

# Histograms
axes[0, 0].hist(df["SepalLengthCm"])
axes[0, 0].set_title("Sepal Length")

axes[0, 1].hist(df["SepalWidthCm"])
axes[0, 1].set_title("Sepal Width")

axes[1, 0].hist(df["PetalLengthCm"])
axes[1, 0].set_title("Petal Length")

axes[1, 1].hist(df["PetalWidthCm"])
axes[1, 1].set_title("Petal Width")

# Scatter plots coloured by species
for species in df["Species"].unique():
    sub = df[df["Species"] == species]
    axes[2, 0].scatter(sub["SepalLengthCm"], sub["SepalWidthCm"], label=species)
    axes[2, 1].scatter(sub["PetalLengthCm"], sub["PetalWidthCm"], label=species)

axes[2, 0].set_xlabel("Sepal Length")
axes[2, 0].set_ylabel("Sepal Width")
axes[2, 0].set_title("Sepal Length vs Sepal Width")
axes[2, 0].legend()

axes[2, 1].set_xlabel("Petal Length")
axes[2, 1].set_ylabel("Petal Width")
axes[2, 1].set_title("Petal Length vs Petal Width")
axes[2, 1].legend()

# Box plot with shorter labels
box_data = features.rename(columns={
    "SepalLengthCm": "Sepal L",
    "SepalWidthCm": "Sepal W",
    "PetalLengthCm": "Petal L",
    "PetalWidthCm": "Petal W"
})
box_data.plot(kind="box", ax=axes[3, 0])
axes[3, 0].set_title("Iris Dataset Box Plot")
axes[3, 0].set_ylabel("Centimetres")

# Remove the unused space
axes[3, 1].axis("off")

# Adjust the layout, save, then display
plt.tight_layout()
plt.savefig("iris_plots.png")
plt.show()