import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

DATA_PATH = (
    Path(__file__).resolve().parent.parent
    / "data"
    / "raw"
    / "sentry_synthetic_dataset.csv"
)

df = pd.read_csv(DATA_PATH)

features = [
    "VMQ2",
    "VMQ3",
    "VMQ135",
    "VSEN0568",
    "dVdt_max",
    "temperature",
    "humidity",
]

print("=" * 70)
print("SENTRY DATASET INSPECTION")
print("=" * 70)

print(f"\nDataset shape: {df.shape}")

print("\nColumns:")
print(df.columns.tolist())

print("\nMissing values:")
print(df.isnull().sum())

print("\nClass distribution:")
print(df["label"].value_counts())

print("\nFeature statistics:")
print(df[features].describe().round(3))

print("\nMean feature values by class:")
print(
    df.groupby("label")[features]
    .mean()
    .round(3)
)

print("\nFeature standard deviation by class:")
print(
    df.groupby("label")[features]
    .std()
    .round(3)
)

print("\nCorrelation matrix:")
print(df[features].corr().round(2))

# ------------------------------------------------------------
# Create results directory
# ------------------------------------------------------------

RESULTS_DIR = (
    Path(__file__).resolve().parent.parent / "results"
)
RESULTS_DIR.mkdir(exist_ok=True)

# ------------------------------------------------------------
# Plot feature distributions
# ------------------------------------------------------------

for feature in features:

    plt.figure(figsize=(9, 5))

    for label in sorted(df["label"].unique()):

        subset = df[df["label"] == label]

        plt.hist(
            subset[feature],
            bins=20,
            alpha=0.45,
            label=label
        )

    plt.title(f"{feature} Distribution by Class")
    plt.xlabel(feature)
    plt.ylabel("Count")
    plt.legend()
    plt.tight_layout()

    output = RESULTS_DIR / f"{feature}_distribution.png"
    plt.savefig(output, dpi=150)
    plt.close()

print("\nDistribution plots saved to:")
print(RESULTS_DIR)

print("\nDataset inspection complete.")