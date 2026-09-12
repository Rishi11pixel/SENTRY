import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

train_df = pd.read_csv('data/raw/sentry_synthetic_dataset.csv')
features = ['VMQ2', 'VMQ3', 'VMQ135', 'VSEN0567', 'dVdt_max', 'temperature', 'humidity']
X = train_df[features].values

X_train_split, X_val_split, _, _ = train_test_split(
    X, train_df['label'], test_size=0.20, random_state=42, stratify=train_df['label']
)

scaler_check = StandardScaler()
scaler_check.fit(X_train_split)

scaler_mean_saved = np.load('models/tinyml/scaler_mean.npy')
scaler_scale_saved = np.load('models/tinyml/scaler_scale.npy')

print("Recomputed scaler (80% train split, random_state=42):")
print("Mean:", scaler_check.mean_)
print("\nSaved scaler (from training):")
print("Mean:", scaler_mean_saved)
print("\nDifference in means:")
print(np.abs(scaler_check.mean_ - scaler_mean_saved))
print("\nMax absolute difference:", np.max(np.abs(scaler_check.mean_ - scaler_mean_saved)))

print("\n" + "="*70)
print("Scaler Scale comparison:")
print("Recomputed:", scaler_check.scale_)
print("Saved:     ", scaler_scale_saved)
print("\nDifference in scales:")
print(np.abs(scaler_check.scale_ - scaler_scale_saved))
print("Max absolute difference:", np.max(np.abs(scaler_check.scale_ - scaler_scale_saved)))

# Check if they're close enough
print("\n" + "="*70)
print("Numerical match check (default tolerance):")
print("Means match (rtol=1e-5):", np.allclose(scaler_check.mean_, scaler_mean_saved, rtol=1e-5))
print("Scales match (rtol=1e-5):", np.allclose(scaler_check.scale_, scaler_scale_saved, rtol=1e-5))
print("Means match (rtol=1e-3):", np.allclose(scaler_check.mean_, scaler_mean_saved, rtol=1e-3))
print("Scales match (rtol=1e-3):", np.allclose(scaler_check.scale_, scaler_scale_saved, rtol=1e-3))
