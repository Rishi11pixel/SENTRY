import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import sklearn

print(f"scikit-learn version: {sklearn.__version__}")

train_df = pd.read_csv('data/raw/sentry_synthetic_dataset.csv')
features = ['VMQ2', 'VMQ3', 'VMQ135', 'VSEN0567', 'dVdt_max', 'temperature', 'humidity']
X = train_df[features].values

# Test multiple times with the exact same seed
for attempt in range(3):
    X_train_split, X_val_split, _, _ = train_test_split(
        X, train_df['label'], test_size=0.20, random_state=42, stratify=train_df['label']
    )
    
    scaler_check = StandardScaler()
    scaler_check.fit(X_train_split)
    
    print(f"\nAttempt {attempt + 1}:")
    print(f"  Mean[0] (VMQ2): {scaler_check.mean_[0]:.16f}")
    print(f"  Scale[0] (VMQ2): {scaler_check.scale_[0]:.16f}")

scaler_mean_saved = np.load('models/tinyml/scaler_mean.npy')
scaler_scale_saved = np.load('models/tinyml/scaler_scale.npy')

print(f"\nSaved parameters:")
print(f"  Mean[0] (VMQ2): {scaler_mean_saved[0]:.16f}")
print(f"  Scale[0] (VMQ2): {scaler_scale_saved[0]:.16f}")

# Try computing with float64 throughout
print("\n" + "="*70)
print("ADDITIONAL TEST: Data types")
print("X dtype:", X.dtype)
print("X_train dtype:", X_train_split.dtype)

# Try to see if there's a difference if we use the exact same X dtype
print("\n" + "="*70)
print("Trying with explicit float64 conversion:")
X_float64 = train_df[features].values.astype(np.float64)

X_train_f64, X_val_f64, _, _ = train_test_split(
    X_float64, train_df['label'], test_size=0.20, random_state=42, stratify=train_df['label']
)

scaler_f64 = StandardScaler()
scaler_f64.fit(X_train_f64)

print(f"  Mean[0] with float64: {scaler_f64.mean_[0]:.16f}")
print(f"  Saved Mean[0]:        {scaler_mean_saved[0]:.16f}")
print(f"  Match: {np.allclose(scaler_f64.mean_, scaler_mean_saved)}")
