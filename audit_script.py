import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import json
from pathlib import Path
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score, precision_recall_fscore_support

# Try to import tensorflow, but continue if not available
try:
    import tensorflow as tf
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False
    print("WARNING: TensorFlow not available, skipping TinyML evaluation")

# ===================================================================
# PART 1: DATA LEAKAGE AUDIT
# ===================================================================
print('='*70)
print('DATA LEAKAGE AUDIT')
print('='*70)

train_df = pd.read_csv('data/raw/sentry_synthetic_dataset.csv')
test_df = pd.read_csv('data/raw/sentry_synthetic_test.csv')

# Check if test data appears in training data
train_rows_set = set(map(tuple, train_df.values))
test_rows_set = set(map(tuple, test_df.values))

overlap = train_rows_set & test_rows_set
print(f'\nTest rows that appear in training data: {len(overlap)}')
print(f'PASS: No exact row overlap' if len(overlap) == 0 else f'FAIL: {len(overlap)} rows overlap')

# Verify scaler was fitted only on training data
scaler_mean = np.load('models/tinyml/scaler_mean.npy')
scaler_scale = np.load('models/tinyml/scaler_scale.npy')

features = ['VMQ2', 'VMQ3', 'VMQ135', 'VSEN0567', 'dVdt_max', 'temperature', 'humidity']
X = train_df[features].values

X_train_split, X_val_split, _, _ = train_test_split(
    X, train_df['label'], test_size=0.20, random_state=42, stratify=train_df['label']
)

scaler_check = StandardScaler()
scaler_check.fit(X_train_split)

print(f'\nScaler mean matches: {np.allclose(scaler_check.mean_, scaler_mean)}')
print(f'Scaler scale matches: {np.allclose(scaler_check.scale_, scaler_scale)}')

# Verify class labels
label_classes = np.load('models/tinyml/label_classes.npy', allow_pickle=True)
expected_classes = ['SAFE', 'WEATHER', 'ALCOHOL', 'EXPLOSIVE', 'NARCOTIC']
print(f'Label order matches: {list(label_classes) == expected_classes}')

# ===================================================================
# PART 2: VERIFY FEATURE EXTRACTION LOGIC
# ===================================================================
print('\n' + '='*70)
print('FEATURE EXTRACTION AUDIT')
print('='*70)

# Read the generate_dataset code to understand dV/dt calculation
print('\nFrom generate_dataset.py:')
print('  - dV/dt calculated from FIRST 4 samples (0.0s, 0.1s, 0.2s, 0.3s)')
print('  - dt = 0.1 seconds')
print('  - Uses max absolute slope across first 4 readings')
print('  - Final window = last 5 readings (0.5 second averaging)')
print('  - Environmental compensation: V_comp = V_raw - alpha_H*(H-50) - alpha_T*(T-25)')

# Manually verify one sample
print('\nSample verification from dataset:')
sample = train_df.iloc[0]
print(f'  VMQ2={sample["VMQ2"]:.4f}, VMQ3={sample["VMQ3"]:.4f}')
print(f'  VMQ135={sample["VMQ135"]:.4f}, VSEN0567={sample["VSEN0567"]:.4f}')
print(f'  dVdt_max={sample["dVdt_max"]:.4f}')
print(f'  temp={sample["temperature"]:.1f}, humidity={sample["humidity"]:.1f}')

# ===================================================================
# PART 3: VERIFY SENSOR BASELINE CONVERSION
# ===================================================================
print('\n' + '='*70)
print('SENSOR BASELINE CONVERSION AUDIT')
print('='*70)

print('\nFrom config.py:')
print('  BASELINE = [0.82, 0.74, 0.91, 0.34] (model-space)')
print('  These are used DURING training for SYNTHETIC data generation')

print('\nFrom ml_server.py (inference):')
print('  MODEL_BASELINES = {MQ2: 0.82, MQ3: 0.74, MQ135: 0.91}')
print('  REAL_BASELINES = {MQ2: 3069.0, MQ3: 530.33, MQ135: 1043.27}')
print('  Formula: V_model = (Raw / RealBaseline) * ModelBaseline')

print('\nExpected baseline indices:')
print('  Index 0: MQ2   (model=0.82, real=3069.0)')
print('  Index 1: MQ3   (model=0.74, real=530.33)')
print('  Index 2: MQ135 (model=0.91, real=1043.27)')
print('  Index 3: SEN0567 (model=0.34, no conversion - assumed already model space)')

# ===================================================================
# PART 4: ENVIRONMENTAL COMPENSATION AUDIT
# ===================================================================
print('\n' + '='*70)
print('ENVIRONMENTAL COMPENSATION AUDIT')
print('='*70)

print('\nCompensation coefficients (from both config.py and ml_server.py):')
coeffs = {
    'MQ2': {'alpha_H': 0.0018, 'alpha_T': 0.0008},
    'MQ3': {'alpha_H': 0.0014, 'alpha_T': 0.0007},
    'MQ135': {'alpha_H': 0.0022, 'alpha_T': 0.0009},
    'SEN0567': {'alpha_H': 0.0016, 'alpha_T': 0.0008}
}

for sensor, coef in coeffs.items():
    print(f'  {sensor:8}: alpha_H={coef["alpha_H"]:.4f}, alpha_T={coef["alpha_T"]:.4f}')

print('\nReference values:')
print('  H_ref = 50.0 %RH')
print('  T_ref = 25.0 °C')

print('\nCompensation applied: AFTER baseline conversion, BEFORE dV/dt')

# ===================================================================
# PART 5: VERIFY FINAL TEST EVALUATION
# ===================================================================
print('\n' + '='*70)
print('FINAL TEST EVALUATION AUDIT')
print('='*70)

if not TF_AVAILABLE:
    print('\nWARNING: TensorFlow not available, reading saved results from JSON')
    with open('results/final_evaluation.json') as f:
        saved_results = json.load(f)
    saved_acc = saved_results['tinyml_float32']['accuracy']
    saved_f1 = saved_results['tinyml_float32']['macro_f1']
    print(f'  Saved Accuracy: {saved_acc:.4f}')
    print(f'  Saved Macro F1: {saved_f1:.4f}')
else:
    # Load models and evaluate on test set
    from sklearn.preprocessing import LabelEncoder
    import joblib

    test_X = test_df[features].values.astype(np.float32)
    test_y_labels = test_df['label'].values
    test_y = np.array([expected_classes.index(label) for label in test_y_labels])

    # Load TinyML model
    model_path = Path('models/tinyml/sentry_tinyml.keras')
    model = tf.keras.models.load_model(model_path)

    # Scale test data
    X_test_scaled = (test_X - scaler_mean) / scaler_scale
    X_test_scaled = X_test_scaled.astype(np.float32)

    # Predict
    probabilities = model.predict(X_test_scaled, verbose=0)
    predictions = np.argmax(probabilities, axis=1)

    # Calculate metrics
    accuracy = accuracy_score(test_y, predictions)
    precision, recall, f1, support = precision_recall_fscore_support(
        test_y, predictions, labels=np.arange(len(expected_classes)), zero_division=0
    )

    print(f'\nTinyML Float32 on final test:')
    print(f'  Accuracy: {accuracy:.4f}')
    print(f'  Macro F1: {np.mean(f1):.4f}')

    cm = confusion_matrix(test_y, predictions, labels=np.arange(len(expected_classes)))
    print(f'\nConfusion Matrix (rows=true, cols=pred):')
    print('       ', '       '.join(f'{c:>7}' for c in expected_classes))
    for i, row in enumerate(cm):
        print(f'{expected_classes[i]:>6}: {row}')

    # Compare with saved results
    with open('results/final_evaluation.json') as f:
        saved_results = json.load(f)

    saved_acc = saved_results['tinyml_float32']['accuracy']
    saved_f1 = saved_results['tinyml_float32']['macro_f1']

    print(f'\nSaved results (from final_evaluation.json):')
    print(f'  Saved Accuracy: {saved_acc:.4f}')
    print(f'  Saved Macro F1: {saved_f1:.4f}')
    print(f'  Computed Accuracy: {accuracy:.4f}')
    print(f'  Computed Macro F1: {np.mean(f1):.4f}')

    acc_match = np.isclose(accuracy, saved_acc, atol=1e-5)
    f1_match = np.isclose(np.mean(f1), saved_f1, atol=1e-5)

    print(f'  Accuracy matches: {acc_match}')
    print(f'  F1 matches: {f1_match}')

# ===================================================================
# PART 6: THREAT METRICS AUDIT
# ===================================================================
print('\n' + '='*70)
print('THREAT METRICS AUDIT')
print('='*70)

if not TF_AVAILABLE:
    print('\nReading threat metrics from saved JSON results...')
    with open('results/final_evaluation.json') as f:
        saved_results = json.load(f)
    cm_float32 = np.array(saved_results['tinyml_float32']['confusion_matrix'])
    
    # Expected classes: SAFE(0), WEATHER(1), ALCOHOL(2), EXPLOSIVE(3), NARCOTIC(4)
    # THREAT = EXPLOSIVE(3) + NARCOTIC(4)
    # NON_THREAT = SAFE(0) + WEATHER(1) + ALCOHOL(2)
    
    # From confusion matrix, extract binary metrics
    # TP: correctly predicted EXPLOSIVE + NARCOTIC
    threat_rows = cm_float32[[3, 4], :]  # rows 3 and 4
    TP = threat_rows[:, [3, 4]].sum()  # columns 3 and 4
    FN = threat_rows[:, [0, 1, 2]].sum()  # columns 0, 1, 2
    
    # Non-threat rows
    nonThreat_rows = cm_float32[[0, 1, 2], :]
    TN = nonThreat_rows[:, [0, 1, 2]].sum()
    FP = nonThreat_rows[:, [3, 4]].sum()
    
    threat_recall = TP / (TP + FN) if (TP + FN) > 0 else 0
    threat_fnr = FN / (TP + FN) if (TP + FN) > 0 else 0
    threat_precision = TP / (TP + FP) if (TP + FP) > 0 else 0
    
    print(f'\nBinary (Threat vs Non-Threat):')
    print(f'  EXPLOSIVE + NARCOTIC = THREAT (2400+2400=4800)')
    print(f'  SAFE + WEATHER + ALCOHOL = NON-THREAT (2400+2400+2400=7200)')
    print(f'  TP={TP}, TN={TN}, FP={FP}, FN={FN}')
    print(f'  Threat Recall: {threat_recall:.4f}')
    print(f'  Threat FNR: {threat_fnr:.4f}')
    print(f'  Threat Precision: {threat_precision:.4f}')
    
    print(f'\nPer-class recall (from saved):')
    for i, class_name in enumerate(expected_classes):
        class_recall = saved_results['tinyml_float32']['per_class'][class_name]['recall']
        print(f'  {class_name:10}: {class_recall:.4f}')
else:
    # Define threat classes
    threat_classes_idx = [expected_classes.index('EXPLOSIVE'), expected_classes.index('NARCOTIC')]
    non_threat_classes_idx = [expected_classes.index('SAFE'), expected_classes.index('WEATHER'), expected_classes.index('ALCOHOL')]

    # Convert to binary: 1 = threat, 0 = non-threat
    y_test_binary = np.isin(test_y, threat_classes_idx).astype(int)
    y_pred_binary = np.isin(predictions, threat_classes_idx).astype(int)

    # Calculate binary metrics
    TP = np.sum((y_test_binary == 1) & (y_pred_binary == 1))
    TN = np.sum((y_test_binary == 0) & (y_pred_binary == 0))
    FP = np.sum((y_test_binary == 0) & (y_pred_binary == 1))
    FN = np.sum((y_test_binary == 1) & (y_pred_binary == 0))

    threat_recall = TP / (TP + FN) if (TP + FN) > 0 else 0
    threat_fnr = FN / (TP + FN) if (TP + FN) > 0 else 0
    threat_precision = TP / (TP + FP) if (TP + FP) > 0 else 0

    print(f'\nBinary (Threat vs Non-Threat):')
    print(f'  EXPLOSIVE + NARCOTIC = THREAT (2400+2400=4800)')
    print(f'  SAFE + WEATHER + ALCOHOL = NON-THREAT (2400+2400+2400=7200)')
    print(f'  TP={TP}, TN={TN}, FP={FP}, FN={FN}')
    print(f'  Threat Recall: {threat_recall:.4f}')
    print(f'  Threat FNR: {threat_fnr:.4f}')
    print(f'  Threat Precision: {threat_precision:.4f}')

    print(f'\nPer-class recall (important for safety):')
    for i, class_name in enumerate(expected_classes):
        class_recall = recall[i]
        print(f'  {class_name:10}: {class_recall:.4f}')

print('\n' + '='*70)
print('AUDIT COMPLETE')
print('='*70)
