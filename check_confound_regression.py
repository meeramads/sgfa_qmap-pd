#!/usr/bin/env python3
"""
Check if imaging data (SN region) is already confound-regressed.

If data is already regressed for age, sex, and TIV:
1. Correlations with these confounds should be near zero
2. Data should have zero or near-zero mean (residuals are centered)
3. Variance should be consistent with residual variance
"""

import numpy as np
import pandas as pd
from scipy import stats

# Load SN imaging data
print("Loading SN imaging data...")
sn_data = pd.read_csv("qMAP-PD_data/volume_matrices/volume_sn_voxels.tsv",
                       sep="\t", header=None)
print(f"SN data shape: {sn_data.shape}")

# Load clinical data
print("\nLoading clinical data...")
clinical = pd.read_csv("qMAP-PD_data/data_clinical/pd_motor_gfa_data.tsv", sep="\t")
print(f"Clinical data shape: {clinical.shape}")

# Extract confounds
age = clinical['age'].values
sex = clinical['sex'].values
tiv = clinical['tiv'].values

print("\n" + "="*60)
print("CONFOUND VARIABLE STATISTICS")
print("="*60)
print(f"Age: mean={age.mean():.2f}, std={age.std():.2f}, range=[{age.min():.2f}, {age.max():.2f}]")
print(f"Sex: unique values={np.unique(sex)}, counts={np.bincount(sex.astype(int))}")
print(f"TIV: mean={tiv.mean():.4f}, std={tiv.std():.4f}, range=[{tiv.min():.4f}, {tiv.max():.4f}]")

# Check correlations between imaging data and confounds
print("\n" + "="*60)
print("IMAGING DATA PROPERTIES")
print("="*60)

# Convert to numpy array
sn_array = sn_data.values

# Basic statistics
print(f"\nImaging data statistics:")
print(f"  Overall mean: {sn_array.mean():.6f}")
print(f"  Overall std: {sn_array.std():.6f}")
print(f"  Overall range: [{sn_array.min():.6f}, {sn_array.max():.6f}]")
print(f"  Per-voxel mean (avg across voxels): {sn_array.mean(axis=0).mean():.6f}")
print(f"  Per-subject mean (avg across subjects): {sn_array.mean(axis=1).mean():.6f}")

# Compute correlations with confounds for each voxel
print("\n" + "="*60)
print("CORRELATIONS WITH CONFOUNDS")
print("="*60)

correlations_age = []
correlations_sex = []
correlations_tiv = []

n_voxels = sn_array.shape[1]
sample_voxels = min(100, n_voxels)  # Sample first 100 voxels for detailed stats

for i in range(n_voxels):
    voxel_values = sn_array[:, i]

    # Pearson correlation with each confound
    r_age, _ = stats.pearsonr(voxel_values, age)
    r_sex, _ = stats.pearsonr(voxel_values, sex)
    r_tiv, _ = stats.pearsonr(voxel_values, tiv)

    correlations_age.append(r_age)
    correlations_sex.append(r_sex)
    correlations_tiv.append(r_tiv)

correlations_age = np.array(correlations_age)
correlations_sex = np.array(correlations_sex)
correlations_tiv = np.array(correlations_tiv)

print(f"\nCorrelations with AGE across {n_voxels} voxels:")
print(f"  Mean: {correlations_age.mean():.6f}")
print(f"  Std: {correlations_age.std():.6f}")
print(f"  Range: [{correlations_age.min():.6f}, {correlations_age.max():.6f}]")
print(f"  Median: {np.median(correlations_age):.6f}")
print(f"  % with |r| < 0.05: {100 * np.mean(np.abs(correlations_age) < 0.05):.1f}%")
print(f"  % with |r| < 0.10: {100 * np.mean(np.abs(correlations_age) < 0.10):.1f}%")

print(f"\nCorrelations with SEX across {n_voxels} voxels:")
print(f"  Mean: {correlations_sex.mean():.6f}")
print(f"  Std: {correlations_sex.std():.6f}")
print(f"  Range: [{correlations_sex.min():.6f}, {correlations_sex.max():.6f}]")
print(f"  Median: {np.median(correlations_sex):.6f}")
print(f"  % with |r| < 0.05: {100 * np.mean(np.abs(correlations_sex) < 0.05):.1f}%")
print(f"  % with |r| < 0.10: {100 * np.mean(np.abs(correlations_sex) < 0.10):.1f}%")

print(f"\nCorrelations with TIV across {n_voxels} voxels:")
print(f"  Mean: {correlations_tiv.mean():.6f}")
print(f"  Std: {correlations_tiv.std():.6f}")
print(f"  Range: [{correlations_tiv.min():.6f}, {correlations_tiv.max():.6f}]")
print(f"  Median: {np.median(correlations_tiv):.6f}")
print(f"  % with |r| < 0.05: {100 * np.mean(np.abs(correlations_tiv) < 0.05):.1f}%")
print(f"  % with |r| < 0.10: {100 * np.mean(np.abs(correlations_tiv) < 0.10):.1f}%")

# Multiple regression check
print("\n" + "="*60)
print("MULTIPLE REGRESSION TEST (sample voxels)")
print("="*60)

from sklearn.linear_model import LinearRegression

# Test on a sample of voxels
r2_values = []
for i in range(sample_voxels):
    voxel_values = sn_array[:, i]

    # Fit linear model: voxel ~ age + sex + tiv
    X_confounds = np.column_stack([age, sex, tiv])
    lr = LinearRegression()
    lr.fit(X_confounds, voxel_values)

    # R² = variance explained by confounds
    r2 = lr.score(X_confounds, voxel_values)
    r2_values.append(r2)

r2_values = np.array(r2_values)

print(f"\nR² (variance explained by age+sex+tiv) for first {sample_voxels} voxels:")
print(f"  Mean R²: {r2_values.mean():.6f}")
print(f"  Std R²: {r2_values.std():.6f}")
print(f"  Range: [{r2_values.min():.6f}, {r2_values.max():.6f}]")
print(f"  Median R²: {np.median(r2_values):.6f}")
print(f"  % with R² < 0.01: {100 * np.mean(r2_values < 0.01):.1f}%")
print(f"  % with R² < 0.05: {100 * np.mean(r2_values < 0.05):.1f}%")

# INTERPRETATION
print("\n" + "="*60)
print("INTERPRETATION")
print("="*60)

if correlations_age.mean() < 0.01 and correlations_sex.mean() < 0.01 and correlations_tiv.mean() < 0.01:
    if r2_values.mean() < 0.01:
        print("\n✅ DATA IS LIKELY ALREADY CONFOUND-REGRESSED")
        print("   - Very low correlations with confounds")
        print("   - Very low R² from multiple regression")
        print("   → Use --drop-confounds-from-clinical-only age sex tiv")
    else:
        print("\n⚠️  UNCLEAR - Correlations low but R² moderate")
        print("   → Inspect further or consult with supervisor")
else:
    print("\n❌ DATA IS LIKELY NOT CONFOUND-REGRESSED")
    print("   - Substantial correlations with confounds detected")
    print("   → Use --regress-confounds age sex tiv")

print("\n" + "="*60)
