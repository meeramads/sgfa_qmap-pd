#!/usr/bin/env python
"""Debug clinical feature variances."""

import sys
import numpy as np
import logging

sys.path.insert(0, '.')

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def examine_clinical_features():
    """Examine clinical features and their variances."""
    try:
        from data.qmap_pd import load_qmap_pd
        
        # Load data with basic preprocessing
        data = load_qmap_pd(
            data_dir="qMAP-PD_data",
            enable_advanced_preprocessing=False
        )
        
        # Get clinical data
        clinical_df = data['clinical']
        clinical_X = data['X_list'][3]  # Clinical is typically the 4th view
        
        print("=== Clinical Data Analysis ===")
        print(f"Clinical DataFrame shape: {clinical_df.shape}")
        print(f"Clinical X matrix shape: {clinical_X.shape}")
        print(f"View names: {data['view_names']}")
        
        # Check clinical column names
        clinical_cols = [col for col in clinical_df.columns if col != 'sid']
        print(f"\nClinical variables ({len(clinical_cols)}):")
        for i, col in enumerate(clinical_cols):
            print(f"  {i+1:2d}. {col}")
        
        # Compute variances for clinical features
        print(f"\n=== Clinical Feature Variances ===")
        clinical_data_cols = clinical_df.drop(columns=['sid']).select_dtypes(include=[np.number])
        
        for i, col in enumerate(clinical_data_cols.columns):
            values = clinical_data_cols[col].dropna()
            if len(values) > 0:
                var = np.var(values)
                std = np.std(values)
                unique_vals = len(values.unique())
                print(f"{i+1:2d}. {col:25s}: var={var:8.4f}, std={std:6.3f}, unique={unique_vals:3d}, range=({values.min():.2f}, {values.max():.2f})")
            else:
                print(f"{i+1:2d}. {col:25s}: ALL NaN")
        
        # Check if there are constant or near-constant features
        print(f"\n=== Variance Threshold Analysis ===")
        thresholds = [0.0, 0.001, 0.01, 0.1]
        
        for threshold in thresholds:
            from sklearn.feature_selection import VarianceThreshold
            selector = VarianceThreshold(threshold=threshold)
            
            # Apply to clinical data only
            clinical_matrix = clinical_data_cols.values
            selected = selector.fit_transform(clinical_matrix)
            n_selected = selected.shape[1]
            n_removed = clinical_matrix.shape[1] - n_selected
            
            print(f"Threshold {threshold:5.3f}: {n_selected:2d} features kept, {n_removed:2d} removed")
            
            if n_removed > 0:
                removed_indices = ~selector.get_support()
                removed_features = clinical_data_cols.columns[removed_indices].tolist()
                print(f"                    Removed: {removed_features}")
        
        return True
        
    except Exception as e:
        logger.error(f"Error examining clinical features: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    examine_clinical_features()