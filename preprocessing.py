"""
Preprocessing module for Sparse Bayesian Group Factor Analysis.

This module can be used in two ways:

1. As a Python module (imported by other scripts):
   from preprocessing import AdvancedPreprocessor, cross_validate_source_combinations

2. As a standalone CLI tool (run directly):
   python preprocessing.py --data_dir qMAP-PD_data --inspect_only
   python preprocessing.py --data_dir qMAP-PD_data --compare_methods
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Optional, Any
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.feature_selection import VarianceThreshold, SelectKBest, f_classif, mutual_info_classif
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import logging

# For standalone CLI functionality
import argparse
import os
import sys
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns

logging.basicConfig(level=logging.INFO)

# == CORE PREPROCESSING CLASSES (for import) ==

class AdvancedPreprocessor:
    """
    Preprocessing steps.
    
    Key features:
    - Multiple imputation strategies
    - Variance-based and statistical feature selection
    - Cross-validation for optimal feature/source selection
    - Robust scaling with outlier handling
    """
    
    def __init__(self, 
                 missing_threshold: float = 0.1,
                 variance_threshold: float = 0.0,
                 n_top_features: Optional[int] = None,
                 imputation_strategy: str = 'median',
                 feature_selection_method: str = 'variance',
                 random_state: int = 42):
        """
        Parameters
        ----------
        missing_threshold : float
            Remove features with > this fraction of missing values
        variance_threshold : float
            Remove features with variance below this threshold
        n_top_features : int, optional
            Number of top features to select (if None, uses variance_threshold)
        imputation_strategy : str
            'median', 'mean', 'knn', or 'iterative'
        feature_selection_method : str
            'variance', 'statistical', 'mutual_info', or 'combined'
        """
        self.missing_threshold = missing_threshold
        self.variance_threshold = variance_threshold
        self.n_top_features = n_top_features
        self.imputation_strategy = imputation_strategy
        self.feature_selection_method = feature_selection_method
        self.random_state = random_state
        
        # Storage for fitted transformers
        self.imputers_ = {}
        self.scalers_ = {}
        self.feature_selectors_ = {}
        self.selected_features_ = {}
        
    def fit_transform_imputation(self, X: np.ndarray, view_name: str) -> np.ndarray:
        """
        Fit and apply imputation strategy to data.
        """
        logging.info(f"Applying {self.imputation_strategy} imputation to {view_name}")
        
        # Check missing data percentage
        missing_pct = np.isnan(X).mean(axis=0)
        features_to_drop = missing_pct > self.missing_threshold
        
        if np.any(features_to_drop):
            logging.warning(f"Dropping {np.sum(features_to_drop)} features with >{self.missing_threshold*100}% missing data")
            X = X[:, ~features_to_drop]
            
        if self.imputation_strategy == 'median':
            imputer = SimpleImputer(strategy='median')
        elif self.imputation_strategy == 'mean':
            imputer = SimpleImputer(strategy='mean')
        elif self.imputation_strategy == 'knn':
            # KNN imputation - more sophisticated than simple median
            imputer = KNNImputer(n_neighbors=5)
        elif self.imputation_strategy == 'iterative':
            from sklearn.experimental import enable_iterative_imputer
            from sklearn.impute import IterativeImputer
            imputer = IterativeImputer(random_state=self.random_state)
        else:
            raise ValueError(f"Unknown imputation strategy: {self.imputation_strategy}")
            
        X_imputed = imputer.fit_transform(X)
        self.imputers_[view_name] = imputer
        
        # Store feature mask for consistent application during transform
        self.selected_features_[f"{view_name}_missing_mask"] = ~features_to_drop
        
        return X_imputed
    
    def fit_transform_feature_selection(self, X: np.ndarray, view_name: str, 
                                      y: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Apply feature selection following both papers' methodologies.
        
        Ferreira et al.: "We reduced the dimensionality to the 500 genes with the highest 
        average variance over the data views"
        
        Bunte et al.: Cross-validation to select most predictive features
        """
        logging.info(f"Applying {self.feature_selection_method} feature selection to {view_name}")
        
        if self.feature_selection_method == 'variance':
            return self._variance_based_selection(X, view_name)
        elif self.feature_selection_method == 'statistical' and y is not None:
            return self._statistical_selection(X, view_name, y)
        elif self.feature_selection_method == 'mutual_info' and y is not None:
            return self._mutual_info_selection(X, view_name, y)
        elif self.feature_selection_method == 'combined' and y is not None:
            return self._combined_selection(X, view_name, y)
        else:
            logging.warning(f"Unsupervised feature selection for {view_name} - using variance")
            return self._variance_based_selection(X, view_name)
    
    def _variance_based_selection(self, X: np.ndarray, view_name: str) -> np.ndarray:
        """Variance-based feature selection as in Ferreira et al."""
        if self.n_top_features is not None:
            # Select top-k features by variance
            variances = np.var(X, axis=0)
            top_indices = np.argsort(variances)[::-1][:self.n_top_features]
            X_selected = X[:, top_indices]
            self.selected_features_[f"{view_name}_variance_indices"] = top_indices
        else:
            # Use variance threshold
            selector = VarianceThreshold(threshold=self.variance_threshold)
            X_selected = selector.fit_transform(X)
            self.feature_selectors_[f"{view_name}_variance"] = selector
            
        logging.info(f"Selected {X_selected.shape[1]} features for {view_name} (was {X.shape[1]})")
        return X_selected
    
    def _statistical_selection(self, X: np.ndarray, view_name: str, y: np.ndarray) -> np.ndarray:
        """Statistical feature selection using F-test or correlation."""
        n_features = self.n_top_features or min(500, X.shape[1] // 2)
        
        if len(np.unique(y)) > 10:  # Regression
            # Use correlation for continuous targets
            correlations = np.abs([np.corrcoef(X[:, i], y)[0, 1] for i in range(X.shape[1])])
            correlations = np.nan_to_num(correlations)  # Handle NaN correlations
            top_indices = np.argsort(correlations)[::-1][:n_features]
            X_selected = X[:, top_indices]
            self.selected_features_[f"{view_name}_corr_indices"] = top_indices
        else:  # Classification
            selector = SelectKBest(score_func=f_classif, k=n_features)
            X_selected = selector.fit_transform(X, y)
            self.feature_selectors_[f"{view_name}_ftest"] = selector
            
        logging.info(f"Selected {X_selected.shape[1]} features for {view_name} using statistical selection")
        return X_selected
    
    def _mutual_info_selection(self, X: np.ndarray, view_name: str, y: np.ndarray) -> np.ndarray:
        """Mutual information-based feature selection."""
        n_features = self.n_top_features or min(500, X.shape[1] // 2)
        
        selector = SelectKBest(score_func=mutual_info_classif, k=n_features)
        X_selected = selector.fit_transform(X, y)
        self.feature_selectors_[f"{view_name}_mutinfo"] = selector
        
        logging.info(f"Selected {X_selected.shape[1]} features for {view_name} using mutual information")
        return X_selected
    
    def _combined_selection(self, X: np.ndarray, view_name: str, y: np.ndarray) -> np.ndarray:
        """Combined variance + statistical selection."""
        # First apply variance threshold
        X_var = self._variance_based_selection(X, f"{view_name}_temp")
        
        # Then apply statistical selection on reduced set
        if y is not None:
            X_selected = self._statistical_selection(X_var, view_name, y)
        else:
            X_selected = X_var
            
        return X_selected
    
    def fit_transform_scaling(self, X: np.ndarray, view_name: str, 
                            robust: bool = True) -> np.ndarray:
        """
        Apply standardization.
        """
        logging.info(f"Applying {'robust' if robust else 'standard'} scaling to {view_name}")
        
        if robust:
            from sklearn.preprocessing import RobustScaler
            scaler = RobustScaler()
        else:
            scaler = StandardScaler()
        
        X_scaled = scaler.fit_transform(X)
        self.scalers_[view_name] = scaler
        
        return X_scaled
    
    def fit_transform(self, X_list: List[np.ndarray], view_names: List[str],
                     y: Optional[np.ndarray] = None) -> List[np.ndarray]:
        """
        Complete preprocessing pipeline.
        
        Parameters
        ----------
        X_list : List of data matrices
        view_names : List of view names
        y : Target variable (optional, for supervised feature selection)
        
        Returns
        -------
        List of preprocessed data matrices
        """
        logging.info("Starting complete preprocessing pipeline")
        
        X_processed = []
        
        for X, view_name in zip(X_list, view_names):
            logging.info(f"Processing view: {view_name} (shape: {X.shape})")
            
            # Step 1: Handle missing data
            X_imputed = self.fit_transform_imputation(X, view_name)
            
            # Step 2: Feature selection
            X_selected = self.fit_transform_feature_selection(X_imputed, view_name, y)
            
            # Step 3: Scaling
            X_scaled = self.fit_transform_scaling(X_selected, view_name)
            
            X_processed.append(X_scaled)
            
            logging.info(f"Final shape for {view_name}: {X_scaled.shape}")
        
        return X_processed
    
    def transform(self, X_list: List[np.ndarray], view_names: List[str]) -> List[np.ndarray]:
        """Transform new data using fitted preprocessors."""
        X_processed = []
        
        for X, view_name in zip(X_list, view_names):
            # Apply stored transformations
            
            # 1. Missing data handling
            if f"{view_name}_missing_mask" in self.selected_features_:
                mask = self.selected_features_[f"{view_name}_missing_mask"]
                X = X[:, mask]
                
            X_imputed = self.imputers_[view_name].transform(X)
            
            # 2. Feature selection
            if f"{view_name}_variance_indices" in self.selected_features_:
                indices = self.selected_features_[f"{view_name}_variance_indices"]
                X_selected = X_imputed[:, indices]
            elif f"{view_name}_variance" in self.feature_selectors_:
                X_selected = self.feature_selectors_[f"{view_name}_variance"].transform(X_imputed)
            else:
                X_selected = X_imputed
                
            # 3. Scaling
            X_scaled = self.scalers_[view_name].transform(X_selected)
            
            X_processed.append(X_scaled)
            
        return X_processed

def cross_validate_source_combinations(X_list: List[np.ndarray], 
                                     view_names: List[str],
                                     y: np.ndarray,
                                     cv_folds: int = 7) -> Dict[str, float]:
    """
    Cross-validate different combinations of data sources following Bunte et al.
    """
    from itertools import combinations
    from sklearn.linear_model import Ridge
    from sklearn.metrics import mean_squared_error
    from scipy.stats import pearsonr, spearmanr
    
    results = {}
    
    # Test all combinations of views
    for r in range(1, len(view_names) + 1):
        for combo in combinations(range(len(view_names)), r):
            combo_names = [view_names[i] for i in combo]
            combo_name = ", ".join(combo_names)
            
            # Concatenate selected views
            X_combo = np.concatenate([X_list[i] for i in combo], axis=1)
            
            # Cross-validation
            model = Ridge(alpha=1.0)
            cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42) if len(np.unique(y)) < 10 else cv_folds
            
            scores = []
            for train_idx, test_idx in cv.split(X_combo, y) if hasattr(cv, 'split') else [(None, None)] * cv_folds:
                if train_idx is None:  # Simple k-fold for regression
                    fold_scores = cross_val_score(model, X_combo, y, cv=cv_folds, 
                                                scoring='neg_mean_squared_error')
                    rmse_scores = np.sqrt(-fold_scores)
                    scores.extend(rmse_scores)
                    break
                else:
                    X_train, X_test = X_combo[train_idx], X_combo[test_idx]
                    y_train, y_test = y[train_idx], y[test_idx]
                    
                    model.fit(X_train, y_train)
                    y_pred = model.predict(X_test)
                    
                    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
                    scores.append(rmse)
            
            results[combo_name] = {
                'rmse_mean': np.mean(scores),
                'rmse_std': np.std(scores),
                'n_features': X_combo.shape[1]
            }
    
    return results

def optimize_preprocessing_parameters(X_list: List[np.ndarray], 
                                    view_names: List[str],
                                    y: Optional[np.ndarray] = None,
                                    param_grid: Optional[Dict] = None) -> Dict:
    """
    Optimize preprocessing parameters using cross-validation.
    
    Tests different combinations of:
    - Imputation strategies
    - Feature selection methods
    - Number of top features
    """
    if param_grid is None:
        param_grid = {
            'imputation_strategy': ['median', 'mean', 'knn'],
            'feature_selection_method': ['variance', 'statistical', 'combined'],
            'n_top_features': [100, 200, 500, None],
            'variance_threshold': [0.0, 0.01, 0.1]
        }
    
    from itertools import product
    
    best_score = float('inf')
    best_params = {}
    results = []
    
    # Generate parameter combinations
    keys = list(param_grid.keys())
    for values in product(*[param_grid[key] for key in keys]):
        params = dict(zip(keys, values))
        
        # Skip incompatible combinations
        if params['n_top_features'] is not None and params['variance_threshold'] > 0:
            continue
            
        try:
            preprocessor = AdvancedPreprocessor(**params)
            X_processed = preprocessor.fit_transform(X_list, view_names, y)
            
            # Simple evaluation metric
            if y is not None:
                # Use total explained variance as a proxy for quality
                from sklearn.decomposition import PCA
                X_combined = np.concatenate(X_processed, axis=1)
                pca = PCA(n_components=min(10, X_combined.shape[1], X_combined.shape[0]))
                pca.fit(X_combined)
                score = -pca.explained_variance_ratio_.sum()  # Negative for minimization
            else:
                # Use reconstruction error as quality metric
                score = sum([np.mean(X**2) for X in X_processed])
            
            results.append({
                'params': params,
                'score': score,
                'n_features_total': sum([X.shape[1] for X in X_processed])
            })
            
            if score < best_score:
                best_score = score
                best_params = params
                
        except Exception as e:
            logging.warning(f"Failed with params {params}: {e}")
            continue
    
    return {
        'best_params': best_params,
        'best_score': best_score,
        'all_results': results
    }

# == STANDALONE CLI FUNCTIONALITY ==

class PreprocessingInspector:
    """Class for inspecting preprocessing pipeline behavior."""
    
    def __init__(self, data_dir: str, **load_kwargs):
        self.data_dir = Path(data_dir)
        self.load_kwargs = load_kwargs
        self.raw_data = None
        self.preprocessed_data = None
        
    def load_raw_data(self):
        """Load raw data for inspection."""
        logging.info("Loading raw data...")
        
        # Import here to avoid dependency issues when used as module
        from loader_qmap_pd import load_qmap_pd
        
        self.raw_data = load_qmap_pd(
            data_dir=str(self.data_dir),
            enable_advanced_preprocessing=False,  # No advanced preprocessing
            **self.load_kwargs
        )
        
        logging.info(f"Loaded data: {len(self.raw_data['X_list'])} views")
        for i, (view_name, X) in enumerate(zip(self.raw_data['view_names'], self.raw_data['X_list'])):
            logging.info(f"  {view_name}: {X.shape} (N={X.shape[0]}, D={X.shape[1]})")
            
    def inspect_raw_data(self):
        """Inspect characteristics of raw data."""
        if self.raw_data is None:
            self.load_raw_data()
            
        logging.info("=== RAW DATA INSPECTION ===")
        
        for view_name, X in zip(self.raw_data['view_names'], self.raw_data['X_list']):
            logging.info(f"\n--- {view_name.upper()} ---")
            
            # Basic statistics
            logging.info(f"Shape: {X.shape}")
            logging.info(f"Data type: {X.dtype}")
            
            # Missing data
            missing_pct = np.isnan(X).sum() / X.size * 100
            logging.info(f"Missing data: {missing_pct:.2f}%")
            
            if missing_pct > 0:
                missing_per_feature = np.isnan(X).mean(axis=0)
                high_missing = np.sum(missing_per_feature > 0.1)
                logging.info(f"Features with >10% missing: {high_missing}/{X.shape[1]}")
            
            # Variance statistics
            if not np.any(np.isnan(X)):
                variances = np.var(X, axis=0)
                zero_var_features = np.sum(variances < 1e-10)
                low_var_features = np.sum(variances < 0.01)
                logging.info(f"Zero variance features: {zero_var_features}/{X.shape[1]}")
                logging.info(f"Low variance features (< 0.01): {low_var_features}/{X.shape[1]}")
                logging.info(f"Variance range: [{np.min(variances):.6f}, {np.max(variances):.2f}]")
            
            # Value ranges
            if not np.any(np.isnan(X)):
                logging.info(f"Value range: [{np.min(X):.3f}, {np.max(X):.3f}]")
                logging.info(f"Mean absolute value: {np.mean(np.abs(X)):.3f}")
                
    def create_inspection_plots(self, output_dir: str = "preprocessing_inspection"):
        """Create plots for data inspection."""
        if self.raw_data is None:
            self.load_raw_data()
            
        output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=True)
        
        logging.info(f"Creating inspection plots in {output_dir}")
        
        for view_idx, (view_name, X) in enumerate(zip(self.raw_data['view_names'], self.raw_data['X_list'])):
            
            fig, axes = plt.subplots(2, 2, figsize=(12, 10))
            fig.suptitle(f'{view_name.title()} - Data Inspection', fontsize=16)
            
            # 1. Missing data pattern
            if np.any(np.isnan(X)):
                missing_pattern = np.isnan(X).astype(int)
                im1 = axes[0,0].imshow(missing_pattern.T, aspect='auto', cmap='Reds')
                axes[0,0].set_title('Missing Data Pattern')
                axes[0,0].set_xlabel('Subjects')
                axes[0,0].set_ylabel('Features')
                plt.colorbar(im1, ax=axes[0,0])
            else:
                axes[0,0].text(0.5, 0.5, 'No Missing Data', ha='center', va='center', 
                              transform=axes[0,0].transAxes, fontsize=14)
                axes[0,0].set_title('Missing Data Pattern')
            
            # 2. Feature variance distribution
            if not np.any(np.isnan(X)):
                variances = np.var(X, axis=0)
                axes[0,1].hist(np.log10(variances + 1e-10), bins=50, alpha=0.7, color='skyblue')
                axes[0,1].set_title('Feature Variance Distribution')
                axes[0,1].set_xlabel('Log10(Variance)')
                axes[0,1].set_ylabel('Count')
                axes[0,1].axvline(np.log10(0.01), color='red', linestyle='--', 
                                 label='Low variance threshold')
                axes[0,1].legend()
            
            # 3. Data distribution sample
            sample_features = min(5, X.shape[1])
            feature_indices = np.random.choice(X.shape[1], sample_features, replace=False)
            
            for i, feat_idx in enumerate(feature_indices):
                if not np.any(np.isnan(X[:, feat_idx])):
                    axes[1,0].hist(X[:, feat_idx], bins=30, alpha=0.6, 
                                  label=f'Feature {feat_idx}')
            axes[1,0].set_title('Sample Feature Distributions')
            axes[1,0].set_xlabel('Value')
            axes[1,0].set_ylabel('Count')
            axes[1,0].legend()
            
            # 4. Correlation matrix sample
            if X.shape[1] > 1 and not np.any(np.isnan(X)):
                sample_size = min(50, X.shape[1])
                sample_indices = np.random.choice(X.shape[1], sample_size, replace=False)
                X_sample = X[:, sample_indices]
                corr_matrix = np.corrcoef(X_sample.T)
                
                im4 = axes[1,1].imshow(corr_matrix, cmap='RdBu_r', vmin=-1, vmax=1)
                axes[1,1].set_title('Feature Correlation Sample')
                plt.colorbar(im4, ax=axes[1,1])
            
            plt.tight_layout()
            plt.savefig(output_dir / f'{view_name}_inspection.png', dpi=150, bbox_inches='tight')
            plt.close()
            
        logging.info("Inspection plots saved")
    
    def test_preprocessing_methods(self, methods_to_test: List[str] = None):
        """Test different preprocessing methods and compare results."""
        if self.raw_data is None:
            self.load_raw_data()
            
        if methods_to_test is None:
            methods_to_test = ['median', 'mean', 'knn']
        
        logging.info("=== TESTING PREPROCESSING METHODS ===")
        
        results = {}
        
        for method in methods_to_test:
            logging.info(f"\nTesting imputation method: {method}")
            
            try:
                # Test preprocessing
                preprocessor = AdvancedPreprocessor(
                    imputation_strategy=method,
                    feature_selection_method='variance',
                    n_top_features=None,
                    variance_threshold=0.01
                )
                
                X_processed = preprocessor.fit_transform(
                    self.raw_data['X_list'], 
                    self.raw_data['view_names']
                )
                
                # Collect results
                method_results = {
                    'processed_shapes': [X.shape for X in X_processed],
                    'feature_reduction': {}
                }
                
                for i, view_name in enumerate(self.raw_data['view_names']):
                    original_features = self.raw_data['X_list'][i].shape[1]
                    processed_features = X_processed[i].shape[1]
                    
                    method_results['feature_reduction'][view_name] = {
                        'original': original_features,
                        'processed': processed_features,
                        'reduction_ratio': processed_features / original_features
                    }
                
                results[method] = method_results
                logging.info(f"  ✓ {method} succeeded")
                
            except Exception as e:
                logging.error(f"  ✗ {method} failed: {e}")
                results[method] = {'error': str(e)}
        
        # Print comparison
        logging.info("\n=== PREPROCESSING METHOD COMPARISON ===")
        
        for view_name in self.raw_data['view_names']:
            logging.info(f"\n--- {view_name} ---")
            logging.info("Method\t\tOriginal\tProcessed\tRetention")
            
            for method, result in results.items():
                if 'error' not in result:
                    view_result = result['feature_reduction'][view_name]
                    logging.info(f"{method:12s}\t{view_result['original']:8d}\t{view_result['processed']:8d}\t{view_result['reduction_ratio']:8.2%}")
                else:
                    logging.info(f"{method:12s}\tFAILED")
        
        return results
    
    def optimize_preprocessing_pipeline(self, target_variable: str = None):
        """Test preprocessing parameter optimization."""
        if self.raw_data is None:
            self.load_raw_data()
            
        logging.info("=== OPTIMIZING PREPROCESSING PARAMETERS ===")
        
        # Extract target if specified
        y = None
        if target_variable and 'clinical' in self.raw_data:
            clinical_df = self.raw_data['clinical']
            if target_variable in clinical_df.columns:
                y = clinical_df[target_variable].values
                logging.info(f"Using {target_variable} as target for optimization")
            else:
                logging.warning(f"Target variable {target_variable} not found in clinical data")
                logging.info(f"Available columns: {list(clinical_df.columns)}")
        
        # Run optimization
        try:
            optimization_results = optimize_preprocessing_parameters(
                self.raw_data['X_list'], 
                self.raw_data['view_names'],
                y
            )
            
            logging.info("Optimization completed!")
            logging.info(f"Best parameters: {optimization_results['best_params']}")
            logging.info(f"Best score: {optimization_results['best_score']:.4f}")
            
            # Show top 5 parameter combinations
            sorted_results = sorted(optimization_results['all_results'], 
                                  key=lambda x: x['score'])[:5]
            
            logging.info("\nTop 5 parameter combinations:")
            for i, result in enumerate(sorted_results, 1):
                logging.info(f"{i}. Score: {result['score']:.4f}")
                logging.info(f"   Params: {result['params']}")
                logging.info(f"   Features: {result['n_features_total']}")
            
            return optimization_results
            
        except Exception as e:
            logging.error(f"Optimization failed: {e}")
            return None

# == CLI MAIN FUNCTION ==

def main():
    """Main function for standalone CLI usage."""
    parser = argparse.ArgumentParser(description="Preprocessing Module - Standalone Mode")
    
    # Data arguments
    parser.add_argument("--data_dir", type=str, default="qMAP-PD_data")
    parser.add_argument("--clinical_rel", type=str, default="data_clinical/pd_motor_gfa_data.tsv")
    parser.add_argument("--volumes_rel", type=str, default="volume_matrices")
    parser.add_argument("--id_col", type=str, default="sid")
    parser.add_argument("--roi_views", action="store_true")
    
    # Inspection modes
    parser.add_argument("--inspect_only", action="store_true",
                       help="Only inspect raw data without preprocessing")
    parser.add_argument("--create_plots", action="store_true",
                       help="Create inspection plots")
    parser.add_argument("--compare_methods", action="store_true", 
                       help="Compare different preprocessing methods")
    parser.add_argument("--optimize_params", action="store_true",
                       help="Optimize preprocessing parameters")
    
    # Preprocessing parameters (for testing specific configurations)
    parser.add_argument("--imputation_strategy", type=str, default='median',
                       choices=['median', 'mean', 'knn', 'iterative'])
    parser.add_argument("--feature_selection", type=str, default='variance',
                       choices=['variance', 'statistical', 'mutual_info', 'combined'])
    parser.add_argument("--n_top_features", type=int, default=None)
    parser.add_argument("--target_variable", type=str, default=None)
    
    # Output
    parser.add_argument("--output_dir", type=str, default="preprocessing_inspection")
    
    args = parser.parse_args()
    
    # Create inspector
    inspector = PreprocessingInspector(
        data_dir=args.data_dir,
        clinical_rel=args.clinical_rel,
        volumes_rel=args.volumes_rel,
        imaging_as_single_view=not args.roi_views,
        id_col=args.id_col
    )
    
    # Run requested analyses
    if args.inspect_only:
        inspector.inspect_raw_data()
        
        if args.create_plots:
            inspector.create_inspection_plots(args.output_dir)
            
    elif args.compare_methods:
        inspector.inspect_raw_data()
        inspector.test_preprocessing_methods()
        
        if args.create_plots:
            inspector.create_inspection_plots(args.output_dir)
            
    elif args.optimize_params:
        inspector.inspect_raw_data()
        inspector.optimize_preprocessing_pipeline(args.target_variable)
        
    else:
        # Test specific preprocessing configuration
        logging.info("=== TESTING SPECIFIC PREPROCESSING CONFIGURATION ===")
        inspector.inspect_raw_data()
        
        # Apply preprocessing with specified parameters
        preprocessor = AdvancedPreprocessor(
            imputation_strategy=args.imputation_strategy,
            feature_selection_method=args.feature_selection,
            n_top_features=args.n_top_features,
        )
        
        # Extract target if specified
        y = None
        if args.target_variable and 'clinical' in inspector.raw_data:
            clinical_df = inspector.raw_data['clinical']
            if args.target_variable in clinical_df.columns:
                y = clinical_df[args.target_variable].values
                logging.info(f"Using {args.target_variable} as target")
        
        # Process data
        X_processed = preprocessor.fit_transform(
            inspector.raw_data['X_list'],
            inspector.raw_data['view_names'],
            y
        )
        
        # Show results
        logging.info("\n=== PREPROCESSING RESULTS ===")
        for i, view_name in enumerate(inspector.raw_data['view_names']):
            orig_shape = inspector.raw_data['X_list'][i].shape
            proc_shape = X_processed[i].shape
            reduction = proc_shape[1] / orig_shape[1]
            
            logging.info(f"{view_name}: {orig_shape} → {proc_shape} ({reduction:.1%} features retained)")
        
        if args.create_plots:
            inspector.create_inspection_plots(args.output_dir)

# == SCRIPT ENTRY POINT ==

if __name__ == "__main__":
    # Runs when the script is executed directly
    main()
else:
    # Runs when the module is imported
    logging.info("Preprocessing module imported successfully")
