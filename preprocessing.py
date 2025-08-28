# preprocessing.py
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

logging.basicConfig(level=logging.INFO)

class AdvancedPreprocessor:
    """
    Preprocessing steps in alignment with Ferreira et al. and Bunte et al. methodologies.
    
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
