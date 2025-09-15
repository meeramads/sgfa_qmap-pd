"""Data validation and preprocessing experiments."""

import logging
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Tuple, Optional
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
import seaborn as sns
import matplotlib.pyplot as plt

from experiments.framework import ExperimentFramework, ExperimentConfig, ExperimentResult
from data.qmap_pd import load_qmap_pd
from core.utils import safe_pickle_save, safe_pickle_load
from core.config_utils import safe_get, get_output_dir, get_data_dir, ConfigAccessor

logger = logging.getLogger(__name__)


def _log_preprocessing_summary(preprocessing_info):
    """Log a concise preprocessing summary instead of full details."""
    if not preprocessing_info:
        logger.info("✅ Preprocessing info: None")
        return

    # Extract key information
    status = preprocessing_info.get('status', 'unknown')

    # Handle different strategy formats
    strategy = preprocessing_info.get('strategy', 'unknown')
    if isinstance(strategy, dict):
        strategy = strategy.get('name', 'unknown')

    # Check for preprocessing results nested structure
    if 'preprocessing_results' in preprocessing_info:
        nested_info = preprocessing_info['preprocessing_results']
        status = nested_info.get('status', status)
        strategy = nested_info.get('strategy', strategy)
        preprocessor_type = nested_info.get('preprocessor_type', 'unknown')
    else:
        preprocessor_type = preprocessing_info.get('preprocessor_type', 'unknown')

    logger.info(f"✅ Preprocessing: {status} ({preprocessor_type})")
    logger.info(f"   Strategy: {strategy}")

    # Get the actual preprocessing data (could be nested)
    data_source = preprocessing_info.get('preprocessing_results', preprocessing_info)

    # Log feature reduction if available
    if 'feature_reduction' in data_source:
        fr = data_source['feature_reduction']
        logger.info(f"   Features: {fr['total_before']:,} → {fr['total_after']:,} ({fr['reduction_ratio']:.3f} ratio)")

    # Log steps applied
    steps = data_source.get('steps_applied', [])
    if steps:
        logger.info(f"   Steps: {', '.join(steps)}")

    # Log basic shapes summary
    original_shapes = data_source.get('original_shapes', [])
    processed_shapes = data_source.get('processed_shapes', [])
    if original_shapes and processed_shapes:
        total_orig_features = sum(shape[1] for shape in original_shapes)
        total_proc_features = sum(shape[1] for shape in processed_shapes)
        logger.info(f"   Data: {len(original_shapes)} views, {total_orig_features:,} → {total_proc_features:,} features")


def run_data_validation(config):
    """Run data validation experiments with remote workstation integration."""
    logger.info("Starting Data Validation Experiments")

    try:
        # Self-contained data validation - no external framework dependencies
        import sys
        import os

        # Add project root for basic imports only
        current_file = os.path.abspath(__file__)
        project_root = os.path.dirname(os.path.dirname(current_file))  # Go up from experiments/ to project root
        if project_root not in sys.path:
            sys.path.insert(0, project_root)

        # Run basic data validation directly without framework
        logger.info("Running simplified data validation...")

        # Advanced data loading and validation with neuroimaging-aware preprocessing
        from data.preprocessing_integration import apply_preprocessing_to_pipeline
        logger.info("🧠 Using advanced neuroimaging preprocessing strategy...")
        X_list, preprocessing_info = apply_preprocessing_to_pipeline(
            config=config,
            data_dir=get_data_dir(config),
            auto_select_strategy=False,  # Don't auto-select, use our preferred strategy
            preferred_strategy="aggressive"  # Use the most advanced preprocessing
        )

        logger.info(f"✅ Data loaded: {len(X_list)} views")
        for i, X in enumerate(X_list):
            logger.info(f"   View {i}: {X.shape}")

        # Log preprocessing summary instead of full details
        _log_preprocessing_summary(preprocessing_info)
        logger.info("✅ Data validation completed successfully")
        return {'status': 'completed', 'views': len(X_list), 'shapes': [X.shape for X in X_list], 'preprocessing': preprocessing_info}

    except Exception as e:
        logger.error(f"Data validation failed: {e}")
        return None


class DataValidationExperiments:
    """Comprehensive data validation and preprocessing experiments."""
    
    def __init__(self, framework: ExperimentFramework):
        """Initialize with experiment framework."""
        self.framework = framework
    
    def run_data_quality_assessment(self, config: ExperimentConfig) -> ExperimentResult:
        """
        Run comprehensive data quality assessment.
        
        Parameters
        ----------
        config : ExperimentConfig
            Experiment configuration.
            
        Returns
        -------
        ExperimentResult : Data quality assessment results.
        """
        def data_quality_experiment(config, output_dir, **kwargs):
            """Internal experiment function."""
            results = {
                'data_summary': {},
                'quality_metrics': {},
                'preprocessing_effects': {},
                'diagnostics': {}
            }
            
            logger.info("Loading qMAP-PD data for quality assessment...")
            
            # Load raw data
            raw_data = load_qmap_pd(
                data_dir=config.data_dir,
                enable_advanced_preprocessing=False,
                **config.preprocessing_config
            )
            
            # Load preprocessed data for comparison
            preprocessed_data = load_qmap_pd(
                data_dir=config.data_dir,
                enable_advanced_preprocessing=True,
                **config.preprocessing_config
            )
            
            # Analyze data quality
            results['data_summary'] = self._analyze_data_structure(raw_data, preprocessed_data)
            results['quality_metrics'] = self._assess_data_quality(raw_data, preprocessed_data)
            results['preprocessing_effects'] = self._analyze_preprocessing_effects(raw_data, preprocessed_data)
            
            # Generate diagnostic plots
            diagnostics_dir = output_dir / "diagnostics"
            diagnostics_dir.mkdir(exist_ok=True)
            results['diagnostics'] = self._generate_data_diagnostics(
                raw_data, preprocessed_data, diagnostics_dir
            )
            
            # Save detailed data summaries
            self._save_data_summaries(raw_data, preprocessed_data, output_dir)
            
            return results
        
        return self.framework.run_experiment(config, data_quality_experiment)
    
    def _analyze_data_structure(self, raw_data: Dict, preprocessed_data: Dict) -> Dict[str, Any]:
        """Analyze basic data structure and dimensions."""
        analysis = {
            'raw_data': {},
            'preprocessed_data': {},
            'comparison': {}
        }
        
        for label, data in [('raw_data', raw_data), ('preprocessed_data', preprocessed_data)]:
            if not data or 'X_list' not in data:
                analysis[label] = {
                    'n_subjects': 0,
                    'n_views': 0,
                    'features_per_view': [],
                    'total_features': 0,
                    'clinical_variables': [],
                    'n_clinical_variables': 0,
                    'data_types': [],
                    'view_names': []
                }
                continue
                
            X_list = data['X_list']
            clinical = data.get('clinical', pd.DataFrame())
            
            analysis[label] = {
                'n_subjects': X_list[0].shape[0] if X_list else 0,
                'n_views': len(X_list),
                'features_per_view': [X.shape[1] for X in X_list],
                'total_features': sum(X.shape[1] for X in X_list),
                'clinical_variables': list(clinical.columns) if not clinical.empty else [],
                'n_clinical_variables': len(clinical.columns) if not clinical.empty else 0,
                'data_types': [str(X.dtype) for X in X_list],
                'view_names': data.get('view_names', [])
            }
        
        # Compare raw vs preprocessed
        raw_features = analysis['raw_data']['total_features']
        proc_features = analysis['preprocessed_data']['total_features']
        
        analysis['comparison'] = {
            'feature_reduction_ratio': (raw_features - proc_features) / raw_features if raw_features > 0 else 0,
            'features_removed': raw_features - proc_features,
            'subjects_unchanged': analysis['raw_data']['n_subjects'] == analysis['preprocessed_data']['n_subjects']
        }
        
        logger.info(f"Data structure analysis completed. "
                   f"Raw: {raw_features} features, Processed: {proc_features} features")
        
        return analysis
    
    def _assess_data_quality(self, raw_data: Dict, preprocessed_data: Dict) -> Dict[str, Any]:
        """Assess various data quality metrics."""
        quality_metrics = {
            'raw_data': {},
            'preprocessed_data': {},
            'improvement_metrics': {}
        }
        
        for label, data in [('raw_data', raw_data), ('preprocessed_data', preprocessed_data)]:
            if not data or 'X_list' not in data:
                quality_metrics[label] = {}
                continue
                
            X_list = data['X_list']
            metrics = {}
            
            for view_idx, X in enumerate(X_list):
                view_name = data.get('view_names', [f'view_{view_idx}'])[view_idx]
                
                # Missing data analysis
                missing_ratio = np.isnan(X).mean()
                
                # Feature variance analysis
                feature_variances = np.nanvar(X, axis=0)
                low_variance_features = np.sum(feature_variances < 1e-8)
                
                # Outlier detection (simple Z-score based)
                z_scores = np.abs((X - np.nanmean(X, axis=0)) / np.nanstd(X, axis=0))
                outlier_ratio = np.mean(z_scores > 3)
                
                # Dynamic range analysis
                feature_ranges = np.nanmax(X, axis=0) - np.nanmin(X, axis=0)
                
                # Correlation analysis
                valid_mask = ~np.isnan(X).any(axis=0)
                if np.sum(valid_mask) > 1:
                    X_valid = X[:, valid_mask]
                    corr_matrix = np.corrcoef(X_valid.T)
                    high_corr_pairs = np.sum(np.abs(corr_matrix) > 0.95) - len(corr_matrix)  # Exclude diagonal
                else:
                    high_corr_pairs = 0
                
                metrics[view_name] = {
                    'missing_data_ratio': float(missing_ratio),
                    'low_variance_features': int(low_variance_features),
                    'low_variance_ratio': float(low_variance_features / X.shape[1]),
                    'outlier_ratio': float(outlier_ratio),
                    'mean_feature_range': float(np.nanmean(feature_ranges)),
                    'std_feature_range': float(np.nanstd(feature_ranges)),
                    'highly_correlated_pairs': int(high_corr_pairs),
                    'condition_number': float(np.linalg.cond(X_valid.T @ X_valid)) if np.sum(valid_mask) > 0 else np.inf
                }
            
            quality_metrics[label] = metrics
        
        # Calculate improvement metrics
        improvement = {}
        for view_name in quality_metrics['raw_data'].keys():
            if view_name in quality_metrics['preprocessed_data']:
                raw_metrics = quality_metrics['raw_data'][view_name]
                proc_metrics = quality_metrics['preprocessed_data'][view_name]
                
                improvement[view_name] = {
                    'missing_data_improvement': raw_metrics['missing_data_ratio'] - proc_metrics['missing_data_ratio'],
                    'low_variance_reduction': raw_metrics['low_variance_features'] - proc_metrics['low_variance_features'],
                    'outlier_reduction': raw_metrics['outlier_ratio'] - proc_metrics['outlier_ratio'],
                    'condition_number_improvement': raw_metrics['condition_number'] / proc_metrics['condition_number'] if proc_metrics['condition_number'] > 0 else np.inf
                }
        
        quality_metrics['improvement_metrics'] = improvement
        
        return quality_metrics
    
    def _analyze_preprocessing_effects(self, raw_data: Dict, preprocessed_data: Dict) -> Dict[str, Any]:
        """Analyze the effects of preprocessing on data distribution and structure."""
        effects = {
            'distribution_changes': {},
            'dimensionality_reduction': {},
            'feature_selection': {},
            'normalization_effects': {}
        }
        
        if not raw_data or 'X_list' not in raw_data or not preprocessed_data or 'X_list' not in preprocessed_data:
            return effects
            
        raw_X_list = raw_data['X_list']
        proc_X_list = preprocessed_data['X_list']
        
        for view_idx, (raw_X, proc_X) in enumerate(zip(raw_X_list, proc_X_list)):
            view_name = raw_data.get('view_names', [f'view_{view_idx}'])[view_idx]
            
            # Distribution changes
            raw_mean = np.nanmean(raw_X, axis=0)
            raw_std = np.nanstd(raw_X, axis=0)
            proc_mean = np.nanmean(proc_X, axis=0)
            proc_std = np.nanstd(proc_X, axis=0)
            
            effects['distribution_changes'][view_name] = {
                'mean_shift': float(np.nanmean(np.abs(proc_mean - raw_mean[:len(proc_mean)]))),
                'std_change_ratio': float(np.nanmean(proc_std / raw_std[:len(proc_std)])) if len(proc_std) > 0 else 0,
                'skewness_change': self._calculate_skewness_change(raw_X, proc_X)
            }
            
            # Dimensionality reduction analysis
            effects['dimensionality_reduction'][view_name] = {
                'original_features': raw_X.shape[1],
                'processed_features': proc_X.shape[1],
                'reduction_ratio': (raw_X.shape[1] - proc_X.shape[1]) / raw_X.shape[1],
                'features_removed': raw_X.shape[1] - proc_X.shape[1]
            }
            
            # Feature selection analysis (if applicable)
            if 'preprocessing' in preprocessed_data:
                preprocessing_info = preprocessed_data['preprocessing']
                if 'feature_selection' in preprocessing_info:
                    effects['feature_selection'][view_name] = preprocessing_info['feature_selection'].get(view_name, {})
            
            # Normalization effects
            effects['normalization_effects'][view_name] = {
                'mean_after_processing': float(np.nanmean(proc_X)),
                'std_after_processing': float(np.nanstd(proc_X)),
                'range_after_processing': float(np.nanmax(proc_X) - np.nanmin(proc_X))
            }
        
        return effects
    
    def _calculate_skewness_change(self, raw_X: np.ndarray, proc_X: np.ndarray) -> float:
        """Calculate change in data skewness due to preprocessing."""
        from scipy.stats import skew
        
        # Calculate skewness for features present in both raw and processed data
        min_features = min(raw_X.shape[1], proc_X.shape[1])
        
        raw_skew = skew(raw_X[:, :min_features], axis=0, nan_policy='omit')
        proc_skew = skew(proc_X[:, :min_features], axis=0, nan_policy='omit')
        
        return float(np.nanmean(np.abs(proc_skew - raw_skew)))
    
    def _generate_data_diagnostics(self, raw_data: Dict, preprocessed_data: Dict, output_dir: Path) -> Dict[str, Any]:
        """Generate comprehensive diagnostic plots and analyses."""
        diagnostics = {
            'plots_generated': [],
            'statistical_tests': {},
            'visualizations': {}
        }
        
        # Feature distribution comparison plots
        self._plot_feature_distributions(raw_data, preprocessed_data, output_dir)
        diagnostics['plots_generated'].append('feature_distributions.png')
        
        # Missing data heatmaps
        self._plot_missing_data_patterns(raw_data, preprocessed_data, output_dir)
        diagnostics['plots_generated'].append('missing_data_patterns.png')
        
        # Correlation matrices
        self._plot_correlation_matrices(raw_data, preprocessed_data, output_dir)
        diagnostics['plots_generated'].append('correlation_matrices.png')
        
        # Dimensionality reduction visualization (PCA)
        pca_results = self._perform_pca_analysis(raw_data, preprocessed_data, output_dir)
        diagnostics['statistical_tests']['pca_analysis'] = pca_results
        diagnostics['plots_generated'].append('pca_visualization.png')
        
        # Quality metrics visualization
        self._plot_quality_metrics(raw_data, preprocessed_data, output_dir)
        diagnostics['plots_generated'].append('quality_metrics.png')
        
        return diagnostics
    
    def _plot_feature_distributions(self, raw_data: Dict, preprocessed_data: Dict, output_dir: Path):
        """Plot feature distribution comparisons."""
        if 'X_list' not in raw_data or 'X_list' not in preprocessed_data:
            logger.warning("Cannot plot feature distributions - missing X_list data")
            return
            
        fig, axes = plt.subplots(2, len(raw_data['X_list']), figsize=(15, 10))
        if len(raw_data['X_list']) == 1:
            axes = axes.reshape(-1, 1)
        
        for view_idx, (raw_X, proc_X) in enumerate(zip(raw_data['X_list'], preprocessed_data['X_list'])):
            view_name = raw_data.get('view_names', [f'View {view_idx}'])[view_idx]
            
            # Raw data distribution
            feature_means_raw = np.nanmean(raw_X, axis=0)
            axes[0, view_idx].hist(feature_means_raw, bins=50, alpha=0.7, label='Raw')
            axes[0, view_idx].set_title(f'{view_name} - Raw Feature Means')
            axes[0, view_idx].set_xlabel('Feature Mean Value')
            axes[0, view_idx].set_ylabel('Frequency')
            
            # Preprocessed data distribution
            feature_means_proc = np.nanmean(proc_X, axis=0)
            axes[1, view_idx].hist(feature_means_proc, bins=50, alpha=0.7, label='Processed', color='orange')
            axes[1, view_idx].set_title(f'{view_name} - Processed Feature Means')
            axes[1, view_idx].set_xlabel('Feature Mean Value')
            axes[1, view_idx].set_ylabel('Frequency')
        
        plt.tight_layout()
        plt.savefig(output_dir / 'feature_distributions.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_missing_data_patterns(self, raw_data: Dict, preprocessed_data: Dict, output_dir: Path):
        """Plot missing data patterns."""
        if 'X_list' not in raw_data or 'X_list' not in preprocessed_data:
            logger.warning("Cannot plot missing data patterns - missing X_list data")
            return
            
        fig, axes = plt.subplots(2, len(raw_data['X_list']), figsize=(15, 8))
        if len(raw_data['X_list']) == 1:
            axes = axes.reshape(-1, 1)
        
        for view_idx, (raw_X, proc_X) in enumerate(zip(raw_data['X_list'], preprocessed_data['X_list'])):
            view_name = raw_data.get('view_names', [f'View {view_idx}'])[view_idx]
            
            # Sample data for visualization (to avoid memory issues)
            sample_size = min(500, raw_X.shape[0])
            sample_features = min(100, raw_X.shape[1])
            
            raw_sample = raw_X[:sample_size, :sample_features]
            proc_sample = proc_X[:sample_size, :min(sample_features, proc_X.shape[1])]
            
            # Raw data missing pattern
            missing_raw = np.isnan(raw_sample)
            axes[0, view_idx].imshow(missing_raw.T, cmap='RdYlBu', aspect='auto')
            axes[0, view_idx].set_title(f'{view_name} - Raw Missing Data')
            axes[0, view_idx].set_xlabel('Subjects')
            axes[0, view_idx].set_ylabel('Features')
            
            # Preprocessed data missing pattern
            missing_proc = np.isnan(proc_sample)
            axes[1, view_idx].imshow(missing_proc.T, cmap='RdYlBu', aspect='auto')
            axes[1, view_idx].set_title(f'{view_name} - Processed Missing Data')
            axes[1, view_idx].set_xlabel('Subjects')
            axes[1, view_idx].set_ylabel('Features')
        
        plt.tight_layout()
        plt.savefig(output_dir / 'missing_data_patterns.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_correlation_matrices(self, raw_data: Dict, preprocessed_data: Dict, output_dir: Path):
        """Plot correlation matrices for each view."""
        if 'X_list' not in raw_data or 'X_list' not in preprocessed_data:
            logger.warning("Cannot plot correlation matrices - missing X_list data")
            return
            
        n_views = len(raw_data['X_list'])
        fig, axes = plt.subplots(2, n_views, figsize=(5*n_views, 10))
        if n_views == 1:
            axes = axes.reshape(-1, 1)
        
        for view_idx, (raw_X, proc_X) in enumerate(zip(raw_data['X_list'], preprocessed_data['X_list'])):
            view_name = raw_data.get('view_names', [f'View {view_idx}'])[view_idx]
            
            # Sample features for correlation analysis
            max_features = 50  # Limit for computational efficiency
            
            # Raw data correlation
            raw_sample = raw_X[:, :min(max_features, raw_X.shape[1])]
            valid_mask_raw = ~np.isnan(raw_sample).any(axis=0)
            if np.sum(valid_mask_raw) > 1:
                raw_corr = np.corrcoef(raw_sample[:, valid_mask_raw].T)
                im1 = axes[0, view_idx].imshow(raw_corr, cmap='RdBu', vmin=-1, vmax=1)
                axes[0, view_idx].set_title(f'{view_name} - Raw Correlations')
                plt.colorbar(im1, ax=axes[0, view_idx])
            
            # Processed data correlation
            proc_sample = proc_X[:, :min(max_features, proc_X.shape[1])]
            valid_mask_proc = ~np.isnan(proc_sample).any(axis=0)
            if np.sum(valid_mask_proc) > 1:
                proc_corr = np.corrcoef(proc_sample[:, valid_mask_proc].T)
                im2 = axes[1, view_idx].imshow(proc_corr, cmap='RdBu', vmin=-1, vmax=1)
                axes[1, view_idx].set_title(f'{view_name} - Processed Correlations')
                plt.colorbar(im2, ax=axes[1, view_idx])
        
        plt.tight_layout()
        plt.savefig(output_dir / 'correlation_matrices.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _perform_pca_analysis(self, raw_data: Dict, preprocessed_data: Dict, output_dir: Path) -> Dict[str, Any]:
        """Perform PCA analysis and visualization."""
        pca_results = {
            'raw_data': {},
            'preprocessed_data': {},
            'comparison': {}
        }
        
        if 'X_list' not in raw_data or 'X_list' not in preprocessed_data:
            logger.warning("Cannot perform PCA analysis - missing X_list data")
            return pca_results
        
        fig, axes = plt.subplots(2, len(raw_data['X_list']), figsize=(15, 10))
        if len(raw_data['X_list']) == 1:
            axes = axes.reshape(-1, 1)
        
        for view_idx, (raw_X, proc_X) in enumerate(zip(raw_data['X_list'], preprocessed_data['X_list'])):
            view_name = raw_data.get('view_names', [f'View {view_idx}'])[view_idx]
            
            # PCA on raw data
            raw_X_clean = raw_X[~np.isnan(raw_X).any(axis=1)]  # Remove subjects with any NaN
            if raw_X_clean.shape[0] > 10 and raw_X_clean.shape[1] > 2:
                pca_raw = PCA(n_components=min(10, raw_X_clean.shape[1]))
                pca_raw.fit(raw_X_clean)
                
                pca_results['raw_data'][view_name] = {
                    'explained_variance_ratio': pca_raw.explained_variance_ratio_.tolist(),
                    'cumulative_variance': np.cumsum(pca_raw.explained_variance_ratio_).tolist(),
                    'n_components_90_var': int(np.argmax(np.cumsum(pca_raw.explained_variance_ratio_) > 0.9) + 1)
                }
                
                # Plot explained variance
                axes[0, view_idx].plot(range(1, len(pca_raw.explained_variance_ratio_) + 1),
                                     np.cumsum(pca_raw.explained_variance_ratio_), 'o-')
                axes[0, view_idx].set_title(f'{view_name} - Raw Data PCA')
                axes[0, view_idx].set_xlabel('Principal Component')
                axes[0, view_idx].set_ylabel('Cumulative Explained Variance')
                axes[0, view_idx].grid(True)
            
            # PCA on processed data
            proc_X_clean = proc_X[~np.isnan(proc_X).any(axis=1)]
            if proc_X_clean.shape[0] > 10 and proc_X_clean.shape[1] > 2:
                pca_proc = PCA(n_components=min(10, proc_X_clean.shape[1]))
                pca_proc.fit(proc_X_clean)
                
                pca_results['preprocessed_data'][view_name] = {
                    'explained_variance_ratio': pca_proc.explained_variance_ratio_.tolist(),
                    'cumulative_variance': np.cumsum(pca_proc.explained_variance_ratio_).tolist(),
                    'n_components_90_var': int(np.argmax(np.cumsum(pca_proc.explained_variance_ratio_) > 0.9) + 1)
                }
                
                # Plot explained variance
                axes[1, view_idx].plot(range(1, len(pca_proc.explained_variance_ratio_) + 1),
                                     np.cumsum(pca_proc.explained_variance_ratio_), 'o-', color='orange')
                axes[1, view_idx].set_title(f'{view_name} - Processed Data PCA')
                axes[1, view_idx].set_xlabel('Principal Component')
                axes[1, view_idx].set_ylabel('Cumulative Explained Variance')
                axes[1, view_idx].grid(True)
        
        plt.tight_layout()
        plt.savefig(output_dir / 'pca_visualization.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        return pca_results
    
    def _plot_quality_metrics(self, raw_data: Dict, preprocessed_data: Dict, output_dir: Path):
        """Plot data quality metrics comparison."""
        if 'X_list' not in raw_data or 'X_list' not in preprocessed_data:
            logger.warning("Cannot plot quality metrics - missing X_list data")
            return
            
        quality_raw = self._assess_data_quality(raw_data, {})['raw_data']
        quality_proc = self._assess_data_quality({}, preprocessed_data)['preprocessed_data']
        
        metrics = ['missing_data_ratio', 'low_variance_ratio', 'outlier_ratio']
        n_views = len(raw_data['X_list'])
        
        fig, axes = plt.subplots(1, len(metrics), figsize=(15, 5))
        
        for metric_idx, metric in enumerate(metrics):
            view_names = []
            raw_values = []
            proc_values = []
            
            for view_name in quality_raw.keys():
                if view_name in quality_proc:
                    view_names.append(view_name)
                    raw_values.append(quality_raw[view_name][metric])
                    proc_values.append(quality_proc[view_name][metric])
            
            x = np.arange(len(view_names))
            width = 0.35
            
            axes[metric_idx].bar(x - width/2, raw_values, width, label='Raw', alpha=0.7)
            axes[metric_idx].bar(x + width/2, proc_values, width, label='Processed', alpha=0.7)
            
            axes[metric_idx].set_xlabel('Views')
            axes[metric_idx].set_ylabel(metric.replace('_', ' ').title())
            axes[metric_idx].set_title(f'Data Quality: {metric.replace("_", " ").title()}')
            axes[metric_idx].set_xticks(x)
            axes[metric_idx].set_xticklabels(view_names)
            axes[metric_idx].legend()
        
        plt.tight_layout()
        plt.savefig(output_dir / 'quality_metrics.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _save_data_summaries(self, raw_data: Dict, preprocessed_data: Dict, output_dir: Path):
        """Save detailed data summaries to files."""
        if 'X_list' not in raw_data or 'X_list' not in preprocessed_data:
            logger.warning("Cannot save data summaries - missing X_list data")
            return
            
        # Raw data summary
        raw_summary = self._create_detailed_summary(raw_data, "raw")
        raw_summary.to_csv(output_dir / 'raw_data_summary.csv', index=False)
        
        # Preprocessed data summary
        proc_summary = self._create_detailed_summary(preprocessed_data, "preprocessed")
        proc_summary.to_csv(output_dir / 'preprocessed_data_summary.csv', index=False)
        
        # Comparison summary
        comparison_df = self._create_comparison_summary(raw_data, preprocessed_data)
        comparison_df.to_csv(output_dir / 'data_comparison_summary.csv', index=False)
    
    def _create_detailed_summary(self, data: Dict, data_type: str) -> pd.DataFrame:
        """Create detailed summary DataFrame for data."""
        if 'X_list' not in data:
            return pd.DataFrame()
            
        summaries = []
        
        for view_idx, X in enumerate(data['X_list']):
            view_name = data.get('view_names', [f'view_{view_idx}'])[view_idx]
            
            summary = {
                'data_type': data_type,
                'view_name': view_name,
                'n_subjects': X.shape[0],
                'n_features': X.shape[1],
                'missing_ratio': np.isnan(X).mean(),
                'mean_value': np.nanmean(X),
                'std_value': np.nanstd(X),
                'min_value': np.nanmin(X),
                'max_value': np.nanmax(X),
                'dtype': str(X.dtype)
            }
            
            summaries.append(summary)
        
        return pd.DataFrame(summaries)
    
    def _create_comparison_summary(self, raw_data: Dict, preprocessed_data: Dict) -> pd.DataFrame:
        """Create comparison summary between raw and preprocessed data."""
        if 'X_list' not in raw_data or 'X_list' not in preprocessed_data:
            return pd.DataFrame()
            
        comparisons = []
        
        for view_idx, (raw_X, proc_X) in enumerate(zip(raw_data['X_list'], preprocessed_data['X_list'])):
            view_name = raw_data.get('view_names', [f'view_{view_idx}'])[view_idx]
            
            comparison = {
                'view_name': view_name,
                'features_raw': raw_X.shape[1],
                'features_processed': proc_X.shape[1],
                'feature_reduction_count': raw_X.shape[1] - proc_X.shape[1],
                'feature_reduction_ratio': (raw_X.shape[1] - proc_X.shape[1]) / raw_X.shape[1],
                'missing_data_improvement': np.isnan(raw_X).mean() - np.isnan(proc_X).mean(),
                'mean_change': np.nanmean(proc_X) - np.nanmean(raw_X[:, :proc_X.shape[1]]),
                'std_change': np.nanstd(proc_X) - np.nanstd(raw_X[:, :proc_X.shape[1]])
            }
            
            comparisons.append(comparison)
        
        return pd.DataFrame(comparisons)
    
    def run_preprocessing_comparison(self, config: ExperimentConfig) -> ExperimentResult:
        """
        Compare different preprocessing strategies.
        
        Parameters
        ----------
        config : ExperimentConfig
            Base configuration for preprocessing comparison.
            
        Returns
        -------
        ExperimentResult : Preprocessing comparison results.
        """
        def preprocessing_comparison_experiment(config, output_dir, **kwargs):
            """Internal experiment function for preprocessing comparison."""
            
            # Define different preprocessing strategies
            preprocessing_strategies = {
                'minimal': {
                    'enable_advanced_preprocessing': False,
                    'imputation_strategy': 'mean',
                    'feature_selection_method': None
                },
                'standard': {
                    'enable_advanced_preprocessing': True,
                    'imputation_strategy': 'median',
                    'feature_selection_method': 'variance',
                    'variance_threshold': 0.01
                },
                'aggressive': {
                    'enable_advanced_preprocessing': True,
                    'imputation_strategy': 'knn',
                    'feature_selection_method': 'mutual_info',
                    'n_top_features': 500,
                    'missing_threshold': 0.05
                }
            }
            
            results = {
                'strategies': {},
                'comparison': {},
                'recommendations': {}
            }
            
            strategy_data = {}
            
            for strategy_name, strategy_config in preprocessing_strategies.items():
                logger.info(f"Testing preprocessing strategy: {strategy_name}")
                
                try:
                    # Load data with this preprocessing strategy
                    data = load_qmap_pd(
                        data_dir=config.data_dir,
                        **strategy_config
                    )
                    
                    # Verify data structure
                    if not isinstance(data, dict):
                        raise ValueError(f"Expected dict from load_qmap_pd, got {type(data)}")
                    
                    if 'X_list' not in data:
                        logger.error(f"Missing 'X_list' key in data for {strategy_name}. Available keys: {list(data.keys())}")
                        raise KeyError(f"Missing 'X_list' key in data structure")
                    
                    logger.info(f"Successfully loaded data for {strategy_name}: X_list has {len(data['X_list'])} views")
                    strategy_data[strategy_name] = data
                    
                    # Analyze results of this strategy
                    analysis = self._analyze_data_structure({}, data)['preprocessed_data']
                    quality = self._assess_data_quality({}, data)['preprocessed_data']
                    
                    results['strategies'][strategy_name] = {
                        'config': strategy_config,
                        'data_structure': analysis,
                        'quality_metrics': quality,
                        'processing_time': 0  # Would need to measure this
                    }
                    
                except Exception as e:
                    logger.error(f"Failed to process strategy {strategy_name}: {e}")
                    results['strategies'][strategy_name] = {
                        'config': strategy_config,
                        'error': str(e)
                    }
            
            # Compare strategies
            results['comparison'] = self._compare_preprocessing_strategies(strategy_data)
            
            # Generate recommendations
            results['recommendations'] = self._generate_preprocessing_recommendations(results['strategies'])
            
            # Save strategy comparison plots
            self._plot_strategy_comparison(strategy_data, output_dir)
            
            return results
        
        modified_config = ExperimentConfig.from_dict(config.to_dict())
        modified_config.experiment_name = f"{config.experiment_name}_preprocessing_comparison"
        
        return self.framework.run_experiment(modified_config, preprocessing_comparison_experiment)
    
    def _compare_preprocessing_strategies(self, strategy_data: Dict) -> Dict[str, Any]:
        """Compare different preprocessing strategies."""
        comparison = {}
        
        if not strategy_data:
            logger.warning("No strategy data available for comparison")
            return comparison
        
        # Get strategy names
        strategy_names = list(strategy_data.keys())
        logger.info(f"Comparing {len(strategy_names)} preprocessing strategies: {strategy_names}")
        
        # Compare data dimensions
        comparison['data_dimensions'] = {}
        for strategy_name, data in strategy_data.items():
            try:
                if 'X_list' not in data:
                    logger.error(f"Strategy {strategy_name} missing X_list. Keys: {list(data.keys())}")
                    continue
                    
                X_list = data['X_list']
                comparison['data_dimensions'][strategy_name] = {
                    'total_features': sum(X.shape[1] for X in X_list),
                    'features_per_view': [X.shape[1] for X in X_list]
                }
            except Exception as e:
                logger.error(f"Error processing dimensions for {strategy_name}: {e}")
                continue
        
        # Compare data quality metrics
        comparison['quality_comparison'] = {}
        if strategy_data:
            # Find a strategy that has X_list to determine number of views
            valid_strategy_data = [(name, data) for name, data in strategy_data.items() if 'X_list' in data]
            
            if valid_strategy_data:
                num_views = len(valid_strategy_data[0][1]['X_list'])
                
                for view_idx in range(num_views):
                    view_name = f'view_{view_idx}'
                    comparison['quality_comparison'][view_name] = {}
                    
                    for strategy_name, data in strategy_data.items():
                        try:
                            if 'X_list' not in data:
                                continue
                                
                            X = data['X_list'][view_idx]
                            comparison['quality_comparison'][view_name][strategy_name] = {
                                'missing_ratio': float(np.isnan(X).mean()),
                                'feature_variance_mean': float(np.nanvar(X, axis=0).mean()),
                                'condition_number': float(np.linalg.cond(X.T @ X))
                            }
                        except Exception as e:
                            logger.error(f"Error computing quality metrics for {strategy_name}, view {view_idx}: {e}")
                            continue
            else:
                logger.warning("No valid strategy data found for quality comparison")
        
        return comparison
    
    def _generate_preprocessing_recommendations(self, strategy_results: Dict) -> Dict[str, Any]:
        """Generate preprocessing recommendations based on results."""
        recommendations = {
            'best_strategy': None,
            'rationale': [],
            'specific_recommendations': {}
        }
        
        # Simple scoring system (would be more sophisticated in practice)
        scores = {}
        
        for strategy_name, results in strategy_results.items():
            if 'error' in results:
                scores[strategy_name] = 0
                continue
                
            score = 0
            
            # Reward strategies that preserve more data
            data_structure = results.get('data_structure', {})
            if data_structure:
                score += data_structure.get('n_subjects', 0) * 0.1
                score += sum(data_structure.get('features_per_view', [])) * 0.001
            
            # Reward strategies with better quality metrics
            quality_metrics = results.get('quality_metrics', {})
            for view_metrics in quality_metrics.values():
                score -= view_metrics.get('missing_data_ratio', 1.0) * 100
                score -= view_metrics.get('low_variance_ratio', 1.0) * 50
                score -= view_metrics.get('outlier_ratio', 1.0) * 25
            
            scores[strategy_name] = score
        
        if scores:
            best_strategy = max(scores.keys(), key=lambda k: scores[k])
            recommendations['best_strategy'] = best_strategy
            
            recommendations['rationale'] = [
                f"Strategy '{best_strategy}' achieved the highest overall score ({scores[best_strategy]:.2f})",
                "Scoring based on data preservation, missing data handling, and feature quality"
            ]
        
        return recommendations
    
    def _plot_strategy_comparison(self, strategy_data: Dict, output_dir: Path):
        """Create comparison plots for different preprocessing strategies."""
        if not strategy_data:
            logger.warning("No strategy data available for plotting")
            return
        
        # Filter to valid strategies only
        valid_strategies = {name: data for name, data in strategy_data.items() if 'X_list' in data}
        if not valid_strategies:
            logger.warning("No valid strategy data with X_list for plotting")
            return
            
        n_strategies = len(valid_strategies)
        strategy_names = list(valid_strategies.keys())
        
        # Feature count comparison
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # 1. Total features per strategy
        total_features = []
        for strategy_name in strategy_names:
            data = valid_strategies[strategy_name]
            total_features.append(sum(X.shape[1] for X in data['X_list']))
        
        axes[0, 0].bar(strategy_names, total_features)
        axes[0, 0].set_title('Total Features by Strategy')
        axes[0, 0].set_ylabel('Number of Features')
        axes[0, 0].tick_params(axis='x', rotation=45)
        
        # 2. Missing data comparison
        if len(valid_strategies) > 0:
            first_data = list(valid_strategies.values())[0]
            n_views = len(first_data['X_list'])
            
            missing_ratios = {strategy: [] for strategy in strategy_names}
            
            for view_idx in range(n_views):
                for strategy_name in strategy_names:
                    X = valid_strategies[strategy_name]['X_list'][view_idx]
                    missing_ratios[strategy_name].append(np.isnan(X).mean())
            
            x = np.arange(n_views)
            width = 0.8 / len(strategy_names)
            
            for i, strategy_name in enumerate(strategy_names):
                axes[0, 1].bar(x + i * width, missing_ratios[strategy_name], 
                             width, label=strategy_name, alpha=0.7)
            
            axes[0, 1].set_title('Missing Data by View and Strategy')
            axes[0, 1].set_xlabel('View')
            axes[0, 1].set_ylabel('Missing Data Ratio')
            axes[0, 1].set_xticks(x + width * (len(strategy_names) - 1) / 2)
            axes[0, 1].set_xticklabels([f'View {i}' for i in range(n_views)])
            axes[0, 1].legend()
        
        # 3. Data variance comparison
        variance_data = {strategy: [] for strategy in strategy_names}
        for strategy_name in strategy_names:
            data = strategy_data[strategy_name]
            for X in data['X_list']:
                variance_data[strategy_name].append(np.nanvar(X))
        
        axes[1, 0].boxplot([variance_data[strategy] for strategy in strategy_names],
                          labels=strategy_names)
        axes[1, 0].set_title('Data Variance Distribution by Strategy')
        axes[1, 0].set_ylabel('Variance')
        axes[1, 0].tick_params(axis='x', rotation=45)
        
        # 4. Feature reduction comparison
        if 'minimal' in strategy_data:
            baseline_features = sum(X.shape[1] for X in strategy_data['minimal']['X_list'])
            reduction_ratios = []
            
            for strategy_name in strategy_names:
                current_features = sum(X.shape[1] for X in strategy_data[strategy_name]['X_list'])
                reduction_ratio = (baseline_features - current_features) / baseline_features
                reduction_ratios.append(reduction_ratio)
            
            axes[1, 1].bar(strategy_names, reduction_ratios)
            axes[1, 1].set_title('Feature Reduction vs Minimal Strategy')
            axes[1, 1].set_ylabel('Reduction Ratio')
            axes[1, 1].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.savefig(output_dir / 'preprocessing_strategy_comparison.png', 
                   dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info("Preprocessing strategy comparison plots saved")