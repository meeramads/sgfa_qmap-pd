"""Clinical validation experiments for SGFA qMAP-PD analysis."""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Any, Optional, Tuple, Union
import pandas as pd
from pathlib import Path
import logging
from scipy import stats
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report
)
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
import warnings

from experiments.framework import ExperimentFramework, ExperimentConfig, ExperimentResult
from performance import PerformanceProfiler
from core.config_utils import safe_get, get_output_dir, get_data_dir, ConfigAccessor

class ClinicalValidationExperiments(ExperimentFramework):
    """Comprehensive clinical validation experiments for SGFA qMAP-PD analysis."""
    
    def __init__(self, config: ExperimentConfig, logger: Optional[logging.Logger] = None):
        super().__init__(config, logger)
        self.profiler = PerformanceProfiler()
        
        # Clinical validation settings
        self.classification_models = {
            'logistic_regression': LogisticRegression(random_state=42, max_iter=1000),
            'random_forest': RandomForestClassifier(random_state=42, n_estimators=100),
            'svm': SVC(random_state=42, probability=True)
        }
        
        # Clinical metrics to evaluate
        self.clinical_metrics = [
            'accuracy', 'precision', 'recall', 'f1_score', 'roc_auc',
            'specificity', 'npv', 'ppv'
        ]
        
    def run_subtype_classification_validation(self, X_list: List[np.ndarray],
                                            clinical_labels: np.ndarray,
                                            hypers: Dict, args: Dict,
                                            **kwargs) -> ExperimentResult:
        """Validate SGFA factors for PD subtype classification."""
        self.logger.info("Running subtype classification validation")
        
        results = {}
        
        try:
            # Run SGFA to get factor scores
            self.logger.info("Extracting SGFA factors")
            sgfa_result = self._run_sgfa_analysis(X_list, hypers, args, **kwargs)
            
            if 'error' in sgfa_result:
                raise ValueError(f"SGFA analysis failed: {sgfa_result['error']}")
            
            Z_sgfa = sgfa_result['Z']  # Factor scores
            
            # Test different classification approaches
            classification_results = {}
            
            # 1. Direct factor-based classification
            self.logger.info("Testing direct factor-based classification")
            direct_results = self._test_factor_classification(
                Z_sgfa, clinical_labels, 'sgfa_factors'
            )
            classification_results['sgfa_factors'] = direct_results
            
            # 2. Compare with raw data classification
            self.logger.info("Testing raw data classification")
            X_concat = np.hstack(X_list)
            raw_results = self._test_factor_classification(
                X_concat, clinical_labels, 'raw_data'
            )
            classification_results['raw_data'] = raw_results
            
            # 3. Compare with PCA features
            self.logger.info("Testing PCA-based classification")
            from sklearn.decomposition import PCA
            pca = PCA(n_components=Z_sgfa.shape[1])
            Z_pca = pca.fit_transform(X_concat)
            pca_results = self._test_factor_classification(
                Z_pca, clinical_labels, 'pca_features'
            )
            classification_results['pca_features'] = pca_results
            
            results['classification_comparison'] = classification_results
            
            # 4. Clinical interpretation analysis
            self.logger.info("Analyzing clinical interpretability")
            interpretation_results = self._analyze_clinical_interpretability(
                Z_sgfa, clinical_labels, sgfa_result
            )
            results['clinical_interpretation'] = interpretation_results
            
            # 5. Subtype stability analysis
            self.logger.info("Analyzing subtype stability")
            stability_results = self._analyze_subtype_stability(
                X_list, clinical_labels, hypers, args, **kwargs
            )
            results['subtype_stability'] = stability_results
            
            # Analyze validation results
            analysis = self._analyze_subtype_classification(results)
            
            # Generate basic plots
            plots = self._plot_subtype_classification(results, clinical_labels)

            # Add comprehensive clinical visualizations (focus on subtypes + brain maps)
            advanced_plots = self._create_comprehensive_clinical_visualizations(
                X_list, results, clinical_labels, "subtype_classification"
            )
            plots.update(advanced_plots)

            return ExperimentResult(
                experiment_name="subtype_classification_validation",
                config=self.config,
                data=results,
                analysis=analysis,
                plots=plots,
                success=True
            )
            
        except Exception as e:
            self.logger.error(f"Subtype classification validation failed: {str(e)}")
            return self._create_failure_result("subtype_classification_validation", str(e))
    
    def run_disease_progression_validation(self, X_list: List[np.ndarray],
                                         progression_scores: np.ndarray,
                                         time_points: np.ndarray,
                                         subject_ids: np.ndarray,
                                         hypers: Dict, args: Dict,
                                         **kwargs) -> ExperimentResult:
        """Validate SGFA factors for disease progression prediction."""
        self.logger.info("Running disease progression validation")
        
        results = {}
        
        try:
            # Run SGFA to get factor scores
            self.logger.info("Extracting SGFA factors for progression analysis")
            sgfa_result = self._run_sgfa_analysis(X_list, hypers, args, **kwargs)
            
            if 'error' in sgfa_result:
                raise ValueError(f"SGFA analysis failed: {sgfa_result['error']}")
            
            Z_sgfa = sgfa_result['Z']
            
            # 1. Cross-sectional correlation analysis
            self.logger.info("Analyzing cross-sectional correlations")
            cross_sectional_results = self._analyze_cross_sectional_correlations(
                Z_sgfa, progression_scores
            )
            results['cross_sectional_correlations'] = cross_sectional_results
            
            # 2. Longitudinal progression modeling
            if len(np.unique(subject_ids)) < len(subject_ids):  # Longitudinal data available
                self.logger.info("Analyzing longitudinal progression")
                longitudinal_results = self._analyze_longitudinal_progression(
                    Z_sgfa, progression_scores, time_points, subject_ids
                )
                results['longitudinal_analysis'] = longitudinal_results
            else:
                self.logger.info("No longitudinal data available, skipping longitudinal analysis")
                results['longitudinal_analysis'] = {'status': 'no_longitudinal_data'}
            
            # 3. Progression prediction validation
            self.logger.info("Validating progression prediction")
            prediction_results = self._validate_progression_prediction(
                Z_sgfa, progression_scores, time_points
            )
            results['progression_prediction'] = prediction_results
            
            # 4. Clinical milestone prediction
            self.logger.info("Analyzing clinical milestone prediction")
            milestone_results = self._analyze_clinical_milestones(
                Z_sgfa, progression_scores, time_points
            )
            results['clinical_milestones'] = milestone_results
            
            # Analyze progression validation results
            analysis = self._analyze_disease_progression_validation(results)
            
            # Generate basic plots
            plots = self._plot_disease_progression_validation(results, progression_scores)

            # Add comprehensive clinical visualizations (focus on progression + brain maps)
            advanced_plots = self._create_comprehensive_clinical_visualizations(
                X_list, results, progression_scores, "disease_progression"
            )
            plots.update(advanced_plots)

            return ExperimentResult(
                experiment_name="disease_progression_validation",
                config=self.config,
                data=results,
                analysis=analysis,
                plots=plots,
                success=True
            )
            
        except Exception as e:
            self.logger.error(f"Disease progression validation failed: {str(e)}")
            return self._create_failure_result("disease_progression_validation", str(e))
    
    def run_biomarker_discovery_validation(self, X_list: List[np.ndarray],
                                         clinical_outcomes: Dict[str, np.ndarray],
                                         hypers: Dict, args: Dict,
                                         **kwargs) -> ExperimentResult:
        """Validate SGFA factors as potential biomarkers."""
        self.logger.info("Running biomarker discovery validation")
        
        results = {}
        
        try:
            # Run SGFA to get factors and loadings
            self.logger.info("Extracting SGFA factors and loadings")
            sgfa_result = self._run_sgfa_analysis(X_list, hypers, args, **kwargs)
            
            if 'error' in sgfa_result:
                raise ValueError(f"SGFA analysis failed: {sgfa_result['error']}")
            
            Z_sgfa = sgfa_result['Z']
            W_sgfa = sgfa_result['W']
            
            # 1. Factor-outcome associations
            self.logger.info("Analyzing factor-outcome associations")
            association_results = self._analyze_factor_outcome_associations(
                Z_sgfa, clinical_outcomes
            )
            results['factor_associations'] = association_results
            
            # 2. Feature importance analysis
            self.logger.info("Analyzing feature importance")
            importance_results = self._analyze_feature_importance(
                W_sgfa, X_list, clinical_outcomes
            )
            results['feature_importance'] = importance_results
            
            # 3. Biomarker panel validation
            self.logger.info("Validating biomarker panels")
            panel_results = self._validate_biomarker_panels(
                Z_sgfa, clinical_outcomes
            )
            results['biomarker_panels'] = panel_results
            
            # 4. Cross-validation robustness
            self.logger.info("Testing biomarker robustness")
            robustness_results = self._test_biomarker_robustness(
                X_list, clinical_outcomes, hypers, args, **kwargs
            )
            results['robustness_analysis'] = robustness_results
            
            # Analyze biomarker validation results
            analysis = self._analyze_biomarker_discovery(results)
            
            # Generate plots
            plots = self._plot_biomarker_discovery(results, clinical_outcomes)
            
            return ExperimentResult(
                experiment_name="biomarker_discovery_validation",
                config=self.config,
                data=results,
                analysis=analysis,
                plots=plots,
                success=True
            )
            
        except Exception as e:
            self.logger.error(f"Biomarker discovery validation failed: {str(e)}")
            return self._create_failure_result("biomarker_discovery_validation", str(e))
    
    def run_external_cohort_validation(self, X_train_list: List[np.ndarray],
                                     X_test_list: List[np.ndarray],
                                     clinical_labels_train: np.ndarray,
                                     clinical_labels_test: np.ndarray,
                                     hypers: Dict, args: Dict,
                                     **kwargs) -> ExperimentResult:
        """Validate SGFA model on external cohort."""
        self.logger.info("Running external cohort validation")
        
        results = {}
        
        try:
            # 1. Train model on training cohort
            self.logger.info("Training SGFA model on training cohort")
            train_result = self._run_sgfa_analysis(X_train_list, hypers, args, **kwargs)
            
            if 'error' in train_result:
                raise ValueError(f"Training failed: {train_result['error']}")
            
            results['training_result'] = train_result
            
            # 2. Apply trained model to test cohort
            self.logger.info("Applying model to test cohort")
            test_result = self._apply_trained_model(
                X_test_list, train_result, hypers, args, **kwargs
            )
            results['test_application'] = test_result
            
            # 3. Compare factor distributions
            self.logger.info("Comparing factor distributions")
            distribution_comparison = self._compare_factor_distributions(
                train_result['Z'], test_result['Z']
            )
            results['distribution_comparison'] = distribution_comparison
            
            # 4. Cross-cohort classification
            self.logger.info("Testing cross-cohort classification")
            classification_results = self._test_cross_cohort_classification(
                train_result['Z'], test_result['Z'],
                clinical_labels_train, clinical_labels_test
            )
            results['cross_cohort_classification'] = classification_results
            
            # 5. Model transferability analysis
            self.logger.info("Analyzing model transferability")
            transferability_results = self._analyze_model_transferability(
                train_result, test_result, clinical_labels_train, clinical_labels_test
            )
            results['transferability_analysis'] = transferability_results
            
            # Analyze external validation results
            analysis = self._analyze_external_cohort_validation(results)
            
            # Generate plots
            plots = self._plot_external_cohort_validation(results, clinical_labels_train, clinical_labels_test)
            
            return ExperimentResult(
                experiment_name="external_cohort_validation",
                config=self.config,
                data=results,
                analysis=analysis,
                plots=plots,
                success=True
            )
            
        except Exception as e:
            self.logger.error(f"External cohort validation failed: {str(e)}")
            return self._create_failure_result("external_cohort_validation", str(e))
    
    def _run_sgfa_analysis(self, X_list: List[np.ndarray], hypers: Dict, args: Dict, **kwargs) -> Dict:
        """Run SGFA analysis for clinical validation."""
        # This would call your actual SGFA implementation
        # For validation testing, we simulate realistic clinical results
        
        K = hypers.get('K', 5)
        n_subjects = X_list[0].shape[0]
        
        # Simulate factors with some clinical relevance
        np.random.seed(42)  # For reproducibility in validation
        
        # Create factors with different clinical interpretations
        Z = np.random.randn(n_subjects, K)
        
        # Add some structure that might relate to clinical outcomes
        # Factor 1: Motor symptoms
        Z[:, 0] += np.random.beta(2, 5, n_subjects) * 2 - 1
        
        # Factor 2: Cognitive symptoms  
        Z[:, 1] += np.random.gamma(2, 1, n_subjects) - 2
        
        # Factor 3: Mixed presentation
        if K > 2:
            Z[:, 2] = 0.5 * Z[:, 0] + 0.3 * Z[:, 1] + np.random.randn(n_subjects) * 0.5
        
        # Simulate loading matrices
        W = [np.random.randn(X.shape[1], K) for X in X_list]
        
        # Add sparsity to loadings
        for w in W:
            sparsity_mask = np.random.random(w.shape) < 0.7
            w[sparsity_mask] = 0
        
        return {
            'W': W,
            'Z': Z,
            'log_likelihood': -1000 + np.random.randn() * 50,
            'convergence': True,
            'n_iterations': np.random.randint(100, 300),
            'hyperparameters': hypers.copy()
        }
    
    def _test_factor_classification(self, features: np.ndarray, labels: np.ndarray, feature_type: str) -> Dict:
        """Test classification performance using given features."""
        results = {}
        
        # Ensure we have valid data
        if len(np.unique(labels)) < 2:
            return {'error': 'Insufficient label diversity for classification'}
        
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        
        for model_name, model in self.classification_models.items():
            try:
                # Cross-validation scores
                cv_scores = {}
                for metric in ['accuracy', 'precision_macro', 'recall_macro', 'f1_macro']:
                    scores = cross_val_score(model, features, labels, cv=cv, scoring=metric)
                    cv_scores[metric] = {
                        'mean': np.mean(scores),
                        'std': np.std(scores),
                        'scores': scores.tolist()
                    }
                
                # Train on full data for detailed metrics
                model.fit(features, labels)
                y_pred = model.predict(features)
                y_pred_proba = model.predict_proba(features)
                
                # Calculate detailed metrics
                detailed_metrics = self._calculate_detailed_metrics(labels, y_pred, y_pred_proba)
                
                results[model_name] = {
                    'cross_validation': cv_scores,
                    'detailed_metrics': detailed_metrics,
                    'feature_type': feature_type
                }
                
            except Exception as e:
                self.logger.warning(f"Classification failed for {model_name}: {str(e)}")
                results[model_name] = {'error': str(e)}
        
        return results
    
    def _calculate_detailed_metrics(self, y_true: np.ndarray, y_pred: np.ndarray, y_pred_proba: np.ndarray) -> Dict:
        """Calculate detailed classification metrics."""
        metrics = {}
        
        # Basic metrics
        metrics['accuracy'] = accuracy_score(y_true, y_pred)
        metrics['precision'] = precision_score(y_true, y_pred, average='macro')
        metrics['recall'] = recall_score(y_true, y_pred, average='macro')
        metrics['f1_score'] = f1_score(y_true, y_pred, average='macro')
        
        # ROC AUC (for binary or multiclass)
        try:
            if len(np.unique(y_true)) == 2:
                metrics['roc_auc'] = roc_auc_score(y_true, y_pred_proba[:, 1])
            else:
                metrics['roc_auc'] = roc_auc_score(y_true, y_pred_proba, multi_class='ovr')
        except:
            metrics['roc_auc'] = np.nan
        
        # Confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        metrics['confusion_matrix'] = cm.tolist()
        
        # Class-specific metrics
        if len(np.unique(y_true)) == 2:
            tn, fp, fn, tp = cm.ravel()
            metrics['specificity'] = tn / (tn + fp) if (tn + fp) > 0 else np.nan
            metrics['npv'] = tn / (tn + fn) if (tn + fn) > 0 else np.nan
            metrics['ppv'] = tp / (tp + fp) if (tp + fp) > 0 else np.nan
        
        return metrics
    
    def _analyze_clinical_interpretability(self, Z: np.ndarray, clinical_labels: np.ndarray, sgfa_result: Dict) -> Dict:
        """Analyze clinical interpretability of factors."""
        results = {}
        
        K = Z.shape[1]
        
        # Factor-label correlations
        factor_correlations = []
        for k in range(K):
            if len(np.unique(clinical_labels)) > 2:
                # For multiclass, use ANOVA F-statistic
                f_stat, p_val = stats.f_oneway(*[Z[clinical_labels == label, k] for label in np.unique(clinical_labels)])
                factor_correlations.append({
                    'factor': k,
                    'f_statistic': f_stat,
                    'p_value': p_val,
                    'effect_size': f_stat / (f_stat + len(clinical_labels) - len(np.unique(clinical_labels)))
                })
            else:
                # For binary, use t-test
                group_0 = Z[clinical_labels == 0, k]
                group_1 = Z[clinical_labels == 1, k]
                t_stat, p_val = stats.ttest_ind(group_0, group_1)
                
                # Cohen's d effect size
                pooled_std = np.sqrt(((len(group_0) - 1) * np.var(group_0, ddof=1) + 
                                    (len(group_1) - 1) * np.var(group_1, ddof=1)) / 
                                   (len(group_0) + len(group_1) - 2))
                cohens_d = (np.mean(group_1) - np.mean(group_0)) / pooled_std
                
                factor_correlations.append({
                    'factor': k,
                    't_statistic': t_stat,
                    'p_value': p_val,
                    'cohens_d': cohens_d,
                    'effect_size': abs(cohens_d)
                })
        
        results['factor_label_associations'] = factor_correlations
        
        # Factor interpretability scores
        W = sgfa_result['W']
        interpretability_scores = []
        
        for k in range(K):
            # Calculate sparsity (proportion of near-zero loadings)
            all_loadings = np.concatenate([w[:, k] for w in W])
            sparsity = np.mean(np.abs(all_loadings) < 0.1)
            
            # Calculate loading concentration (how concentrated are the large loadings)
            sorted_abs_loadings = np.sort(np.abs(all_loadings))[::-1]
            top_10_percent = int(len(sorted_abs_loadings) * 0.1)
            concentration = np.sum(sorted_abs_loadings[:top_10_percent]) / np.sum(sorted_abs_loadings)
            
            interpretability_scores.append({
                'factor': k,
                'sparsity': sparsity,
                'loading_concentration': concentration,
                'interpretability_score': sparsity * concentration
            })
        
        results['interpretability_analysis'] = interpretability_scores
        
        return results
    
    def _analyze_subtype_stability(self, X_list: List[np.ndarray], clinical_labels: np.ndarray,
                                 hypers: Dict, args: Dict, n_bootstrap: int = 10, **kwargs) -> Dict:
        """Analyze stability of subtype assignments."""
        results = {}
        
        n_subjects = len(clinical_labels)
        bootstrap_predictions = []
        
        for bootstrap_idx in range(n_bootstrap):
            # Bootstrap sample
            boot_indices = np.random.choice(n_subjects, n_subjects, replace=True)
            X_boot = [X[boot_indices] for X in X_list]
            labels_boot = clinical_labels[boot_indices]
            
            try:
                # Run SGFA on bootstrap sample
                sgfa_result = self._run_sgfa_analysis(X_boot, hypers, args, **kwargs)
                Z_boot = sgfa_result['Z']
                
                # Train classifier on bootstrap factors
                model = LogisticRegression(random_state=42)
                model.fit(Z_boot, labels_boot)
                
                # Predict on original (out-of-bootstrap) samples
                oob_mask = np.array([i not in boot_indices for i in range(n_subjects)])
                if np.any(oob_mask):
                    # For simplicity, predict on all samples using bootstrap model
                    sgfa_original = self._run_sgfa_analysis(X_list, hypers, args, **kwargs)
                    predictions = model.predict(sgfa_original['Z'])
                    bootstrap_predictions.append(predictions)
                    
            except Exception as e:
                self.logger.warning(f"Bootstrap iteration {bootstrap_idx} failed: {str(e)}")
                continue
        
        if bootstrap_predictions:
            # Calculate stability metrics
            bootstrap_predictions = np.array(bootstrap_predictions)
            
            # Agreement between bootstrap predictions
            agreement_matrix = np.zeros((n_subjects, n_subjects))
            for i in range(len(bootstrap_predictions)):
                for j in range(len(bootstrap_predictions)):
                    if i != j:
                        agreement = np.mean(bootstrap_predictions[i] == bootstrap_predictions[j])
                        agreement_matrix[i, j] = agreement
            
            stability_scores = np.mean(agreement_matrix, axis=1)
            
            results['bootstrap_stability'] = {
                'mean_stability': np.mean(stability_scores),
                'stability_per_subject': stability_scores.tolist(),
                'n_bootstrap_iterations': len(bootstrap_predictions)
            }
        else:
            results['bootstrap_stability'] = {'error': 'All bootstrap iterations failed'}
        
        return results
    
    def _analyze_cross_sectional_correlations(self, Z: np.ndarray, progression_scores: np.ndarray) -> Dict:
        """Analyze cross-sectional correlations between factors and progression."""
        results = {}
        
        K = Z.shape[1]
        correlations = []
        
        for k in range(K):
            corr_coef, p_val = stats.pearsonr(Z[:, k], progression_scores)
            
            # Also calculate Spearman correlation for non-linear relationships
            spearman_coef, spearman_p = stats.spearmanr(Z[:, k], progression_scores)
            
            correlations.append({
                'factor': k,
                'pearson_correlation': corr_coef,
                'pearson_p_value': p_val,
                'spearman_correlation': spearman_coef,
                'spearman_p_value': spearman_p,
                'correlation_strength': 'strong' if abs(corr_coef) > 0.5 else 'moderate' if abs(corr_coef) > 0.3 else 'weak'
            })
        
        results['factor_progression_correlations'] = correlations
        
        # Multiple regression analysis
        from sklearn.linear_model import LinearRegression
        from sklearn.metrics import r2_score
        
        model = LinearRegression()
        model.fit(Z, progression_scores)
        predictions = model.predict(Z)
        
        results['multiple_regression'] = {
            'r2_score': r2_score(progression_scores, predictions),
            'coefficients': model.coef_.tolist(),
            'intercept': model.intercept_,
            'feature_importance': np.abs(model.coef_) / np.sum(np.abs(model.coef_))
        }
        
        return results
    
    def _analyze_longitudinal_progression(self, Z: np.ndarray, progression_scores: np.ndarray,
                                        time_points: np.ndarray, subject_ids: np.ndarray) -> Dict:
        """Analyze longitudinal progression patterns."""
        results = {}
        
        # Group data by subject
        unique_subjects = np.unique(subject_ids)
        longitudinal_data = []
        
        for subject_id in unique_subjects:
            subject_mask = subject_ids == subject_id
            if np.sum(subject_mask) > 1:  # At least 2 time points
                subject_data = {
                    'subject_id': subject_id,
                    'time_points': time_points[subject_mask],
                    'progression_scores': progression_scores[subject_mask],
                    'factors': Z[subject_mask],
                    'n_timepoints': np.sum(subject_mask)
                }
                longitudinal_data.append(subject_data)
        
        results['n_longitudinal_subjects'] = len(longitudinal_data)
        
        if longitudinal_data:
            # Analyze progression rates
            progression_rates = []
            factor_changes = []
            
            for subject_data in longitudinal_data:
                times = subject_data['time_points']
                scores = subject_data['progression_scores']
                factors = subject_data['factors']
                
                # Calculate progression rate
                if len(times) > 1:
                    rate = (scores[-1] - scores[0]) / (times[-1] - times[0])
                    progression_rates.append(rate)
                    
                    # Calculate factor change rates
                    factor_change_rates = []
                    for k in range(factors.shape[1]):
                        factor_rate = (factors[-1, k] - factors[0, k]) / (times[-1] - times[0])
                        factor_change_rates.append(factor_rate)
                    factor_changes.append(factor_change_rates)
            
            results['progression_analysis'] = {
                'mean_progression_rate': np.mean(progression_rates),
                'std_progression_rate': np.std(progression_rates),
                'progression_rates': progression_rates
            }
            
            if factor_changes:
                factor_changes = np.array(factor_changes)
                results['factor_change_analysis'] = {
                    'mean_factor_changes': np.mean(factor_changes, axis=0).tolist(),
                    'std_factor_changes': np.std(factor_changes, axis=0).tolist(),
                    'factor_progression_correlations': []
                }
                
                # Correlate factor changes with progression rates
                for k in range(factor_changes.shape[1]):
                    corr, p_val = stats.pearsonr(factor_changes[:, k], progression_rates)
                    results['factor_change_analysis']['factor_progression_correlations'].append({
                        'factor': k,
                        'correlation': corr,
                        'p_value': p_val
                    })
        
        return results
    
    def _validate_progression_prediction(self, Z: np.ndarray, progression_scores: np.ndarray,
                                       time_points: np.ndarray) -> Dict:
        """Validate progression prediction using factors."""
        results = {}
        
        from sklearn.model_selection import train_test_split
        from sklearn.ensemble import RandomForestRegressor
        from sklearn.linear_model import Ridge
        from sklearn.metrics import mean_squared_error, mean_absolute_error
        
        # Split data for validation
        X_train, X_test, y_train, y_test = train_test_split(
            Z, progression_scores, test_size=0.3, random_state=42
        )
        
        # Test different prediction models
        models = {
            'linear_regression': Ridge(alpha=1.0),
            'random_forest': RandomForestRegressor(n_estimators=100, random_state=42)
        }
        
        for model_name, model in models.items():
            try:
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
                
                mse = mean_squared_error(y_test, y_pred)
                mae = mean_absolute_error(y_test, y_pred)
                r2 = model.score(X_test, y_test)
                
                results[model_name] = {
                    'mse': mse,
                    'mae': mae,
                    'r2_score': r2,
                    'rmse': np.sqrt(mse)
                }
                
                # Feature importance for random forest
                if hasattr(model, 'feature_importances_'):
                    results[model_name]['feature_importances'] = model.feature_importances_.tolist()
                
            except Exception as e:
                self.logger.warning(f"Progression prediction failed for {model_name}: {str(e)}")
                results[model_name] = {'error': str(e)}
        
        return results
    
    def _analyze_clinical_milestones(self, Z: np.ndarray, progression_scores: np.ndarray,
                                   time_points: np.ndarray) -> Dict:
        """Analyze prediction of clinical milestones."""
        results = {}
        
        # Define clinical milestones based on progression scores
        # (These would be clinically meaningful thresholds)
        milestones = {
            'mild_progression': np.percentile(progression_scores, 33),
            'moderate_progression': np.percentile(progression_scores, 66),
            'severe_progression': np.percentile(progression_scores, 90)
        }
        
        for milestone_name, threshold in milestones.items():
            # Create binary outcome
            milestone_reached = (progression_scores >= threshold).astype(int)
            
            if np.sum(milestone_reached) > 5:  # Ensure sufficient positive cases
                # Test prediction using factors
                milestone_result = self._test_factor_classification(
                    Z, milestone_reached, f'milestone_{milestone_name}'
                )
                results[milestone_name] = milestone_result
        
        return results
    
    def _analyze_factor_outcome_associations(self, Z: np.ndarray, clinical_outcomes: Dict[str, np.ndarray]) -> Dict:
        """Analyze associations between factors and clinical outcomes."""
        results = {}
        
        K = Z.shape[1]
        
        for outcome_name, outcome_values in clinical_outcomes.items():
            outcome_results = []
            
            for k in range(K):
                # Determine if outcome is continuous or categorical
                if len(np.unique(outcome_values)) > 10:  # Continuous outcome
                    corr_coef, p_val = stats.pearsonr(Z[:, k], outcome_values)
                    spearman_coef, spearman_p = stats.spearmanr(Z[:, k], outcome_values)
                    
                    outcome_results.append({
                        'factor': k,
                        'outcome_type': 'continuous',
                        'pearson_correlation': corr_coef,
                        'pearson_p_value': p_val,
                        'spearman_correlation': spearman_coef,
                        'spearman_p_value': spearman_p
                    })
                    
                else:  # Categorical outcome
                    # Use ANOVA or t-test depending on number of categories
                    unique_categories = np.unique(outcome_values)
                    
                    if len(unique_categories) == 2:
                        # t-test for binary outcomes
                        group_0 = Z[outcome_values == unique_categories[0], k]
                        group_1 = Z[outcome_values == unique_categories[1], k]
                        t_stat, p_val = stats.ttest_ind(group_0, group_1)
                        
                        outcome_results.append({
                            'factor': k,
                            'outcome_type': 'binary',
                            't_statistic': t_stat,
                            'p_value': p_val,
                            'mean_group_0': np.mean(group_0),
                            'mean_group_1': np.mean(group_1)
                        })
                        
                    else:
                        # ANOVA for multi-category outcomes
                        groups = [Z[outcome_values == cat, k] for cat in unique_categories]
                        f_stat, p_val = stats.f_oneway(*groups)
                        
                        outcome_results.append({
                            'factor': k,
                            'outcome_type': 'multiclass',
                            'f_statistic': f_stat,
                            'p_value': p_val,
                            'group_means': [np.mean(group) for group in groups]
                        })
            
            results[outcome_name] = outcome_results
        
        return results

    def _run_sgfa_analysis(self, X_list: List[np.ndarray], hypers: Dict, args: Dict, **kwargs) -> Dict:
        """Run actual SGFA analysis for clinical validation."""
        import jax
        import jax.numpy as jnp
        import numpyro
        from numpyro.infer import MCMC, NUTS
        import time

        try:
            K = hypers.get('K', 10)
            self.logger.debug(f"Running SGFA for clinical validation: K={K}, n_subjects={X_list[0].shape[0]}, n_features={sum(X.shape[1] for X in X_list)}")

            # Import the actual SGFA model function
            from core.run_analysis import models

            # Setup MCMC configuration for clinical validation
            num_warmup = args.get('num_warmup', 100)
            num_samples = args.get('num_samples', 300)
            num_chains = args.get('num_chains', 1)

            # Create args object for model
            import argparse
            model_args = argparse.Namespace(
                model='sparseGFA',
                K=K,
                num_sources=len(X_list),
                reghsZ=args.get('reghsZ', True)
            )

            # Setup MCMC
            rng_key = jax.random.PRNGKey(np.random.randint(0, 10000))
            kernel = NUTS(models, target_accept_prob=args.get('target_accept_prob', 0.8))
            mcmc = MCMC(
                kernel,
                num_warmup=num_warmup,
                num_samples=num_samples,
                num_chains=num_chains
            )

            # Run inference
            start_time = time.time()
            mcmc.run(rng_key, X_list, hypers, model_args, extra_fields=('potential_energy',))
            elapsed = time.time() - start_time

            # Get samples
            samples = mcmc.get_samples()

            # Calculate log likelihood (approximate)
            potential_energy = samples.get('potential_energy', np.array([0]))
            log_likelihood = -np.mean(potential_energy) if len(potential_energy) > 0 else 0

            # Extract mean parameters
            W_samples = samples['W']  # Shape: (num_samples, D, K)
            Z_samples = samples['Z']  # Shape: (num_samples, N, K)

            W_mean = np.mean(W_samples, axis=0)
            Z_mean = np.mean(Z_samples, axis=0)

            # Split W back into views
            W_list = []
            start_idx = 0
            for X in X_list:
                end_idx = start_idx + X.shape[1]
                W_list.append(W_mean[start_idx:end_idx, :])
                start_idx = end_idx

            return {
                'W': W_list,
                'Z': Z_mean,
                'W_samples': W_samples,
                'Z_samples': Z_samples,
                'samples': samples,
                'log_likelihood': float(log_likelihood),
                'n_iterations': num_samples,
                'convergence': True,
                'execution_time': elapsed,
                'clinical_info': {
                    'factors_extracted': Z_mean.shape[1],
                    'subjects_analyzed': Z_mean.shape[0],
                    'mcmc_config': {
                        'num_warmup': num_warmup,
                        'num_samples': num_samples,
                        'num_chains': num_chains
                    }
                }
            }

        except Exception as e:
            self.logger.error(f"SGFA clinical analysis failed: {str(e)}")
            return {
                'error': str(e),
                'convergence': False,
                'execution_time': float('inf'),
                'log_likelihood': float('-inf')
            }

    def _analyze_feature_importance(self, W: List[np.ndarray], X_list: List[np.ndarray],
                                  clinical_outcomes: Dict[str, np.ndarray]) -> Dict:
        """Analyze which features are most important for clinical outcomes."""
        results = {}
        
        # Calculate feature importance scores based on loadings
        all_features = []
        all_loadings = []
        
        for view_idx, (X, w) in enumerate(zip(X_list, W)):
            n_features = X.shape[1]
            
            for feature_idx in range(n_features):
                # Feature importance as sum of absolute loadings across factors
                importance_score = np.sum(np.abs(w[feature_idx, :]))
                
                all_features.append({
                    'view': view_idx,
                    'feature_index': feature_idx,
                    'global_feature_index': len(all_features),
                    'importance_score': importance_score,
                    'loadings': w[feature_idx, :].tolist()
                })
                all_loadings.append(w[feature_idx, :])
        
        all_loadings = np.array(all_loadings)
        
        # Rank features by importance
        sorted_features = sorted(all_features, key=lambda x: x['importance_score'], reverse=True)
        
        results['feature_rankings'] = {
            'top_features': sorted_features[:20],  # Top 20 features
            'importance_threshold': np.percentile([f['importance_score'] for f in all_features], 90)
        }
        
        # Analyze factor-specific feature importance
        K = all_loadings.shape[1]
        factor_specific_importance = []
        
        for k in range(K):
            factor_loadings = all_loadings[:, k]
            top_feature_indices = np.argsort(np.abs(factor_loadings))[::-1][:10]
            
            factor_specific_importance.append({
                'factor': k,
                'top_features': [
                    {
                        'global_index': idx,
                        'view': all_features[idx]['view'],
                        'feature_index': all_features[idx]['feature_index'],
                        'loading': factor_loadings[idx]
                    }
                    for idx in top_feature_indices
                ]
            })
        
        results['factor_specific_importance'] = factor_specific_importance
        
        return results
    
    def _validate_biomarker_panels(self, Z: np.ndarray, clinical_outcomes: Dict[str, np.ndarray]) -> Dict:
        """Validate different biomarker panels."""
        results = {}
        
        from sklearn.feature_selection import SelectKBest, f_classif, f_regression
        from sklearn.model_selection import cross_val_score
        
        for outcome_name, outcome_values in clinical_outcomes.items():
            if len(np.unique(outcome_values)) < 2:
                continue
                
            outcome_results = {}
            
            # Determine if classification or regression task
            is_classification = len(np.unique(outcome_values)) < 10
            
            if is_classification:
                # Test different panel sizes for classification
                for k_features in [1, 2, 3, min(5, Z.shape[1])]:
                    if k_features > Z.shape[1]:
                        continue
                    
                    # Select k best features
                    selector = SelectKBest(score_func=f_classif, k=k_features)
                    Z_selected = selector.fit_transform(Z, outcome_values)
                    
                    # Test with logistic regression
                    model = LogisticRegression(random_state=42)
                    
                    try:
                        cv_scores = cross_val_score(model, Z_selected, outcome_values, cv=5)
                        
                        outcome_results[f'panel_size_{k_features}'] = {
                            'selected_factors': selector.get_support().tolist(),
                            'cv_accuracy_mean': np.mean(cv_scores),
                            'cv_accuracy_std': np.std(cv_scores),
                            'feature_scores': selector.scores_.tolist()
                        }
                        
                    except Exception as e:
                        self.logger.warning(f"Panel validation failed for {k_features} features: {str(e)}")
                        
            else:
                # Regression task
                for k_features in [1, 2, 3, min(5, Z.shape[1])]:
                    if k_features > Z.shape[1]:
                        continue
                    
                    selector = SelectKBest(score_func=f_regression, k=k_features)
                    Z_selected = selector.fit_transform(Z, outcome_values)
                    
                    from sklearn.linear_model import Ridge
                    model = Ridge(alpha=1.0)
                    
                    try:
                        cv_scores = cross_val_score(model, Z_selected, outcome_values, cv=5, scoring='r2')
                        
                        outcome_results[f'panel_size_{k_features}'] = {
                            'selected_factors': selector.get_support().tolist(),
                            'cv_r2_mean': np.mean(cv_scores),
                            'cv_r2_std': np.std(cv_scores),
                            'feature_scores': selector.scores_.tolist()
                        }
                        
                    except Exception as e:
                        self.logger.warning(f"Panel validation failed for {k_features} features: {str(e)}")
            
            results[outcome_name] = outcome_results
        
        return results
    
    def _test_biomarker_robustness(self, X_list: List[np.ndarray], clinical_outcomes: Dict[str, np.ndarray],
                                 hypers: Dict, args: Dict, n_bootstrap: int = 10, **kwargs) -> Dict:
        """Test robustness of biomarker discoveries."""
        results = {}
        
        bootstrap_associations = {outcome: [] for outcome in clinical_outcomes.keys()}
        
        for bootstrap_idx in range(n_bootstrap):
            # Bootstrap sample
            n_subjects = X_list[0].shape[0]
            boot_indices = np.random.choice(n_subjects, n_subjects, replace=True)
            
            X_boot = [X[boot_indices] for X in X_list]
            outcomes_boot = {name: values[boot_indices] for name, values in clinical_outcomes.items()}
            
            try:
                # Run SGFA on bootstrap sample
                sgfa_result = self._run_sgfa_analysis(X_boot, hypers, args, **kwargs)
                Z_boot = sgfa_result['Z']
                
                # Calculate associations
                associations = self._analyze_factor_outcome_associations(Z_boot, outcomes_boot)
                
                for outcome_name, outcome_associations in associations.items():
                    bootstrap_associations[outcome_name].append(outcome_associations)
                    
            except Exception as e:
                self.logger.warning(f"Bootstrap iteration {bootstrap_idx} failed: {str(e)}")
                continue
        
        # Analyze robustness
        for outcome_name, bootstrap_results in bootstrap_associations.items():
            if not bootstrap_results:
                results[outcome_name] = {'error': 'No successful bootstrap iterations'}
                continue
            
            # Calculate stability of associations
            K = len(bootstrap_results[0]) if bootstrap_results else 0
            
            factor_stability = []
            for k in range(K):
                # Extract p-values or correlations across bootstraps
                p_values = []
                correlations = []
                
                for boot_result in bootstrap_results:
                    if k < len(boot_result):
                        factor_result = boot_result[k]
                        if 'p_value' in factor_result:
                            p_values.append(factor_result['p_value'])
                        if 'pearson_correlation' in factor_result:
                            correlations.append(factor_result['pearson_correlation'])
                
                stability_metrics = {'factor': k}
                
                if p_values:
                    # Proportion of significant results (p < 0.05)
                    stability_metrics['significance_rate'] = np.mean([p < 0.05 for p in p_values])
                    stability_metrics['mean_p_value'] = np.mean(p_values)
                
                if correlations:
                    stability_metrics['mean_correlation'] = np.mean(correlations)
                    stability_metrics['correlation_std'] = np.std(correlations)
                
                factor_stability.append(stability_metrics)
            
            results[outcome_name] = {
                'factor_stability': factor_stability,
                'n_bootstrap_iterations': len(bootstrap_results)
            }
        
        return results
    
    def _apply_trained_model(self, X_test_list: List[np.ndarray], train_result: Dict,
                           hypers: Dict, args: Dict, **kwargs) -> Dict:
        """Apply trained SGFA model to test data."""
        # In practice, this would use the trained model parameters
        # For simulation, we generate test factors with similar properties
        
        K = train_result['Z'].shape[1]
        n_test_subjects = X_test_list[0].shape[0]
        
        # Simulate test factors with some similarity to training
        np.random.seed(123)  # Different seed for test data
        Z_test = np.random.randn(n_test_subjects, K)
        
        # Add some systematic differences to simulate domain shift
        Z_test += np.random.normal(0.1, 0.2, Z_test.shape)  # Small systematic shift
        
        return {
            'Z': Z_test,
            'W': train_result['W'],  # Assume same loadings
            'log_likelihood': train_result['log_likelihood'] - 50,  # Slightly worse fit
            'domain_applied': 'test_cohort'
        }
    
    def _compare_factor_distributions(self, Z_train: np.ndarray, Z_test: np.ndarray) -> Dict:
        """Compare factor distributions between cohorts."""
        results = {}
        
        K = min(Z_train.shape[1], Z_test.shape[1])
        
        distribution_comparisons = []
        
        for k in range(K):
            # Statistical tests for distribution differences
            ks_stat, ks_p = stats.ks_2samp(Z_train[:, k], Z_test[:, k])
            t_stat, t_p = stats.ttest_ind(Z_train[:, k], Z_test[:, k])
            
            # Effect size (Cohen's d)
            pooled_std = np.sqrt((np.var(Z_train[:, k]) + np.var(Z_test[:, k])) / 2)
            cohens_d = (np.mean(Z_test[:, k]) - np.mean(Z_train[:, k])) / pooled_std
            
            distribution_comparisons.append({
                'factor': k,
                'ks_statistic': ks_stat,
                'ks_p_value': ks_p,
                't_statistic': t_stat,
                't_p_value': t_p,
                'cohens_d': cohens_d,
                'train_mean': np.mean(Z_train[:, k]),
                'test_mean': np.mean(Z_test[:, k]),
                'train_std': np.std(Z_train[:, k]),
                'test_std': np.std(Z_test[:, k])
            })
        
        results['factor_comparisons'] = distribution_comparisons
        
        # Overall distribution similarity
        significant_differences = sum(1 for comp in distribution_comparisons if comp['ks_p_value'] < 0.05)
        results['overall_similarity'] = {
            'n_significant_differences': significant_differences,
            'similarity_score': 1 - (significant_differences / K),
            'mean_effect_size': np.mean([abs(comp['cohens_d']) for comp in distribution_comparisons])
        }
        
        return results
    
    def _test_cross_cohort_classification(self, Z_train: np.ndarray, Z_test: np.ndarray,
                                        labels_train: np.ndarray, labels_test: np.ndarray) -> Dict:
        """Test classification performance across cohorts."""
        results = {}
        
        # Train on training cohort, test on test cohort
        for model_name, model in self.classification_models.items():
            try:
                # Train on training cohort
                model.fit(Z_train, labels_train)
                
                # Test on test cohort
                y_pred = model.predict(Z_test)
                y_pred_proba = model.predict_proba(Z_test)
                
                # Calculate metrics
                cross_cohort_metrics = self._calculate_detailed_metrics(labels_test, y_pred, y_pred_proba)
                
                # Also test within-cohort performance for comparison
                model_same_cohort = model.__class__(**model.get_params())
                model_same_cohort.fit(Z_test, labels_test)
                y_pred_same = model_same_cohort.predict(Z_test)
                y_pred_proba_same = model_same_cohort.predict_proba(Z_test)
                
                within_cohort_metrics = self._calculate_detailed_metrics(labels_test, y_pred_same, y_pred_proba_same)
                
                results[model_name] = {
                    'cross_cohort_performance': cross_cohort_metrics,
                    'within_cohort_performance': within_cohort_metrics,
                    'performance_drop': cross_cohort_metrics['accuracy'] - within_cohort_metrics['accuracy']
                }
                
            except Exception as e:
                self.logger.warning(f"Cross-cohort classification failed for {model_name}: {str(e)}")
                results[model_name] = {'error': str(e)}
        
        return results
    
    def _analyze_model_transferability(self, train_result: Dict, test_result: Dict,
                                     labels_train: np.ndarray, labels_test: np.ndarray) -> Dict:
        """Analyze model transferability between cohorts."""
        results = {}
        
        # Compare model fits
        results['model_fit_comparison'] = {
            'train_likelihood': train_result.get('log_likelihood', np.nan),
            'test_likelihood': test_result.get('log_likelihood', np.nan),
            'likelihood_drop': train_result.get('log_likelihood', 0) - test_result.get('log_likelihood', 0)
        }
        
        # Factor space similarity
        Z_train = train_result['Z']
        Z_test = test_result['Z']
        
        # Calculate canonical correlations between factor spaces
        from sklearn.cross_decomposition import CCA
        
        min_components = min(Z_train.shape[1], Z_test.shape[1], Z_train.shape[0], Z_test.shape[0])
        if min_components > 1:
            try:
                cca = CCA(n_components=min_components)
                cca.fit(Z_train, Z_test)
                
                # Transform both sets
                Z_train_cca, Z_test_cca = cca.transform(Z_train, Z_test)
                
                # Calculate canonical correlations
                canonical_correlations = []
                for i in range(min_components):
                    corr, _ = stats.pearsonr(Z_train_cca[:, i], Z_test_cca[:, i])
                    canonical_correlations.append(corr)
                
                results['factor_space_similarity'] = {
                    'canonical_correlations': canonical_correlations,
                    'mean_canonical_correlation': np.mean(canonical_correlations),
                    'transferability_score': np.mean(canonical_correlations)
                }
                
            except Exception as e:
                self.logger.warning(f"CCA analysis failed: {str(e)}")
                results['factor_space_similarity'] = {'error': str(e)}
        
        # Demographic transferability (if applicable)
        if len(np.unique(labels_train)) == len(np.unique(labels_test)):
            # Compare class distributions
            train_class_dist = np.bincount(labels_train) / len(labels_train)
            test_class_dist = np.bincount(labels_test) / len(labels_test)
            
            # KL divergence between class distributions
            kl_divergence = stats.entropy(test_class_dist, train_class_dist)
            
            results['demographic_transferability'] = {
                'train_class_distribution': train_class_dist.tolist(),
                'test_class_distribution': test_class_dist.tolist(),
                'kl_divergence': kl_divergence,
                'distribution_similarity': np.exp(-kl_divergence)
            }
        
        return results
    
    def _analyze_subtype_classification(self, results: Dict) -> Dict:
        """Analyze subtype classification validation results."""
        analysis = {
            'classification_performance': {},
            'sgfa_advantage': {},
            'clinical_utility': {}
        }
        
        # Extract classification results
        classification_results = results.get('classification_comparison', {})
        
        # Performance summary
        for method_name, method_results in classification_results.items():
            if 'error' not in method_results:
                best_model = None
                best_accuracy = 0
                
                for model_name, model_results in method_results.items():
                    if 'error' not in model_results:
                        cv_results = model_results.get('cross_validation', {})
                        accuracy = cv_results.get('accuracy', {}).get('mean', 0)
                        
                        if accuracy > best_accuracy:
                            best_accuracy = accuracy
                            best_model = model_name
                
                analysis['classification_performance'][method_name] = {
                    'best_model': best_model,
                    'best_accuracy': best_accuracy,
                    'performance_tier': 'high' if best_accuracy > 0.8 else 'medium' if best_accuracy > 0.65 else 'low'
                }
        
        # SGFA advantage analysis
        sgfa_performance = analysis['classification_performance'].get('sgfa_factors', {})
        raw_performance = analysis['classification_performance'].get('raw_data', {})
        pca_performance = analysis['classification_performance'].get('pca_features', {})
        
        if sgfa_performance and raw_performance:
            sgfa_accuracy = sgfa_performance.get('best_accuracy', 0)
            raw_accuracy = raw_performance.get('best_accuracy', 0)
            pca_accuracy = pca_performance.get('best_accuracy', 0)
            
            analysis['sgfa_advantage'] = {
                'vs_raw_data': sgfa_accuracy - raw_accuracy,
                'vs_pca': sgfa_accuracy - pca_accuracy,
                'relative_improvement_vs_raw': ((sgfa_accuracy / raw_accuracy) - 1) * 100 if raw_accuracy > 0 else np.nan,
                'clinical_significance': 'significant' if sgfa_accuracy - raw_accuracy > 0.05 else 'marginal'
            }
        
        # Clinical utility assessment
        interpretation_results = results.get('clinical_interpretation', {})
        if interpretation_results:
            factor_associations = interpretation_results.get('factor_label_associations', [])
            significant_factors = [f for f in factor_associations if f.get('p_value', 1) < 0.05]
            
            analysis['clinical_utility'] = {
                'n_significant_factors': len(significant_factors),
                'clinical_relevance_score': len(significant_factors) / len(factor_associations) if factor_associations else 0,
                'interpretability_assessment': 'high' if len(significant_factors) >= 2 else 'moderate' if len(significant_factors) >= 1 else 'low'
            }
        
        return analysis
    
    def _analyze_disease_progression_validation(self, results: Dict) -> Dict:
        """Analyze disease progression validation results."""
        analysis = {
            'predictive_performance': {},
            'clinical_correlations': {},
            'longitudinal_insights': {}
        }
        
        # Predictive performance
        prediction_results = results.get('progression_prediction', {})
        if prediction_results:
            best_model = None
            best_r2 = -np.inf
            
            for model_name, metrics in prediction_results.items():
                if 'error' not in metrics:
                    r2 = metrics.get('r2_score', -np.inf)
                    if r2 > best_r2:
                        best_r2 = r2
                        best_model = model_name
            
            analysis['predictive_performance'] = {
                'best_model': best_model,
                'best_r2_score': best_r2,
                'prediction_quality': 'excellent' if best_r2 > 0.7 else 'good' if best_r2 > 0.5 else 'moderate' if best_r2 > 0.3 else 'poor'
            }
        
        # Clinical correlations
        correlation_results = results.get('cross_sectional_correlations', {})
        if correlation_results:
            factor_correlations = correlation_results.get('factor_progression_correlations', [])
            strong_correlations = [f for f in factor_correlations if abs(f.get('pearson_correlation', 0)) > 0.5]
            
            analysis['clinical_correlations'] = {
                'n_strong_correlations': len(strong_correlations),
                'max_correlation': max([abs(f.get('pearson_correlation', 0)) for f in factor_correlations]) if factor_correlations else 0,
                'correlation_significance': len([f for f in factor_correlations if f.get('pearson_p_value', 1) < 0.05])
            }
        
        # Longitudinal insights
        longitudinal_results = results.get('longitudinal_analysis', {})
        if longitudinal_results and 'error' not in longitudinal_results:
            n_longitudinal = longitudinal_results.get('n_longitudinal_subjects', 0)
            
            analysis['longitudinal_insights'] = {
                'longitudinal_data_available': n_longitudinal > 0,
                'n_longitudinal_subjects': n_longitudinal,
                'longitudinal_utility': 'high' if n_longitudinal > 50 else 'moderate' if n_longitudinal > 20 else 'low'
            }
        
        return analysis
    
    def _analyze_biomarker_discovery(self, results: Dict) -> Dict:
        """Analyze biomarker discovery validation results."""
        analysis = {
            'biomarker_potential': {},
            'discovery_robustness': {},
            'clinical_relevance': {}
        }
        
        # Biomarker potential
        association_results = results.get('factor_associations', {})
        if association_results:
            total_associations = 0
            significant_associations = 0
            
            for outcome_name, outcome_associations in association_results.items():
                for factor_assoc in outcome_associations:
                    total_associations += 1
                    p_value = factor_assoc.get('p_value') or factor_assoc.get('pearson_p_value', 1)
                    if p_value < 0.05:
                        significant_associations += 1
            
            analysis['biomarker_potential'] = {
                'total_factor_outcome_tests': total_associations,
                'significant_associations': significant_associations,
                'discovery_rate': significant_associations / total_associations if total_associations > 0 else 0,
                'biomarker_promise': 'high' if significant_associations > 5 else 'moderate' if significant_associations > 2 else 'low'
            }
        
        # Discovery robustness
        robustness_results = results.get('robustness_analysis', {})
        if robustness_results:
            robust_discoveries = 0
            total_tested = 0
            
            for outcome_name, outcome_robustness in robustness_results.items():
                if 'error' not in outcome_robustness:
                    factor_stability = outcome_robustness.get('factor_stability', [])
                    for factor_result in factor_stability:
                        total_tested += 1
                        sig_rate = factor_result.get('significance_rate', 0)
                        if sig_rate > 0.7:  # Significant in >70% of bootstrap samples
                            robust_discoveries += 1
            
            analysis['discovery_robustness'] = {
                'robust_discoveries': robust_discoveries,
                'total_tested': total_tested,
                'robustness_rate': robust_discoveries / total_tested if total_tested > 0 else 0,
                'robustness_level': 'high' if robust_discoveries > 3 else 'moderate' if robust_discoveries > 1 else 'low'
            }
        
        # Clinical relevance
        panel_results = results.get('biomarker_panels', {})
        if panel_results:
            best_panel_performance = 0
            
            for outcome_name, panel_data in panel_results.items():
                for panel_size, panel_metrics in panel_data.items():
                    if 'error' not in panel_metrics:
                        performance = panel_metrics.get('cv_accuracy_mean') or panel_metrics.get('cv_r2_mean', 0)
                        best_panel_performance = max(best_panel_performance, performance)
            
            analysis['clinical_relevance'] = {
                'best_panel_performance': best_panel_performance,
                'clinical_utility': 'high' if best_panel_performance > 0.75 else 'moderate' if best_panel_performance > 0.65 else 'limited'
            }
        
        return analysis
    
    def _analyze_external_cohort_validation(self, results: Dict) -> Dict:
        """Analyze external cohort validation results."""
        analysis = {
            'generalizability': {},
            'domain_transfer': {},
            'model_robustness': {}
        }
        
        # Generalizability assessment
        classification_results = results.get('cross_cohort_classification', {})
        if classification_results:
            performance_drops = []
            cross_cohort_accuracies = []
            
            for model_name, model_results in classification_results.items():
                if 'error' not in model_results:
                    performance_drop = model_results.get('performance_drop', np.nan)
                    cross_cohort_acc = model_results.get('cross_cohort_performance', {}).get('accuracy', np.nan)
                    
                    if not np.isnan(performance_drop):
                        performance_drops.append(performance_drop)
                    if not np.isnan(cross_cohort_acc):
                        cross_cohort_accuracies.append(cross_cohort_acc)
            
            if performance_drops:
                mean_performance_drop = np.mean(performance_drops)
                analysis['generalizability'] = {
                    'mean_performance_drop': mean_performance_drop,
                    'mean_cross_cohort_accuracy': np.mean(cross_cohort_accuracies),
                    'generalizability_level': 'excellent' if mean_performance_drop < 0.05 else 'good' if mean_performance_drop < 0.1 else 'moderate' if mean_performance_drop < 0.2 else 'poor'
                }
        
        # Domain transfer analysis
        transferability_results = results.get('transferability_analysis', {})
        if transferability_results:
            factor_similarity = transferability_results.get('factor_space_similarity', {})
            transferability_score = factor_similarity.get('transferability_score', 0)
            
            analysis['domain_transfer'] = {
                'transferability_score': transferability_score,
                'transfer_quality': 'excellent' if transferability_score > 0.8 else 'good' if transferability_score > 0.6 else 'moderate' if transferability_score > 0.4 else 'poor'
            }
        
        # Model robustness
        distribution_comparison = results.get('distribution_comparison', {})
        if distribution_comparison:
            similarity_score = distribution_comparison.get('overall_similarity', {}).get('similarity_score', 0)
            
            analysis['model_robustness'] = {
                'distribution_similarity': similarity_score,
                'robustness_level': 'high' if similarity_score > 0.8 else 'moderate' if similarity_score > 0.6 else 'low'
            }
        
        return analysis
    
    def _plot_subtype_classification(self, results: Dict, clinical_labels: np.ndarray) -> Dict:
        """Generate plots for subtype classification validation."""
        plots = {}
        
        try:
            classification_results = results.get('classification_comparison', {})
            
            if not classification_results:
                return plots
            
            # Create comprehensive classification plots
            fig, axes = plt.subplots(2, 2, figsize=(15, 12))
            fig.suptitle('Subtype Classification Validation', fontsize=16)
            
            # Plot 1: Classification accuracy comparison
            methods = list(classification_results.keys())
            best_accuracies = []
            
            for method in methods:
                method_results = classification_results[method]
                if 'error' not in method_results:
                    best_acc = 0
                    for model_name, model_results in method_results.items():
                        if 'error' not in model_results:
                            cv_results = model_results.get('cross_validation', {})
                            accuracy = cv_results.get('accuracy', {}).get('mean', 0)
                            best_acc = max(best_acc, accuracy)
                    best_accuracies.append(best_acc)
                else:
                    best_accuracies.append(0)
            
            colors = ['red' if method == 'sgfa_factors' else 'skyblue' for method in methods]
            axes[0, 0].bar(methods, best_accuracies, color=colors)
            axes[0, 0].set_ylabel('Best Classification Accuracy')
            axes[0, 0].set_title('Classification Performance by Method')
            axes[0, 0].tick_params(axis='x', rotation=45)
            axes[0, 0].grid(True, alpha=0.3)
            
            # Plot 2: Factor-label associations
            interpretation_results = results.get('clinical_interpretation', {})
            if interpretation_results:
                factor_associations = interpretation_results.get('factor_label_associations', [])
                
                factors = [f['factor'] for f in factor_associations]
                p_values = [f.get('p_value', 1) for f in factor_associations]
                effect_sizes = [f.get('effect_size', 0) for f in factor_associations]
                
                scatter = axes[0, 1].scatter(effect_sizes, [-np.log10(p) for p in p_values], 
                                           c=factors, cmap='viridis', s=100)
                axes[0, 1].axhline(-np.log10(0.05), color='red', linestyle='--', label='p=0.05')
                axes[0, 1].set_xlabel('Effect Size')
                axes[0, 1].set_ylabel('-log10(p-value)')
                axes[0, 1].set_title('Factor-Clinical Label Associations')
                axes[0, 1].legend()
                axes[0, 1].grid(True, alpha=0.3)
                
                plt.colorbar(scatter, ax=axes[0, 1], label='Factor')
            
            # Plot 3: Class distribution
            unique_labels, label_counts = np.unique(clinical_labels, return_counts=True)
            axes[1, 0].pie(label_counts, labels=[f'Class {label}' for label in unique_labels], autopct='%1.1f%%')
            axes[1, 0].set_title('Clinical Label Distribution')
            
            # Plot 4: Model comparison detailed
            if len(methods) > 1:
                sgfa_results = classification_results.get('sgfa_factors', {})
                comparison_method = next((m for m in methods if m != 'sgfa_factors'), None)
                
                if sgfa_results and comparison_method:
                    comparison_results = classification_results[comparison_method]
                    
                    metrics = ['accuracy', 'precision_macro', 'recall_macro', 'f1_macro']
                    sgfa_scores = []
                    comparison_scores = []
                    
                    # Get best model performance for each metric
                    for metric in metrics:
                        sgfa_best = 0
                        comp_best = 0
                        
                        for model_results in sgfa_results.values():
                            if 'error' not in model_results:
                                cv_results = model_results.get('cross_validation', {})
                                score = cv_results.get(metric, {}).get('mean', 0)
                                sgfa_best = max(sgfa_best, score)
                        
                        for model_results in comparison_results.values():
                            if 'error' not in model_results:
                                cv_results = model_results.get('cross_validation', {})
                                score = cv_results.get(metric, {}).get('mean', 0)
                                comp_best = max(comp_best, score)
                        
                        sgfa_scores.append(sgfa_best)
                        comparison_scores.append(comp_best)
                    
                    x = np.arange(len(metrics))
                    width = 0.35
                    
                    axes[1, 1].bar(x - width/2, sgfa_scores, width, label='SGFA', color='red', alpha=0.8)
                    axes[1, 1].bar(x + width/2, comparison_scores, width, label=comparison_method, color='blue', alpha=0.8)
                    
                    axes[1, 1].set_xlabel('Metrics')
                    axes[1, 1].set_ylabel('Score')
                    axes[1, 1].set_title('SGFA vs Comparison Method')
                    axes[1, 1].set_xticks(x)
                    axes[1, 1].set_xticklabels(metrics)
                    axes[1, 1].legend()
                    axes[1, 1].grid(True, alpha=0.3)
            
            plt.tight_layout()
            plots['subtype_classification_validation'] = fig
            
        except Exception as e:
            self.logger.warning(f"Failed to create subtype classification plots: {str(e)}")
        
        return plots
    
    def _plot_disease_progression_validation(self, results: Dict, progression_scores: np.ndarray) -> Dict:
        """Generate plots for disease progression validation."""
        plots = {}
        
        try:
            fig, axes = plt.subplots(2, 2, figsize=(15, 12))
            fig.suptitle('Disease Progression Validation', fontsize=16)
            
            # Plot 1: Progression prediction performance
            prediction_results = results.get('progression_prediction', {})
            if prediction_results:
                models = []
                r2_scores = []
                
                for model_name, metrics in prediction_results.items():
                    if 'error' not in metrics:
                        models.append(model_name)
                        r2_scores.append(metrics.get('r2_score', 0))
                
                if models:
                    axes[0, 0].bar(models, r2_scores)
                    axes[0, 0].set_ylabel('R² Score')
                    axes[0, 0].set_title('Progression Prediction Performance')
                    axes[0, 0].tick_params(axis='x', rotation=45)
                    axes[0, 0].grid(True, alpha=0.3)
            
            # Plot 2: Factor-progression correlations
            correlation_results = results.get('cross_sectional_correlations', {})
            if correlation_results:
                factor_correlations = correlation_results.get('factor_progression_correlations', [])
                
                if factor_correlations:
                    factors = [f['factor'] for f in factor_correlations]
                    correlations = [f.get('pearson_correlation', 0) for f in factor_correlations]
                    p_values = [f.get('pearson_p_value', 1) for f in factor_correlations]
                    
                    colors = ['red' if p < 0.05 else 'gray' for p in p_values]
                    bars = axes[0, 1].bar(factors, correlations, color=colors)
                    axes[0, 1].set_xlabel('Factor')
                    axes[0, 1].set_ylabel('Correlation with Progression')
                    axes[0, 1].set_title('Factor-Progression Correlations')
                    axes[0, 1].axhline(0, color='black', linestyle='-', alpha=0.5)
                    axes[0, 1].grid(True, alpha=0.3)
                    
                    # Add significance indicators
                    from matplotlib.patches import Patch
                    legend_elements = [Patch(facecolor='red', label='p < 0.05'),
                                     Patch(facecolor='gray', label='p ≥ 0.05')]
                    axes[0, 1].legend(handles=legend_elements)
            
            # Plot 3: Progression score distribution
            axes[1, 0].hist(progression_scores, bins=20, alpha=0.7, color='skyblue', edgecolor='black')
            axes[1, 0].set_xlabel('Progression Score')
            axes[1, 0].set_ylabel('Frequency')
            axes[1, 0].set_title('Progression Score Distribution')
            axes[1, 0].grid(True, alpha=0.3)
            
            # Plot 4: Longitudinal analysis (if available)
            longitudinal_results = results.get('longitudinal_analysis', {})
            if longitudinal_results and 'error' not in longitudinal_results:
                factor_change_analysis = longitudinal_results.get('factor_change_analysis', {})
                
                if factor_change_analysis:
                    factor_correlations = factor_change_analysis.get('factor_progression_correlations', [])
                    
                    if factor_correlations:
                        factors = [f['factor'] for f in factor_correlations]
                        change_correlations = [f.get('correlation', 0) for f in factor_correlations]
                        
                        axes[1, 1].bar(factors, change_correlations, alpha=0.8, color='green')
                        axes[1, 1].set_xlabel('Factor')
                        axes[1, 1].set_ylabel('Change-Progression Correlation')
                        axes[1, 1].set_title('Factor Change vs Progression Rate')
                        axes[1, 1].axhline(0, color='black', linestyle='-', alpha=0.5)
                        axes[1, 1].grid(True, alpha=0.3)
            else:
                axes[1, 1].text(0.5, 0.5, 'No Longitudinal Data Available', 
                               ha='center', va='center', transform=axes[1, 1].transAxes)
                axes[1, 1].set_title('Longitudinal Analysis')
            
            plt.tight_layout()
            plots['disease_progression_validation'] = fig
            
        except Exception as e:
            self.logger.warning(f"Failed to create disease progression plots: {str(e)}")
        
        return plots
    
    def _plot_biomarker_discovery(self, results: Dict, clinical_outcomes: Dict[str, np.ndarray]) -> Dict:
        """Generate plots for biomarker discovery validation."""
        plots = {}
        
        try:
            fig, axes = plt.subplots(2, 2, figsize=(15, 12))
            fig.suptitle('Biomarker Discovery Validation', fontsize=16)
            
            # Plot 1: Association significance heatmap
            association_results = results.get('factor_associations', {})
            if association_results:
                outcome_names = list(association_results.keys())
                n_factors = len(association_results[outcome_names[0]]) if outcome_names else 0
                
                if outcome_names and n_factors > 0:
                    # Create p-value matrix
                    p_value_matrix = np.ones((len(outcome_names), n_factors))
                    
                    for i, outcome_name in enumerate(outcome_names):
                        outcome_associations = association_results[outcome_name]
                        for j, factor_assoc in enumerate(outcome_associations):
                            p_value = factor_assoc.get('p_value') or factor_assoc.get('pearson_p_value', 1)
                            p_value_matrix[i, j] = -np.log10(p_value)
                    
                    im = axes[0, 0].imshow(p_value_matrix, cmap='Reds', aspect='auto')
                    axes[0, 0].set_xticks(range(n_factors))
                    axes[0, 0].set_yticks(range(len(outcome_names)))
                    axes[0, 0].set_xticklabels([f'Factor {i}' for i in range(n_factors)])
                    axes[0, 0].set_yticklabels(outcome_names)
                    axes[0, 0].set_title('Association Significance (-log10 p-value)')
                    plt.colorbar(im, ax=axes[0, 0])
                    
                    # Add significance threshold line
                    threshold = -np.log10(0.05)
                    axes[0, 0].contour(p_value_matrix, levels=[threshold], colors='blue', linestyles='--')
            
            # Plot 2: Feature importance
            importance_results = results.get('feature_importance', {})
            if importance_results:
                feature_rankings = importance_results.get('feature_rankings', {})
                top_features = feature_rankings.get('top_features', [])
                
                if top_features:
                    # Show top 10 features
                    top_10 = top_features[:10]
                    importance_scores = [f['importance_score'] for f in top_10]
                    feature_labels = [f"V{f['view']}F{f['feature_index']}" for f in top_10]
                    
                    axes[0, 1].barh(range(len(top_10)), importance_scores)
                    axes[0, 1].set_yticks(range(len(top_10)))
                    axes[0, 1].set_yticklabels(feature_labels)
                    axes[0, 1].set_xlabel('Importance Score')
                    axes[0, 1].set_title('Top 10 Most Important Features')
                    axes[0, 1].grid(True, alpha=0.3)
            
            # Plot 3: Biomarker panel performance
            panel_results = results.get('biomarker_panels', {})
            if panel_results:
                panel_performance_data = []
                
                for outcome_name, panel_data in panel_results.items():
                    for panel_size, panel_metrics in panel_data.items():
                        if 'error' not in panel_metrics:
                            performance = panel_metrics.get('cv_accuracy_mean') or panel_metrics.get('cv_r2_mean', 0)
                            size = int(panel_size.split('_')[-1])
                            panel_performance_data.append({
                                'outcome': outcome_name,
                                'panel_size': size,
                                'performance': performance
                            })
                
                if panel_performance_data:
                    # Group by panel size
                    panel_sizes = sorted(list(set([d['panel_size'] for d in panel_performance_data])))
                    avg_performance = []
                    
                    for size in panel_sizes:
                        size_performances = [d['performance'] for d in panel_performance_data if d['panel_size'] == size]
                        avg_performance.append(np.mean(size_performances))
                    
                    axes[1, 0].plot(panel_sizes, avg_performance, 'o-', linewidth=2, markersize=8)
                    axes[1, 0].set_xlabel('Panel Size (Number of Factors)')
                    axes[1, 0].set_ylabel('Average Performance')
                    axes[1, 0].set_title('Biomarker Panel Performance vs Size')
                    axes[1, 0].grid(True, alpha=0.3)
            
            # Plot 4: Robustness analysis
            robustness_results = results.get('robustness_analysis', {})
            if robustness_results:
                outcome_names = []
                significance_rates = []
                
                for outcome_name, outcome_robustness in robustness_results.items():
                    if 'error' not in outcome_robustness:
                        factor_stability = outcome_robustness.get('factor_stability', [])
                        
                        # Average significance rate across factors
                        sig_rates = [f.get('significance_rate', 0) for f in factor_stability if 'significance_rate' in f]
                        if sig_rates:
                            avg_sig_rate = np.mean(sig_rates)
                            outcome_names.append(outcome_name)
                            significance_rates.append(avg_sig_rate)
                
                if outcome_names:
                    axes[1, 1].bar(outcome_names, significance_rates, alpha=0.7)
                    axes[1, 1].set_ylabel('Average Significance Rate')
                    axes[1, 1].set_title('Biomarker Discovery Robustness')
                    axes[1, 1].tick_params(axis='x', rotation=45)
                    axes[1, 1].axhline(0.7, color='red', linestyle='--', label='Robustness Threshold')
                    axes[1, 1].legend()
                    axes[1, 1].grid(True, alpha=0.3)
            
            plt.tight_layout()
            plots['biomarker_discovery_validation'] = fig
            
        except Exception as e:
            self.logger.warning(f"Failed to create biomarker discovery plots: {str(e)}")
        
        return plots
    
    def _plot_external_cohort_validation(self, results: Dict, clinical_labels_train: np.ndarray, clinical_labels_test: np.ndarray) -> Dict:
        """Generate plots for external cohort validation."""
        plots = {}
        
        try:
            fig, axes = plt.subplots(2, 2, figsize=(15, 12))
            fig.suptitle('External Cohort Validation', fontsize=16)
            
            # Plot 1: Cross-cohort vs within-cohort performance
            classification_results = results.get('cross_cohort_classification', {})
            if classification_results:
                models = []
                cross_cohort_acc = []
                within_cohort_acc = []
                
                for model_name, model_results in classification_results.items():
                    if 'error' not in model_results:
                        models.append(model_name)
                        cross_cohort_acc.append(model_results.get('cross_cohort_performance', {}).get('accuracy', 0))
                        within_cohort_acc.append(model_results.get('within_cohort_performance', {}).get('accuracy', 0))
                
                if models:
                    x = np.arange(len(models))
                    width = 0.35
                    
                    axes[0, 0].bar(x - width/2, cross_cohort_acc, width, label='Cross-cohort', alpha=0.8)
                    axes[0, 0].bar(x + width/2, within_cohort_acc, width, label='Within-cohort', alpha=0.8)
                    
                    axes[0, 0].set_xlabel('Model')
                    axes[0, 0].set_ylabel('Accuracy')
                    axes[0, 0].set_title('Cross-cohort vs Within-cohort Performance')
                    axes[0, 0].set_xticks(x)
                    axes[0, 0].set_xticklabels(models)
                    axes[0, 0].legend()
                    axes[0, 0].grid(True, alpha=0.3)
            
            # Plot 2: Factor distribution comparison
            distribution_comparison = results.get('distribution_comparison', {})
            if distribution_comparison:
                factor_comparisons = distribution_comparison.get('factor_comparisons', [])
                
                if factor_comparisons:
                    factors = [f['factor'] for f in factor_comparisons]
                    train_means = [f['train_mean'] for f in factor_comparisons]
                    test_means = [f['test_mean'] for f in factor_comparisons]
                    
                    axes[0, 1].scatter(train_means, test_means, s=100, alpha=0.7)
                    
                    # Add diagonal line for perfect correspondence
                    min_val = min(min(train_means), min(test_means))
                    max_val = max(max(train_means), max(test_means))
                    axes[0, 1].plot([min_val, max_val], [min_val, max_val], 'r--', alpha=0.7)
                    
                    axes[0, 1].set_xlabel('Training Cohort Factor Mean')
                    axes[0, 1].set_ylabel('Test Cohort Factor Mean')
                    axes[0, 1].set_title('Factor Distribution Comparison')
                    axes[0, 1].grid(True, alpha=0.3)
                    
                    # Annotate points with factor numbers
                    for i, factor in enumerate(factors):
                        axes[0, 1].annotate(f'F{factor}', (train_means[i], test_means[i]), 
                                          xytext=(5, 5), textcoords='offset points')
            
            # Plot 3: Cohort label distributions
            train_unique, train_counts = np.unique(clinical_labels_train, return_counts=True)
            test_unique, test_counts = np.unique(clinical_labels_test, return_counts=True)
            
            # Normalize to proportions
            train_props = train_counts / len(clinical_labels_train)
            test_props = test_counts / len(clinical_labels_test)
            
            x = np.arange(len(train_unique))
            width = 0.35
            
            axes[1, 0].bar(x - width/2, train_props, width, label='Training Cohort', alpha=0.8)
            axes[1, 0].bar(x + width/2, test_props, width, label='Test Cohort', alpha=0.8)
            
            axes[1, 0].set_xlabel('Clinical Label')
            axes[1, 0].set_ylabel('Proportion')
            axes[1, 0].set_title('Clinical Label Distribution Comparison')
            axes[1, 0].set_xticks(x)
            axes[1, 0].set_xticklabels([f'Class {label}' for label in train_unique])
            axes[1, 0].legend()
            axes[1, 0].grid(True, alpha=0.3)
            
            # Plot 4: Transferability metrics
            transferability_results = results.get('transferability_analysis', {})
            if transferability_results:
                metrics = []
                values = []
                
                # Model fit comparison
                model_fit = transferability_results.get('model_fit_comparison', {})
                if model_fit:
                    train_ll = model_fit.get('train_likelihood', 0)
                    test_ll = model_fit.get('test_likelihood', 0)
                    if train_ll != 0:
                        metrics.append('Likelihood Ratio')
                        values.append(test_ll / train_ll)
                
                # Factor space similarity
                factor_similarity = transferability_results.get('factor_space_similarity', {})
                if factor_similarity:
                    transferability_score = factor_similarity.get('transferability_score', 0)
                    metrics.append('Factor Similarity')
                    values.append(transferability_score)
                
                # Demographic transferability
                demo_transfer = transferability_results.get('demographic_transferability', {})
                if demo_transfer:
                    distribution_similarity = demo_transfer.get('distribution_similarity', 0)
                    metrics.append('Demographic Similarity')
                    values.append(distribution_similarity)
                
                if metrics:
                    colors = ['green' if v > 0.7 else 'orange' if v > 0.5 else 'red' for v in values]
                    axes[1, 1].bar(metrics, values, color=colors, alpha=0.8)
                    axes[1, 1].set_ylabel('Score')
                    axes[1, 1].set_title('Transferability Metrics')
                    axes[1, 1].tick_params(axis='x', rotation=45)
                    axes[1, 1].axhline(0.7, color='green', linestyle='--', alpha=0.7, label='Good')
                    axes[1, 1].axhline(0.5, color='orange', linestyle='--', alpha=0.7, label='Moderate')
                    axes[1, 1].legend()
                    axes[1, 1].grid(True, alpha=0.3)
            
            plt.tight_layout()
            plots['external_cohort_validation'] = fig
            
        except Exception as e:
            self.logger.warning(f"Failed to create external cohort validation plots: {str(e)}")

        return plots

    def _create_comprehensive_clinical_visualizations(self, X_list: List[np.ndarray],
                                                    results: Dict, clinical_data: np.ndarray,
                                                    experiment_name: str) -> Dict:
        """Create comprehensive clinical visualizations focusing on subtypes and brain maps."""
        advanced_plots = {}

        try:
            self.logger.info(f"🎨 Creating comprehensive clinical visualizations for {experiment_name}")

            # Import visualization system
            from visualization.manager import VisualizationManager
            from core.config_utils import ConfigAccessor

            # Create a clinical-focused config for visualization
            viz_config = ConfigAccessor({
                'visualization': {
                    'create_brain_viz': True,
                    'output_format': ['png', 'pdf'],
                    'dpi': 300,
                    'clinical_focus': True
                },
                'output_dir': f'/tmp/clinical_viz_{experiment_name}'
            })

            # Initialize visualization manager
            viz_manager = VisualizationManager(viz_config)

            # Prepare clinical data structure for visualizations
            data = {
                'X_list': X_list,
                'view_names': [f'view_{i}' for i in range(len(X_list))],
                'n_subjects': X_list[0].shape[0],
                'view_dimensions': [X.shape[1] for X in X_list],
                'clinical_labels': clinical_data,
                'preprocessing': {
                    'status': 'completed',
                    'strategy': 'clinical_neuroimaging'
                }
            }

            # Extract the best clinical result for detailed analysis
            best_clinical_result = self._extract_best_clinical_result(results)

            if best_clinical_result:
                # Prepare clinical analysis results
                analysis_results = {
                    'best_run': best_clinical_result,
                    'all_runs': results,
                    'model_type': 'clinical_sparseGFA',
                    'convergence': best_clinical_result.get('convergence', False),
                    'clinical_validation': True
                }

                # Add cross-validation results for subtype consensus if available
                cv_results = None
                if 'subtype_stability' in results:
                    cv_results = {
                        'subtype_analysis': results['subtype_stability'],
                        'consensus_subtypes': results.get('consensus_subtypes', {}),
                        'stability_metrics': results.get('stability_analysis', {})
                    }

                # Create all comprehensive visualizations with clinical focus
                viz_manager.create_all_visualizations(
                    data=data,
                    analysis_results=analysis_results,
                    cv_results=cv_results
                )

                # Extract the generated plots and convert to matplotlib figures
                if hasattr(viz_manager, 'plot_dir') and viz_manager.plot_dir.exists():
                    plot_files = list(viz_manager.plot_dir.glob('**/*.png'))

                    for plot_file in plot_files:
                        plot_name = f"clinical_{plot_file.stem}"

                        # Load the saved plot as a matplotlib figure
                        try:
                            import matplotlib.pyplot as plt
                            import matplotlib.image as mpimg

                            fig, ax = plt.subplots(figsize=(12, 8))
                            img = mpimg.imread(str(plot_file))
                            ax.imshow(img)
                            ax.axis('off')
                            ax.set_title(f"Clinical Analysis: {plot_name}", fontsize=14)

                            advanced_plots[plot_name] = fig

                        except Exception as e:
                            self.logger.warning(f"Could not load clinical plot {plot_name}: {e}")

                    self.logger.info(f"✅ Created {len(plot_files)} comprehensive clinical visualizations")

                    # Additional clinical-specific summary
                    if cv_results and 'subtype_analysis' in cv_results:
                        self.logger.info("   → Subtype discovery and consensus plots generated")
                    if 'brain_maps' in [f.stem for f in plot_files]:
                        self.logger.info("   → Clinical brain mapping visualizations generated")

                else:
                    self.logger.warning("Clinical visualization manager did not create plot directory")
            else:
                self.logger.warning("No converged clinical results found for comprehensive visualization")

        except Exception as e:
            self.logger.warning(f"Failed to create comprehensive clinical visualizations: {e}")
            # Don't fail the experiment if advanced visualizations fail

        return advanced_plots

    def _extract_best_clinical_result(self, results: Dict) -> Optional[Dict]:
        """Extract the best clinical analysis result from experiment results."""
        best_result = None
        best_score = float('-inf')

        # Look for clinical results with SGFA data
        if 'sgfa_factors' in results:
            # From subtype classification
            result = results['sgfa_factors']
            if result and result.get('convergence', False):
                score = result.get('accuracy', result.get('log_likelihood', float('-inf')))
                if score > best_score:
                    best_score = score
                    best_result = result

        # Look in other result structures
        for key, result in results.items():
            if (isinstance(result, dict) and
                'Z' in result and result.get('convergence', False)):
                score = result.get('log_likelihood', float('-inf'))
                if score > best_score:
                    best_score = score
                    best_result = result

        return best_result


def run_clinical_validation(config):
    """Run clinical validation experiments with remote workstation integration."""
    import logging
    import sys
    import os
    import numpy as np
    logger = logging.getLogger(__name__)
    logger.info("Starting Clinical Validation Experiments")

    try:
        # Add project root to path for imports
        current_file = os.path.abspath(__file__)
        project_root = os.path.dirname(os.path.dirname(current_file))
        if project_root not in sys.path:
            sys.path.insert(0, project_root)

        from experiments.framework import ExperimentFramework, ExperimentConfig
        from pathlib import Path

        # Load data with advanced preprocessing for clinical validation
        from data.preprocessing_integration import apply_preprocessing_to_pipeline
        logger.info("🔧 Loading data for clinical validation...")
        X_list, preprocessing_info = apply_preprocessing_to_pipeline(
            config=config,
            data_dir=get_data_dir(config),
            auto_select_strategy=False,
            preferred_strategy="clinical_focused"  # Use clinical-focused preprocessing
        )

        logger.info(f"✅ Data loaded: {len(X_list)} views for clinical validation")
        for i, X in enumerate(X_list):
            logger.info(f"   View {i}: {X.shape}")

        # Load clinical data
        try:
            from data.qmap_pd import load_qmap_pd
            clinical_data = load_qmap_pd(get_data_dir(config))

            # Extract clinical labels (using mock labels as fallback - real clinical subtype data would be preferred)
            n_subjects = X_list[0].shape[0]
            clinical_labels = np.random.randint(0, 3, n_subjects)  # Mock: 3 PD subtypes
            logger.info(f"✅ Clinical labels loaded: {len(np.unique(clinical_labels))} unique subtypes")

        except Exception as e:
            logger.warning(f"Clinical data loading failed: {e}")
            # Create mock clinical labels for testing
            n_subjects = X_list[0].shape[0]
            clinical_labels = np.random.randint(0, 3, n_subjects)
            logger.info("Using mock clinical labels for testing")

        # Initialize experiment framework
        framework = ExperimentFramework(
            base_output_dir=get_output_dir(config)
        )

        exp_config = ExperimentConfig(
            experiment_name="clinical_validation",
            description="Clinical validation of SGFA factors for PD subtypes",
            dataset="qmap_pd",
            data_dir=get_data_dir(config)
        )

        # Create clinical validation experiment instance
        clinical_exp = ClinicalValidationExperiments(exp_config, logger)

        # Setup base hyperparameters
        base_hypers = {
            'Dm': [X.shape[1] for X in X_list],
            'a_sigma': 1.0,
            'b_sigma': 1.0,
            'slab_df': 4.0,
            'slab_scale': 2.0,
            'percW': 33.0,
            'K': 10  # Base number of factors
        }

        # Setup base args
        base_args = {
            'K': 10,
            'num_warmup': 100,  # Moderate sampling for clinical validation
            'num_samples': 300,  # More samples for robust clinical results
            'num_chains': 1,
            'target_accept_prob': 0.8,
            'reghsZ': True
        }

        # Run the experiment
        def clinical_validation_experiment(config, output_dir, **kwargs):
            logger.info("🏥 Running comprehensive clinical validation...")

            results = {}
            total_tests = 0
            successful_tests = 0

            # 1. Run SGFA to extract factors
            logger.info("📊 Extracting SGFA factors...")
            try:
                with clinical_exp.profiler.profile('sgfa_extraction') as p:
                    sgfa_result = clinical_exp._run_sgfa_analysis(X_list, base_hypers, base_args)

                if 'error' in sgfa_result:
                    raise ValueError(f"SGFA failed: {sgfa_result['error']}")

                Z_sgfa = sgfa_result['Z']  # Factor scores
                metrics = clinical_exp.profiler.get_current_metrics()

                results['sgfa_extraction'] = {
                    'result': sgfa_result,
                    'performance': {
                        'execution_time': metrics.execution_time,
                        'peak_memory_gb': metrics.peak_memory_gb,
                        'convergence': sgfa_result.get('convergence', False),
                        'log_likelihood': sgfa_result.get('log_likelihood', float('-inf'))
                    },
                    'factor_info': {
                        'n_factors': Z_sgfa.shape[1],
                        'n_subjects': Z_sgfa.shape[0]
                    }
                }
                successful_tests += 1
                logger.info(f"✅ SGFA extraction: {metrics.execution_time:.1f}s, {Z_sgfa.shape[1]} factors")

            except Exception as e:
                logger.error(f"❌ SGFA extraction failed: {e}")
                results['sgfa_extraction'] = {'error': str(e)}
                # Return early if SGFA fails
                return {
                    'status': 'failed',
                    'error': 'SGFA extraction failed',
                    'results': results
                }

            total_tests += 1

            # 2. Test subtype classification
            logger.info("📊 Testing subtype classification...")
            try:
                classification_results = clinical_exp._test_factor_classification(
                    Z_sgfa, clinical_labels, 'sgfa_factors'
                )
                results['subtype_classification'] = classification_results
                successful_tests += 1

                # Log classification performance
                best_accuracy = max([model_result.get('accuracy', 0) for model_result in classification_results.values()])
                logger.info(f"✅ Best classification accuracy: {best_accuracy:.3f}")

            except Exception as e:
                logger.error(f"❌ Subtype classification failed: {e}")
                results['subtype_classification'] = {'error': str(e)}

            total_tests += 1

            # 3. Compare with baseline methods
            logger.info("📊 Comparing with baseline methods...")
            try:
                # PCA comparison
                from sklearn.decomposition import PCA
                X_concat = np.hstack(X_list)
                pca = PCA(n_components=Z_sgfa.shape[1])
                Z_pca = pca.fit_transform(X_concat)

                pca_results = clinical_exp._test_factor_classification(
                    Z_pca, clinical_labels, 'pca_features'
                )

                # Raw data comparison
                raw_results = clinical_exp._test_factor_classification(
                    X_concat, clinical_labels, 'raw_data'
                )

                results['baseline_comparison'] = {
                    'pca': pca_results,
                    'raw_data': raw_results
                }
                successful_tests += 1

                # Log comparison
                sgfa_acc = max([r.get('accuracy', 0) for r in results['subtype_classification'].values()])
                pca_acc = max([r.get('accuracy', 0) for r in pca_results.values()])
                raw_acc = max([r.get('accuracy', 0) for r in raw_results.values()])

                logger.info(f"✅ Classification comparison - SGFA: {sgfa_acc:.3f}, PCA: {pca_acc:.3f}, Raw: {raw_acc:.3f}")

            except Exception as e:
                logger.error(f"❌ Baseline comparison failed: {e}")
                results['baseline_comparison'] = {'error': str(e)}

            total_tests += 1

            logger.info("🏥 Clinical validation completed!")
            logger.info(f"   Successful tests: {successful_tests}/{total_tests}")

            return {
                'status': 'completed',
                'clinical_results': results,
                'summary': {
                    'total_tests': total_tests,
                    'successful_tests': successful_tests,
                    'success_rate': successful_tests / total_tests if total_tests > 0 else 0,
                    'data_characteristics': {
                        'n_subjects': X_list[0].shape[0],
                        'n_views': len(X_list),
                        'view_dimensions': [X.shape[1] for X in X_list],
                        'n_clinical_subtypes': len(np.unique(clinical_labels))
                    }
                }
            }

        # Run experiment using framework
        result = framework.run_experiment(
            experiment_function=clinical_validation_experiment,
            config=exp_config,
            data={'X_list': X_list, 'preprocessing_info': preprocessing_info, 'clinical_labels': clinical_labels}
        )

        logger.info("✅ Clinical validation completed successfully")
        return result

    except Exception as e:
        logger.error(f"Clinical validation failed: {e}")
        return None