"""Biomarker discovery and validation utilities.

This module provides utilities for discovering and validating biomarkers
using SGFA-derived factors and clinical outcomes.

⚠️  IMPORTANT: This module is FUTURE WORK and not implemented in the current project scope.
The current qMAP-PD dataset does not include biomarker data. This infrastructure is provided
for future development when biomarker data becomes available.
"""

import logging
import numpy as np
from typing import Dict, List, Optional
from scipy import stats
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.feature_selection import SelectKBest, f_classif, f_regression
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.model_selection import cross_val_score


class BiomarkerAnalyzer:
    """Biomarker discovery and validation."""

    def __init__(
        self,
        data_processor: Optional[object] = None,
        config: Optional[Dict] = None,
        logger: Optional[logging.Logger] = None
    ):
        """
        Initialize biomarker analyzer.

        Args:
            data_processor: Optional ClinicalDataProcessor for robustness testing
            config: Optional configuration dict with cross_validation settings
            logger: Optional logger instance
        """
        self.data_processor = data_processor
        self.config = config or {}
        self.logger = logger or logging.getLogger(__name__)

    def discover_neuroimaging_biomarkers(
        self,
        sgfa_cv_results: Dict,
        clinical_data: Dict
    ) -> Dict:
        """
        Discover neuroimaging biomarkers from SGFA cross-validation results.

        Args:
            sgfa_cv_results: SGFA CV results containing fold_results
            clinical_data: Clinical data dictionary with 'diagnosis' key

        Returns:
            Dict containing:
                - factor_importance: Mean importance scores across folds
                - discriminative_factors: Factors with significant diagnosis associations
                - success: Whether discovery succeeded
        """
        try:
            successful_folds = [
                f for f in sgfa_cv_results.get("fold_results", [])
                if f.get("success", False)
            ]

            if not successful_folds:
                return {"error": "No successful folds for biomarker discovery"}

            biomarker_results = {
                "factor_importance": {},
                "clinical_associations": {},
                "discriminative_factors": []
            }

            # Analyze factor importance across folds
            factor_importances = []
            for fold in successful_folds:
                neuroimaging_metrics = fold.get("neuroimaging_metrics", {})
                if "factor_importance" in neuroimaging_metrics:
                    factor_importances.append(neuroimaging_metrics["factor_importance"])

            if factor_importances:
                # Aggregate factor importance
                mean_importance = np.mean(factor_importances, axis=0)

                biomarker_results["factor_importance"] = {
                    "mean_importance": mean_importance.tolist(),
                    "top_factors": (
                        np.argsort(mean_importance)[-3:].tolist()
                        if len(mean_importance) > 0 else []
                    )
                }

            # Identify discriminative factors
            diagnoses = np.array(clinical_data["diagnosis"])
            unique_diagnoses = np.unique(diagnoses)

            if len(unique_diagnoses) > 1:
                # For each factor, test association with diagnosis
                discriminative_factors = []

                for fold in successful_folds:
                    train_result = fold.get("train_result", {})
                    Z_train = train_result.get("Z")
                    train_idx = fold.get("train_idx", [])

                    if Z_train is not None:
                        train_diagnoses = diagnoses[train_idx]

                        # Test each factor for discriminative power
                        for factor_idx in range(Z_train.shape[1]):
                            factor_values = Z_train[:, factor_idx]

                            # Group by diagnosis
                            groups = [
                                factor_values[train_diagnoses == diag]
                                for diag in unique_diagnoses
                            ]
                            groups = [g for g in groups if len(g) > 0]  # Remove empty

                            if len(groups) > 1:
                                # ANOVA test
                                try:
                                    f_stat, p_value = stats.f_oneway(*groups)
                                    if p_value < 0.05:  # Significant
                                        discriminative_factors.append({
                                            "factor_idx": factor_idx,
                                            "f_statistic": f_stat,
                                            "p_value": p_value,
                                            "fold_idx": fold.get("fold_idx", -1)
                                        })
                                except Exception:
                                    pass

                biomarker_results["discriminative_factors"] = discriminative_factors

            biomarker_results["success"] = True
            return biomarker_results

        except Exception as e:
            return {"error": f"Biomarker discovery failed: {str(e)}"}

    def analyze_factor_outcome_associations(
        self,
        Z: np.ndarray,
        clinical_outcomes: Dict[str, np.ndarray]
    ) -> Dict:
        """
        Analyze associations between factors and clinical outcomes.

        Args:
            Z: Latent factors (N x K)
            clinical_outcomes: Dict of clinical outcome arrays

        Returns:
            Dict with keys for each outcome containing factor associations
        """
        results = {}

        K = Z.shape[1]

        for outcome_name, outcome_values in clinical_outcomes.items():
            outcome_results = []

            for k in range(K):
                # Determine if outcome is continuous or categorical
                if len(np.unique(outcome_values)) > 10:  # Continuous outcome
                    corr_coef, p_val = stats.pearsonr(Z[:, k], outcome_values)
                    spearman_coef, spearman_p = stats.spearmanr(Z[:, k], outcome_values)

                    outcome_results.append(
                        {
                            "factor": k,
                            "outcome_type": "continuous",
                            "pearson_correlation": corr_coef,
                            "pearson_p_value": p_val,
                            "spearman_correlation": spearman_coef,
                            "spearman_p_value": spearman_p,
                        }
                    )

                else:  # Categorical outcome
                    # Use ANOVA or t-test depending on number of categories
                    unique_categories = np.unique(outcome_values)

                    if len(unique_categories) == 2:
                        # t-test for binary outcomes
                        group_0 = Z[outcome_values == unique_categories[0], k]
                        group_1 = Z[outcome_values == unique_categories[1], k]
                        t_stat, p_val = stats.ttest_ind(group_0, group_1)

                        outcome_results.append(
                            {
                                "factor": k,
                                "outcome_type": "binary",
                                "t_statistic": t_stat,
                                "p_value": p_val,
                                "mean_group_0": np.mean(group_0),
                                "mean_group_1": np.mean(group_1),
                            }
                        )

                    else:
                        # ANOVA for multi-category outcomes
                        groups = [
                            Z[outcome_values == cat, k] for cat in unique_categories
                        ]
                        f_stat, p_val = stats.f_oneway(*groups)

                        outcome_results.append(
                            {
                                "factor": k,
                                "outcome_type": "multiclass",
                                "f_statistic": f_stat,
                                "p_value": p_val,
                                "group_means": [np.mean(group) for group in groups],
                            }
                        )

            results[outcome_name] = outcome_results

        return results

    def analyze_feature_importance(
        self,
        W: List[np.ndarray],
        X_list: List[np.ndarray],
        clinical_outcomes: Dict[str, np.ndarray],
    ) -> Dict:
        """
        Analyze which features are most important for clinical outcomes.

        Args:
            W: Loading matrices per view
            X_list: Data views
            clinical_outcomes: Clinical outcomes dictionary

        Returns:
            Dict containing:
                - feature_rankings: Top features ranked by importance
                - factor_specific_importance: Top features per factor
        """
        results = {}

        # Calculate feature importance scores based on loadings
        all_features = []
        all_loadings = []

        for view_idx, (X, w) in enumerate(zip(X_list, W)):
            n_features = X.shape[1]

            for feature_idx in range(n_features):
                # Feature importance as sum of absolute loadings across factors
                importance_score = np.sum(np.abs(w[feature_idx, :]))

                all_features.append(
                    {
                        "view": view_idx,
                        "feature_index": feature_idx,
                        "global_feature_index": len(all_features),
                        "importance_score": importance_score,
                        "loadings": w[feature_idx, :].tolist(),
                    }
                )
                all_loadings.append(w[feature_idx, :])

        all_loadings = np.array(all_loadings)

        # Rank features by importance
        sorted_features = sorted(
            all_features, key=lambda x: x["importance_score"], reverse=True
        )

        results["feature_rankings"] = {
            "top_features": sorted_features[:20],  # Top 20 features
            "importance_threshold": np.percentile(
                [f["importance_score"] for f in all_features], 90
            ),
        }

        # Analyze factor-specific feature importance
        K = all_loadings.shape[1]
        factor_specific_importance = []

        for k in range(K):
            factor_loadings = all_loadings[:, k]
            top_feature_indices = np.argsort(np.abs(factor_loadings))[::-1][:10]

            factor_specific_importance.append(
                {
                    "factor": k,
                    "top_features": [
                        {
                            "global_index": int(idx),
                            "view": all_features[idx]["view"],
                            "feature_index": all_features[idx]["feature_index"],
                            "loading": float(factor_loadings[idx]),
                        }
                        for idx in top_feature_indices
                    ],
                }
            )

        results["factor_specific_importance"] = factor_specific_importance

        return results

    def validate_biomarker_panels(
        self,
        Z: np.ndarray,
        clinical_outcomes: Dict[str, np.ndarray]
    ) -> Dict:
        """
        Validate performance of different biomarker panel sizes.

        Args:
            Z: SGFA factors (N x K)
            clinical_outcomes: Clinical outcomes dictionary

        Returns:
            Dict with keys for each outcome containing panel validation results
        """
        results = {}

        # Get CV settings from config
        cv_settings = self.config.get("cross_validation", {})
        n_folds = cv_settings.get("n_folds", 5)

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
                    model = LogisticRegression(random_state=42, max_iter=1000)

                    try:
                        cv_scores = cross_val_score(
                            model, Z_selected, outcome_values, cv=n_folds
                        )

                        outcome_results[f"panel_size_{k_features}"] = {
                            "selected_factors": selector.get_support().tolist(),
                            "cv_accuracy_mean": np.mean(cv_scores),
                            "cv_accuracy_std": np.std(cv_scores),
                            "feature_scores": selector.scores_.tolist(),
                        }

                    except Exception as e:
                        self.logger.warning(
                            f"Panel validation failed for {k_features} features: {str(e)}"
                        )

            else:
                # Regression task
                for k_features in [1, 2, 3, min(5, Z.shape[1])]:
                    if k_features > Z.shape[1]:
                        continue

                    selector = SelectKBest(score_func=f_regression, k=k_features)
                    Z_selected = selector.fit_transform(Z, outcome_values)

                    model = Ridge(alpha=1.0)

                    try:
                        cv_scores = cross_val_score(
                            model, Z_selected, outcome_values, cv=n_folds, scoring="r2"
                        )

                        outcome_results[f"panel_size_{k_features}"] = {
                            "selected_factors": selector.get_support().tolist(),
                            "cv_r2_mean": np.mean(cv_scores),
                            "cv_r2_std": np.std(cv_scores),
                            "feature_scores": selector.scores_.tolist(),
                        }

                    except Exception as e:
                        self.logger.warning(
                            f"Panel validation failed for {k_features} features: {str(e)}"
                        )

            results[outcome_name] = outcome_results

        return results

    def test_biomarker_robustness(
        self,
        X_list: List[np.ndarray],
        clinical_outcomes: Dict[str, np.ndarray],
        hypers: Dict,
        args: Dict,
        n_bootstrap: int = 10,
        **kwargs,
    ) -> Dict:
        """
        Test robustness of biomarker discoveries via bootstrap resampling.

        Args:
            X_list: Data views
            clinical_outcomes: Clinical outcomes
            hypers: SGFA hyperparameters
            args: SGFA arguments
            n_bootstrap: Number of bootstrap iterations
            **kwargs: Additional arguments

        Returns:
            Dict with keys for each outcome containing robustness metrics

        Note:
            Requires data_processor to be provided during initialization.
        """
        if self.data_processor is None:
            return {"error": "ClinicalDataProcessor required for robustness testing"}

        results = {}

        bootstrap_associations = {outcome: [] for outcome in clinical_outcomes.keys()}

        for bootstrap_idx in range(n_bootstrap):
            # Bootstrap sample
            n_subjects = X_list[0].shape[0]
            boot_indices = np.random.choice(n_subjects, n_subjects, replace=True)

            X_boot = [X[boot_indices] for X in X_list]
            outcomes_boot = {
                name: values[boot_indices] for name, values in clinical_outcomes.items()
            }

            try:
                # Run SGFA on bootstrap sample
                sgfa_result = self.data_processor.run_sgfa_analysis(
                    X_boot, hypers, args, **kwargs
                )
                Z_boot = sgfa_result["Z"]

                # Calculate associations
                associations = self.analyze_factor_outcome_associations(
                    Z_boot, outcomes_boot
                )

                for outcome_name, outcome_associations in associations.items():
                    bootstrap_associations[outcome_name].append(outcome_associations)

            except Exception as e:
                self.logger.warning(
                    f"Bootstrap iteration {bootstrap_idx} failed: {str(e)}"
                )
                continue

        # Analyze robustness
        for outcome_name, bootstrap_results in bootstrap_associations.items():
            if not bootstrap_results:
                results[outcome_name] = {"error": "No successful bootstrap iterations"}
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
                        if "p_value" in factor_result:
                            p_values.append(factor_result["p_value"])
                        if "pearson_correlation" in factor_result:
                            correlations.append(factor_result["pearson_correlation"])

                stability_metrics = {"factor": k}

                if p_values:
                    # Proportion of significant results (p < 0.05)
                    stability_metrics["significance_rate"] = np.mean(
                        [p < 0.05 for p in p_values]
                    )
                    stability_metrics["mean_p_value"] = np.mean(p_values)

                if correlations:
                    stability_metrics["mean_correlation"] = np.mean(correlations)
                    stability_metrics["correlation_std"] = np.std(correlations)

                factor_stability.append(stability_metrics)

            results[outcome_name] = {
                "factor_stability": factor_stability,
                "n_bootstrap_iterations": len(bootstrap_results),
            }

        return results
