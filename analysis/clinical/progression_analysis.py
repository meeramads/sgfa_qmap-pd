"""Disease progression modeling and analysis.

This module provides utilities for analyzing disease progression patterns,
including cross-sectional correlations, longitudinal analysis, and prediction
of clinical milestones.
"""

import logging
import numpy as np
from typing import Dict, Optional
from scipy import stats
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split


class DiseaseProgressionAnalyzer:
    """Disease progression modeling and analysis."""

    def __init__(
        self,
        classifier: Optional[object] = None,
        logger: Optional[logging.Logger] = None
    ):
        """
        Initialize disease progression analyzer.

        Args:
            classifier: Optional ClinicalClassifier instance for milestone prediction
            logger: Optional logger instance
        """
        self.classifier = classifier
        self.logger = logger or logging.getLogger(__name__)

    def analyze_cross_sectional_correlations(
        self,
        Z: np.ndarray,
        progression_scores: np.ndarray
    ) -> Dict:
        """
        Analyze cross-sectional correlations between factors and progression.

        Args:
            Z: Latent factors (N x K)
            progression_scores: Disease progression scores (N,)

        Returns:
            Dict containing:
                - factor_progression_correlations: List of correlations per factor
                  including Pearson and Spearman correlations
                - multiple_regression: Multiple regression analysis with R² and
                  feature importances
        """
        results = {}

        K = Z.shape[1]
        correlations = []

        for k in range(K):
            corr_coef, p_val = stats.pearsonr(Z[:, k], progression_scores)

            # Also calculate Spearman correlation for non-linear relationships
            spearman_coef, spearman_p = stats.spearmanr(Z[:, k], progression_scores)

            correlations.append(
                {
                    "factor": k,
                    "pearson_correlation": corr_coef,
                    "pearson_p_value": p_val,
                    "spearman_correlation": spearman_coef,
                    "spearman_p_value": spearman_p,
                    "correlation_strength": (
                        "strong"
                        if abs(corr_coef) > 0.5
                        else "moderate" if abs(corr_coef) > 0.3 else "weak"
                    ),
                }
            )

        results["factor_progression_correlations"] = correlations

        # Multiple regression analysis
        model = LinearRegression()
        model.fit(Z, progression_scores)
        predictions = model.predict(Z)

        results["multiple_regression"] = {
            "r2_score": r2_score(progression_scores, predictions),
            "coefficients": model.coef_.tolist(),
            "intercept": model.intercept_,
            "feature_importance": np.abs(model.coef_) / np.sum(np.abs(model.coef_)),
        }

        return results

    def analyze_longitudinal_progression(
        self,
        Z: np.ndarray,
        progression_scores: np.ndarray,
        time_points: np.ndarray,
        subject_ids: np.ndarray,
    ) -> Dict:
        """
        Analyze longitudinal disease progression patterns.

        Args:
            Z: Latent factors (N x K) for all time points
            progression_scores: Progression scores (N,) for all time points
            time_points: Time points (N,) in consistent units
            subject_ids: Subject IDs (N,) to group longitudinal measurements

        Returns:
            Dict containing:
                - n_longitudinal_subjects: Number of subjects with multiple time points
                - progression_analysis: Mean and std of progression rates
                - factor_change_analysis: Factor change rates and their correlation
                  with progression
        """
        results = {}

        # Group data by subject
        unique_subjects = np.unique(subject_ids)
        longitudinal_data = []

        for subject_id in unique_subjects:
            subject_mask = subject_ids == subject_id
            if np.sum(subject_mask) > 1:  # At least 2 time points
                subject_data = {
                    "subject_id": subject_id,
                    "time_points": time_points[subject_mask],
                    "progression_scores": progression_scores[subject_mask],
                    "factors": Z[subject_mask],
                    "n_timepoints": np.sum(subject_mask),
                }
                longitudinal_data.append(subject_data)

        results["n_longitudinal_subjects"] = len(longitudinal_data)

        if longitudinal_data:
            # Analyze progression rates
            progression_rates = []
            factor_changes = []

            for subject_data in longitudinal_data:
                times = subject_data["time_points"]
                scores = subject_data["progression_scores"]
                factors = subject_data["factors"]

                # Calculate progression rate
                if len(times) > 1:
                    rate = (scores[-1] - scores[0]) / (times[-1] - times[0])
                    progression_rates.append(rate)

                    # Calculate factor change rates
                    factor_change_rates = []
                    for k in range(factors.shape[1]):
                        factor_rate = (factors[-1, k] - factors[0, k]) / (
                            times[-1] - times[0]
                        )
                        factor_change_rates.append(factor_rate)
                    factor_changes.append(factor_change_rates)

            results["progression_analysis"] = {
                "mean_progression_rate": np.mean(progression_rates),
                "std_progression_rate": np.std(progression_rates),
                "progression_rates": progression_rates,
            }

            if factor_changes:
                factor_changes = np.array(factor_changes)
                results["factor_change_analysis"] = {
                    "mean_factor_changes": np.mean(factor_changes, axis=0).tolist(),
                    "std_factor_changes": np.std(factor_changes, axis=0).tolist(),
                    "factor_progression_correlations": [],
                }

                # Correlate factor changes with progression rates
                for k in range(factor_changes.shape[1]):
                    corr, p_val = stats.pearsonr(
                        factor_changes[:, k], progression_rates
                    )
                    results["factor_change_analysis"][
                        "factor_progression_correlations"
                    ].append({"factor": k, "correlation": corr, "p_value": p_val})

        return results

    def validate_progression_prediction(
        self,
        Z: np.ndarray,
        progression_scores: np.ndarray,
        time_points: Optional[np.ndarray] = None
    ) -> Dict:
        """
        Validate progression prediction using latent factors.

        Args:
            Z: Latent factors (N x K)
            progression_scores: Progression scores (N,)
            time_points: Optional time points (not currently used, for future use)

        Returns:
            Dict with keys for each model containing:
                - mse: Mean squared error
                - mae: Mean absolute error
                - r2_score: R² score
                - rmse: Root mean squared error
                - feature_importances: Feature importances (for random forest)
        """
        results = {}

        # Split data for validation
        X_train, X_test, y_train, y_test = train_test_split(
            Z, progression_scores, test_size=0.3, random_state=42
        )

        # Test different prediction models
        models = {
            "linear_regression": Ridge(alpha=1.0),
            "random_forest": RandomForestRegressor(n_estimators=100, random_state=42),
        }

        for model_name, model in models.items():
            try:
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)

                mse = mean_squared_error(y_test, y_pred)
                mae = mean_absolute_error(y_test, y_pred)
                r2 = model.score(X_test, y_test)

                results[model_name] = {
                    "mse": mse,
                    "mae": mae,
                    "r2_score": r2,
                    "rmse": np.sqrt(mse),
                }

                # Feature importance for random forest
                if hasattr(model, "feature_importances_"):
                    results[model_name][
                        "feature_importances"
                    ] = model.feature_importances_.tolist()

            except Exception as e:
                self.logger.warning(
                    f"Progression prediction failed for {model_name}: {str(e)}"
                )
                results[model_name] = {"error": str(e)}

        return results

    def analyze_clinical_milestones(
        self,
        Z: np.ndarray,
        progression_scores: np.ndarray,
        time_points: Optional[np.ndarray] = None
    ) -> Dict:
        """
        Analyze prediction of clinical milestones.

        Args:
            Z: Latent factors (N x K)
            progression_scores: Progression scores (N,)
            time_points: Optional time points (not currently used, for future use)

        Returns:
            Dict with keys for each milestone containing classification results

        Note:
            If a classifier was provided during initialization, this method will
            use it for milestone prediction. Otherwise, basic statistics are returned.
        """
        results = {}

        # Define clinical milestones based on progression scores
        # (These would be clinically meaningful thresholds)
        milestones = {
            "mild_progression": np.percentile(progression_scores, 33),
            "moderate_progression": np.percentile(progression_scores, 66),
            "severe_progression": np.percentile(progression_scores, 90),
        }

        for milestone_name, threshold in milestones.items():
            # Create binary outcome
            milestone_reached = (progression_scores >= threshold).astype(int)

            if np.sum(milestone_reached) > 5:  # Ensure sufficient positive cases
                if self.classifier is not None:
                    # Use provided classifier for prediction
                    try:
                        milestone_result = self.classifier.test_factor_classification(
                            Z, milestone_reached, f"milestone_{milestone_name}"
                        )
                        results[milestone_name] = milestone_result
                    except Exception as e:
                        self.logger.warning(
                            f"Milestone prediction failed for {milestone_name}: {str(e)}"
                        )
                        results[milestone_name] = {"error": str(e)}
                else:
                    # Basic statistics if no classifier provided
                    results[milestone_name] = {
                        "threshold": threshold,
                        "n_reached": np.sum(milestone_reached),
                        "proportion_reached": np.mean(milestone_reached),
                    }

        return results
