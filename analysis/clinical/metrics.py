"""Clinical metrics calculation and interpretability analysis.

This module provides utilities for calculating clinical validation metrics
and analyzing the clinical interpretability of latent factors.
"""

import logging
import numpy as np
from typing import Dict, List, Optional, Union
from scipy import stats
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score
)


class ClinicalMetrics:
    """Calculate clinical validation metrics and interpretability analyses."""

    def __init__(
        self,
        metrics_list: Optional[List[str]] = None,
        logger: Optional[logging.Logger] = None
    ):
        """
        Initialize clinical metrics calculator.

        Args:
            metrics_list: List of metrics to calculate
                         Default: ["accuracy", "precision", "recall", "f1_score", "roc_auc"]
            logger: Optional logger instance
        """
        self.metrics_list = metrics_list or [
            "accuracy", "precision", "recall", "f1_score", "roc_auc"
        ]
        self.logger = logger or logging.getLogger(__name__)

    def calculate_detailed_metrics(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_pred_proba: np.ndarray,
        requested_metrics: Optional[List[str]] = None
    ) -> Dict[str, Union[float, List]]:
        """
        Calculate detailed classification metrics.

        Args:
            y_true: True labels (N,)
            y_pred: Predicted labels (N,)
            y_pred_proba: Predicted probabilities (N, n_classes)
            requested_metrics: Optional list of metrics to calculate
                              Uses self.metrics_list if None

        Returns:
            Dict containing:
                - accuracy: Accuracy score
                - precision: Macro-averaged precision
                - recall: Macro-averaged recall
                - f1_score: Macro-averaged F1 score
                - roc_auc: ROC AUC score (binary or multiclass)
                - confusion_matrix: Confusion matrix as list
                - specificity: Specificity (binary only)
                - npv: Negative predictive value (binary only)
                - ppv: Positive predictive value (binary only)
        """
        if requested_metrics is None:
            requested_metrics = self.metrics_list

        metrics = {}

        # Calculate only requested metrics
        if "accuracy" in requested_metrics:
            metrics["accuracy"] = accuracy_score(y_true, y_pred)

        if "precision" in requested_metrics:
            metrics["precision"] = precision_score(
                y_true, y_pred, average="macro", zero_division=0
            )

        if "recall" in requested_metrics:
            metrics["recall"] = recall_score(
                y_true, y_pred, average="macro", zero_division=0
            )

        if "f1_score" in requested_metrics:
            metrics["f1_score"] = f1_score(
                y_true, y_pred, average="macro", zero_division=0
            )

        # ROC AUC (for binary or multiclass)
        if "roc_auc" in requested_metrics:
            try:
                if len(np.unique(y_true)) == 2:
                    metrics["roc_auc"] = roc_auc_score(y_true, y_pred_proba[:, 1])
                else:
                    metrics["roc_auc"] = roc_auc_score(
                        y_true, y_pred_proba, multi_class="ovr"
                    )
            except BaseException:
                metrics["roc_auc"] = np.nan

        # Confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        metrics["confusion_matrix"] = cm.tolist()

        # Class-specific metrics (binary classification only)
        if len(np.unique(y_true)) == 2:
            tn, fp, fn, tp = cm.ravel()
            metrics["specificity"] = tn / (tn + fp) if (tn + fp) > 0 else np.nan
            metrics["npv"] = tn / (tn + fn) if (tn + fn) > 0 else np.nan
            metrics["ppv"] = tp / (tp + fp) if (tp + fp) > 0 else np.nan

        return metrics

    def analyze_clinical_interpretability(
        self,
        Z: np.ndarray,
        clinical_labels: np.ndarray,
        sgfa_result: Dict
    ) -> Dict:
        """
        Analyze clinical interpretability of latent factors.

        Args:
            Z: Latent factors (N x K)
            clinical_labels: Clinical labels (N,)
            sgfa_result: SGFA result dictionary containing 'W' (loadings)

        Returns:
            Dict containing:
                - factor_label_associations: List of dicts with statistical tests
                  for each factor's association with clinical labels
                - interpretability_analysis: List of dicts with sparsity and
                  concentration scores for each factor
        """
        results = {}
        K = Z.shape[1]

        # Factor-label correlations
        factor_correlations = []
        for k in range(K):
            if len(np.unique(clinical_labels)) > 2:
                # For multiclass, use ANOVA F-statistic
                f_stat, p_val = stats.f_oneway(
                    *[
                        Z[clinical_labels == label, k]
                        for label in np.unique(clinical_labels)
                    ]
                )
                factor_correlations.append(
                    {
                        "factor": k,
                        "f_statistic": f_stat,
                        "p_value": p_val,
                        "effect_size": f_stat
                        / (
                            f_stat
                            + len(clinical_labels)
                            - len(np.unique(clinical_labels))
                        ),
                    }
                )
            else:
                # For binary, use t-test
                group_0 = Z[clinical_labels == 0, k]
                group_1 = Z[clinical_labels == 1, k]
                t_stat, p_val = stats.ttest_ind(group_0, group_1)

                # Cohen's d effect size
                pooled_std = np.sqrt(
                    (
                        (len(group_0) - 1) * np.var(group_0, ddof=1)
                        + (len(group_1) - 1) * np.var(group_1, ddof=1)
                    )
                    / (len(group_0) + len(group_1) - 2)
                )
                cohens_d = (np.mean(group_1) - np.mean(group_0)) / pooled_std

                factor_correlations.append(
                    {
                        "factor": k,
                        "t_statistic": t_stat,
                        "p_value": p_val,
                        "cohens_d": cohens_d,
                        "effect_size": abs(cohens_d),
                    }
                )

        results["factor_label_associations"] = factor_correlations

        # Factor interpretability scores
        W = sgfa_result["W"]
        interpretability_scores = []

        for k in range(K):
            # Calculate sparsity (proportion of near-zero loadings)
            all_loadings = np.concatenate([w[:, k] for w in W])
            sparsity = np.mean(np.abs(all_loadings) < 0.1)

            # Calculate loading concentration (how concentrated are the large loadings)
            sorted_abs_loadings = np.sort(np.abs(all_loadings))[::-1]
            top_10_percent = int(len(sorted_abs_loadings) * 0.1)
            concentration = np.sum(sorted_abs_loadings[:top_10_percent]) / np.sum(
                sorted_abs_loadings
            )

            interpretability_scores.append(
                {
                    "factor": k,
                    "sparsity": sparsity,
                    "loading_concentration": concentration,
                    "interpretability_score": sparsity * concentration,
                }
            )

        results["interpretability_analysis"] = interpretability_scores

        return results
