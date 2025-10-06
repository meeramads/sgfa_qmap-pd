"""External cohort validation and model transferability analysis.

This module provides utilities for validating models across different cohorts
and analyzing the transferability of learned representations.

⚠️  IMPORTANT: This module is FUTURE WORK and not implemented in the current project scope.
External validation requires PD datasets with similar multimodal data (qMRI/imaging + clinical).
Many widely-available PD datasets lack quantitative MRI or other imaging data, which are the
focus of this project for PD subtyping (traditionally done with clinical data alone). This
infrastructure is provided for future external validation when suitable datasets become available.
"""

import logging
import numpy as np
from typing import Dict, Optional, Tuple
from scipy import stats
from sklearn.cross_decomposition import CCA


class ExternalValidator:
    """External cohort validation and model transferability."""

    def __init__(
        self,
        logger: Optional[logging.Logger] = None
    ):
        """
        Initialize external validator.

        Args:
            logger: Optional logger instance
        """
        self.logger = logger or logging.getLogger(__name__)

    def compare_factor_distributions(
        self,
        Z_train: np.ndarray,
        Z_test: np.ndarray,
        cohort_names: Tuple[str, str] = ("Training", "Test")
    ) -> Dict:
        """
        Compare factor distributions across cohorts.

        Args:
            Z_train: Training cohort factors (N_train x K)
            Z_test: Test cohort factors (N_test x K)
            cohort_names: Names for the cohorts (for reporting)

        Returns:
            Dict containing:
                - factor_comparisons: List of statistical comparisons per factor
                  including KS test, t-test, Cohen's d, means, stds
                - overall_similarity: Summary statistics including similarity score
                  and mean effect size
        """
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

            distribution_comparisons.append(
                {
                    "factor": k,
                    "ks_statistic": ks_stat,
                    "ks_p_value": ks_p,
                    "t_statistic": t_stat,
                    "t_p_value": t_p,
                    "cohens_d": cohens_d,
                    "train_mean": np.mean(Z_train[:, k]),
                    "test_mean": np.mean(Z_test[:, k]),
                    "train_std": np.std(Z_train[:, k]),
                    "test_std": np.std(Z_test[:, k]),
                }
            )

        results["factor_comparisons"] = distribution_comparisons

        # Overall distribution similarity
        significant_differences = sum(
            1 for comp in distribution_comparisons if comp["ks_p_value"] < 0.05
        )
        results["overall_similarity"] = {
            "n_significant_differences": significant_differences,
            "similarity_score": 1 - (significant_differences / K),
            "mean_effect_size": np.mean(
                [abs(comp["cohens_d"]) for comp in distribution_comparisons]
            ),
        }

        return results

    def analyze_model_transferability(
        self,
        train_result: Dict,
        test_result: Dict,
        labels_train: np.ndarray,
        labels_test: np.ndarray,
    ) -> Dict:
        """
        Analyze transferability of model across cohorts.

        Args:
            train_result: Training cohort SGFA result (must contain 'Z', 'log_likelihood')
            test_result: Test cohort SGFA result (must contain 'Z', 'log_likelihood')
            labels_train: Training cohort labels (N_train,)
            labels_test: Test cohort labels (N_test,)

        Returns:
            Dict containing:
                - model_fit_comparison: Comparison of log likelihoods
                - factor_space_similarity: Canonical correlation analysis results
                - demographic_transferability: Class distribution comparison
        """
        results = {}

        # Compare model fits
        results["model_fit_comparison"] = {
            "train_likelihood": train_result.get("log_likelihood", np.nan),
            "test_likelihood": test_result.get("log_likelihood", np.nan),
            "likelihood_drop": (
                train_result.get("log_likelihood", 0)
                - test_result.get("log_likelihood", 0)
            ),
        }

        # Factor space similarity
        Z_train = train_result["Z"]
        Z_test = test_result["Z"]

        # Calculate canonical correlations between factor spaces
        min_components = min(
            Z_train.shape[1], Z_test.shape[1], Z_train.shape[0], Z_test.shape[0]
        )
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

                results["factor_space_similarity"] = {
                    "canonical_correlations": canonical_correlations,
                    "mean_canonical_correlation": np.mean(canonical_correlations),
                    "transferability_score": np.mean(canonical_correlations),
                }

            except Exception as e:
                self.logger.warning(f"CCA analysis failed: {str(e)}")
                results["factor_space_similarity"] = {"error": str(e)}

        # Demographic transferability (if applicable)
        if len(np.unique(labels_train)) == len(np.unique(labels_test)):
            # Compare class distributions
            train_class_dist = np.bincount(labels_train) / len(labels_train)
            test_class_dist = np.bincount(labels_test) / len(labels_test)

            # KL divergence between class distributions
            kl_divergence = stats.entropy(test_class_dist, train_class_dist)

            results["demographic_transferability"] = {
                "train_class_distribution": train_class_dist.tolist(),
                "test_class_distribution": test_class_dist.tolist(),
                "kl_divergence": kl_divergence,
                "distribution_similarity": np.exp(-kl_divergence),
            }

        return results
