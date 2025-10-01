"""Parkinson's disease subtype discovery and validation.

This module provides utilities for discovering and validating PD subtypes
using SGFA-derived latent factors.
"""

import gc
import logging
import numpy as np
from typing import Dict, List, Optional
from scipy import stats
from scipy.stats import f_oneway, kruskal
from sklearn.cluster import KMeans
from sklearn.metrics import (
    silhouette_score,
    calinski_harabasz_score,
    adjusted_rand_score
)


class PDSubtypeAnalyzer:
    """Parkinson's disease subtype discovery and validation."""

    def __init__(
        self,
        data_processor: Optional[object] = None,
        logger: Optional[logging.Logger] = None
    ):
        """
        Initialize PD subtype analyzer.

        Args:
            data_processor: Optional ClinicalDataProcessor for stability analysis
            logger: Optional logger instance
        """
        self.data_processor = data_processor
        self.logger = logger or logging.getLogger(__name__)

    def discover_pd_subtypes(
        self,
        Z_sgfa: np.ndarray,
        cluster_range: Optional[range] = None
    ) -> Dict:
        """
        Discover PD subtypes using clustering on SGFA factors.

        Args:
            Z_sgfa: SGFA latent factors (N x K)
            cluster_range: Range of cluster numbers to test
                          Default: range(2, min(7, n_subjects//3 + 1))

        Returns:
            Dict containing:
                - optimal_k: Optimal number of clusters
                - cluster_range: Range of clusters tested
                - silhouette_scores: Silhouette scores for each k
                - calinski_scores: Calinski-Harabasz scores for each k
                - solutions: All clustering solutions
                - best_solution: Best clustering solution with labels, centers, scores
        """
        results = {}
        n_subjects, n_factors = Z_sgfa.shape

        # Test different numbers of clusters (2-6 subtypes)
        if cluster_range is None:
            cluster_range = range(2, min(7, n_subjects//3 + 1))

        silhouette_scores = []
        calinski_scores = []
        cluster_solutions = {}

        for k in cluster_range:
            # Run KMeans clustering
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            cluster_labels = kmeans.fit_predict(Z_sgfa)

            # Calculate clustering quality metrics
            sil_score = silhouette_score(Z_sgfa, cluster_labels)
            cal_score = calinski_harabasz_score(Z_sgfa, cluster_labels)

            silhouette_scores.append(sil_score)
            calinski_scores.append(cal_score)

            cluster_solutions[k] = {
                "labels": cluster_labels,
                "centers": kmeans.cluster_centers_,
                "silhouette_score": sil_score,
                "calinski_score": cal_score,
                "inertia": kmeans.inertia_
            }

            # Clean up KMeans object after each iteration
            del kmeans
            gc.collect()

        # Select optimal number of clusters based on silhouette score
        best_k_idx = np.argmax(silhouette_scores)
        best_k = list(cluster_range)[best_k_idx]

        results = {
            "optimal_k": best_k,
            "cluster_range": list(cluster_range),
            "silhouette_scores": silhouette_scores,
            "calinski_scores": calinski_scores,
            "solutions": cluster_solutions,
            "best_solution": cluster_solutions[best_k]
        }

        self.logger.info(f"Optimal number of PD subtypes: {best_k}")
        self.logger.info(f"Best silhouette score: {silhouette_scores[best_k_idx]:.3f}")

        return results

    def validate_subtypes_clinical(
        self,
        subtype_results: Dict,
        clinical_data: Dict,
        Z_sgfa: np.ndarray
    ) -> Dict:
        """
        Validate discovered subtypes against clinical measures.

        Args:
            subtype_results: Results from discover_pd_subtypes()
            clinical_data: Clinical data dictionary
            Z_sgfa: SGFA factors used for subtyping

        Returns:
            Dict with clinical validation results for each measure
        """
        validation_results = {}
        best_solution = subtype_results["best_solution"]
        cluster_labels = best_solution["labels"]

        # Group subjects by discovered subtype
        unique_labels = np.unique(cluster_labels)
        subtype_groups = {
            label: np.where(cluster_labels == label)[0] for label in unique_labels
        }

        clinical_validation = {}

        # Test each clinical measure for subtype differences
        for measure_name, measure_data in clinical_data.items():
            if not isinstance(measure_data, (np.ndarray, list)):
                continue

            measure_array = np.array(measure_data)
            if len(measure_array) != len(cluster_labels):
                continue

            # Group clinical measures by subtype
            groups = [measure_array[indices] for indices in subtype_groups.values()]

            # Remove groups with insufficient data
            valid_groups = [
                group for group in groups
                if len(group) > 1 and not np.all(np.isnan(group))
            ]

            if len(valid_groups) < 2:
                continue

            # Statistical tests for subtype differences
            try:
                # ANOVA test (parametric)
                f_stat, p_anova = f_oneway(*valid_groups)

                # Kruskal-Wallis test (non-parametric)
                h_stat, p_kruskal = kruskal(*valid_groups)

                clinical_validation[measure_name] = {
                    "f_statistic": f_stat,
                    "p_value_anova": p_anova,
                    "h_statistic": h_stat,
                    "p_value_kruskal": p_kruskal,
                    "group_means": [np.nanmean(group) for group in valid_groups],
                    "group_stds": [np.nanstd(group) for group in valid_groups],
                    "significant": p_anova < 0.05 or p_kruskal < 0.05
                }

            except Exception as e:
                self.logger.warning(
                    f"Statistical test failed for {measure_name}: {str(e)}"
                )

        validation_results["clinical_associations"] = clinical_validation

        # Count how many clinical measures show significant differences
        significant_measures = sum(
            1 for val in clinical_validation.values() if val.get("significant", False)
        )
        validation_results["summary"] = {
            "n_measures_tested": len(clinical_validation),
            "n_significant_associations": significant_measures,
            "proportion_significant": (
                significant_measures / len(clinical_validation)
                if clinical_validation else 0
            )
        }

        return validation_results

    def analyze_pd_subtype_stability(
        self,
        X_list: List[np.ndarray],
        hypers: Dict,
        args: Dict,
        n_runs: int = 5,
        **kwargs
    ) -> Dict:
        """
        Analyze stability of discovered subtypes across multiple runs.

        Args:
            X_list: Data views
            hypers: SGFA hyperparameters
            args: SGFA arguments
            n_runs: Number of independent runs
            **kwargs: Additional arguments

        Returns:
            Dict with stability metrics including ARI scores

        Note:
            Requires data_processor to be provided during initialization.
        """
        if self.data_processor is None:
            return {"error": "ClinicalDataProcessor required for stability analysis"}

        stability_results = {}
        all_solutions = []

        for run in range(n_runs):
            # Run SGFA with different random seed
            run_args = args.copy()
            run_args['random_seed'] = args.get('random_seed', 42) + run

            try:
                sgfa_result = self.data_processor.run_sgfa_analysis(
                    X_list, hypers, run_args, **kwargs
                )
                Z_sgfa = sgfa_result["Z"]

                # Discover subtypes for this run
                subtype_result = self.discover_pd_subtypes(Z_sgfa)
                all_solutions.append(subtype_result["best_solution"]["labels"])

            except Exception as e:
                self.logger.warning(f"Stability run {run} failed: {e}")

        if len(all_solutions) < 2:
            return {"error": "Insufficient successful runs for stability analysis"}

        # Calculate pairwise ARI scores
        ari_scores = []
        for i in range(len(all_solutions)):
            for j in range(i+1, len(all_solutions)):
                ari = adjusted_rand_score(all_solutions[i], all_solutions[j])
                ari_scores.append(ari)

        mean_ari = np.mean(ari_scores)
        stability_results = {
            "n_successful_runs": len(all_solutions),
            "ari_scores": ari_scores,
            "mean_ari": mean_ari,
            "std_ari": np.std(ari_scores),
            "min_ari": np.min(ari_scores),
            "max_ari": np.max(ari_scores),
            "stability_grade": (
                "High" if mean_ari > 0.7
                else "Medium" if mean_ari > 0.4
                else "Low"
            )
        }

        self.logger.info(
            f"Subtype stability: Mean ARI = {mean_ari:.3f} "
            f"({stability_results['stability_grade']})"
        )

        return stability_results

    def analyze_subtype_factor_patterns(
        self,
        Z_sgfa: np.ndarray,
        subtype_results: Dict,
        sgfa_result: Dict
    ) -> Dict:
        """
        Analyze factor patterns for each discovered subtype.

        Args:
            Z_sgfa: SGFA latent factors (N x K)
            subtype_results: Subtype discovery results
            sgfa_result: SGFA result dictionary (not currently used, for future use)

        Returns:
            Dict with keys for each subtype containing factor statistics
        """
        interpretation_results = {}
        best_solution = subtype_results["best_solution"]
        cluster_labels = best_solution["labels"]

        unique_labels = np.unique(cluster_labels)

        for label in unique_labels:
            subtype_indices = np.where(cluster_labels == label)[0]
            subtype_factors = Z_sgfa[subtype_indices]

            # Calculate subtype-specific factor statistics
            factor_means = np.mean(subtype_factors, axis=0)
            factor_stds = np.std(subtype_factors, axis=0)

            # Identify characteristic factors (high absolute mean, low std)
            factor_importance = np.abs(factor_means) / (factor_stds + 1e-8)
            top_factors = np.argsort(factor_importance)[::-1]

            interpretation_results[f"subtype_{label}"] = {
                "n_subjects": len(subtype_indices),
                "factor_means": factor_means.tolist(),
                "factor_stds": factor_stds.tolist(),
                "factor_importance": factor_importance.tolist(),
                "top_factors": top_factors[:3].tolist(),  # Top 3 characteristic factors
                "factor_center": best_solution["centers"][label].tolist()
            }

        return interpretation_results

    def analyze_subtype_stability(
        self,
        X_list: List[np.ndarray],
        clinical_labels: np.ndarray,
        hypers: Dict,
        args: Dict,
        n_bootstrap: int = 10,
        **kwargs,
    ) -> Dict:
        """
        Analyze subtype stability using bootstrap resampling.

        Args:
            X_list: Data views
            clinical_labels: Clinical labels
            hypers: SGFA hyperparameters
            args: SGFA arguments
            n_bootstrap: Number of bootstrap iterations
            **kwargs: Additional arguments

        Returns:
            Dict with bootstrap stability analysis

        Note:
            Requires data_processor to be provided during initialization.
            This method is for general subtype stability (different from
            analyze_pd_subtype_stability which tests across multiple complete runs).
        """
        if self.data_processor is None:
            return {"error": "ClinicalDataProcessor required for stability analysis"}

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
                sgfa_result = self.data_processor.run_sgfa_analysis(
                    X_boot, hypers, args, **kwargs
                )
                Z_boot = sgfa_result["Z"]

                # Train classifier on bootstrap factors
                from sklearn.linear_model import LogisticRegression
                model = LogisticRegression(random_state=42, max_iter=1000)
                model.fit(Z_boot, labels_boot)

                # Predict on original (out-of-bootstrap) samples
                oob_mask = np.array([i not in boot_indices for i in range(n_subjects)])
                if np.any(oob_mask):
                    # For simplicity, predict on all samples using bootstrap model
                    sgfa_original = self.data_processor.run_sgfa_analysis(
                        X_list, hypers, args, **kwargs
                    )
                    predictions = model.predict(sgfa_original["Z"])
                    bootstrap_predictions.append(predictions)

            except Exception as e:
                self.logger.warning(
                    f"Bootstrap iteration {bootstrap_idx} failed: {str(e)}"
                )
                continue

        if bootstrap_predictions:
            # Calculate stability metrics
            bootstrap_predictions = np.array(bootstrap_predictions)

            # Agreement between bootstrap predictions
            agreement_matrix = np.zeros((n_subjects, n_subjects))
            for i in range(len(bootstrap_predictions)):
                for j in range(len(bootstrap_predictions)):
                    if i != j:
                        agreement = np.mean(
                            bootstrap_predictions[i] == bootstrap_predictions[j]
                        )
                        agreement_matrix[i, j] = agreement

            # Average agreement per subject
            mean_agreement = np.mean(agreement_matrix[agreement_matrix > 0])

            results = {
                "n_bootstrap_iterations": len(bootstrap_predictions),
                "mean_prediction_agreement": mean_agreement,
                "prediction_stability": "High" if mean_agreement > 0.8 else "Medium" if mean_agreement > 0.6 else "Low"
            }

        else:
            results = {"error": "No successful bootstrap iterations"}

        return results

    def analyze_clinical_subtypes_neuroimaging(
        self,
        sgfa_cv_results: Dict,
        clinical_data: Dict
    ) -> Dict:
        """
        Analyze clinical subtypes using neuroimaging factors from CV folds.

        Args:
            sgfa_cv_results: SGFA CV results containing fold_results
            clinical_data: Clinical data dictionary with 'diagnosis' key

        Returns:
            Dict with subtype analysis per diagnosis group
        """
        try:
            successful_folds = [
                f for f in sgfa_cv_results.get("fold_results", [])
                if f.get("success", False)
            ]

            if not successful_folds:
                return {"error": "No successful folds for subtype analysis"}

            # Combine factors from all folds for subtype analysis
            all_factors = []
            all_diagnoses = []

            for fold in successful_folds:
                train_result = fold.get("train_result", {})
                Z_train = train_result.get("Z")
                train_idx = fold.get("train_idx", [])

                if Z_train is not None:
                    all_factors.append(Z_train)
                    all_diagnoses.extend(np.array(clinical_data["diagnosis"])[train_idx])

            if all_factors:
                # Concatenate all factors
                combined_factors = np.vstack(all_factors)
                all_diagnoses = np.array(all_diagnoses)

                # Cluster analysis within each diagnosis group
                subtype_results = {}

                unique_diagnoses = np.unique(all_diagnoses)
                for diagnosis in unique_diagnoses:
                    diag_mask = all_diagnoses == diagnosis
                    diag_factors = combined_factors[diag_mask]

                    if len(diag_factors) > 3:  # Need enough samples for clustering
                        # Try different numbers of clusters
                        best_k = 2
                        best_score = -1

                        for k in range(2, min(6, len(diag_factors))):
                            try:
                                kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
                                cluster_labels = kmeans.fit_predict(diag_factors)
                                score = silhouette_score(diag_factors, cluster_labels)

                                if score > best_score:
                                    best_score = score
                                    best_k = k
                            except Exception:
                                pass

                        # Final clustering with best k
                        if best_score > 0:
                            kmeans = KMeans(n_clusters=best_k, random_state=42, n_init=10)
                            cluster_labels = kmeans.fit_predict(diag_factors)

                            subtype_results[diagnosis] = {
                                "n_subtypes": best_k,
                                "silhouette_score": best_score,
                                "cluster_centers": kmeans.cluster_centers_.tolist(),
                                "n_subjects": len(diag_factors)
                            }

                return {
                    "subtype_analysis": subtype_results,
                    "total_subjects_analyzed": len(combined_factors),
                    "success": True
                }

            else:
                return {"error": "No factors available for subtype analysis"}

        except Exception as e:
            return {"error": f"Subtype analysis failed: {str(e)}"}
