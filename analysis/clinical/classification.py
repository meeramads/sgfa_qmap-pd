"""Clinical classification and prediction utilities.

This module provides utilities for testing classification performance
using SGFA-derived features for clinical prediction tasks.
"""

import logging
import numpy as np
from typing import Dict, Optional
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report

from .metrics import ClinicalMetrics


class ClinicalClassifier:
    """Clinical classification and prediction utilities."""

    def __init__(
        self,
        metrics_calculator: ClinicalMetrics,
        classification_models: Optional[Dict] = None,
        config: Optional[Dict] = None,
        logger: Optional[logging.Logger] = None
    ):
        """
        Initialize clinical classifier.

        Args:
            metrics_calculator: ClinicalMetrics instance for computing metrics
            classification_models: Dict of sklearn classifiers
                                  Default: logistic regression, random forest, SVM
            config: Optional configuration dict with cross_validation settings
            logger: Optional logger instance
        """
        self.metrics = metrics_calculator
        self.classification_models = classification_models or {
            "logistic_regression": LogisticRegression(random_state=42, max_iter=1000),
            "random_forest": RandomForestClassifier(random_state=42, n_estimators=100),
            "svm": SVC(random_state=42, probability=True)
        }
        self.config = config or {}
        self.logger = logger or logging.getLogger(__name__)

    def validate_factors_clinical_prediction(
        self,
        sgfa_cv_results: Dict,
        clinical_data: Dict
    ) -> Dict:
        """
        Validate extracted factors for clinical prediction across CV folds.

        Args:
            sgfa_cv_results: SGFA cross-validation results containing fold_results
            clinical_data: Clinical data dictionary with 'diagnosis' key

        Returns:
            Dict containing:
                - accuracy: Overall accuracy across folds
                - f1_score: Weighted F1 score
                - precision: Weighted precision
                - recall: Weighted recall
                - classification_report: Detailed classification report
                - n_predictions: Total number of predictions
                - success: Whether validation succeeded
        """
        try:
            successful_folds = [
                f for f in sgfa_cv_results.get("fold_results", [])
                if f.get("success", False)
            ]

            if not successful_folds:
                return {"error": "No successful folds for validation"}

            # Aggregate predictions across folds
            all_predictions = []
            all_true_labels = []

            for fold in successful_folds:
                test_metrics = fold.get("test_metrics", {})
                Z_test = test_metrics.get("Z_test")
                test_diagnoses = test_metrics.get("test_diagnoses")

                if Z_test is not None and test_diagnoses is not None:
                    # Get training data for this fold
                    train_idx = fold.get("train_idx", [])
                    train_result = fold.get("train_result", {})
                    Z_train = train_result.get("Z")

                    if Z_train is not None:
                        train_diagnoses = np.array(clinical_data["diagnosis"])[train_idx]

                        # Encode labels
                        le = LabelEncoder()
                        le.fit(clinical_data["diagnosis"])
                        train_labels_encoded = le.transform(train_diagnoses)
                        test_labels_encoded = le.transform(test_diagnoses)

                        # Train classifier (using Random Forest by default)
                        clf = RandomForestClassifier(random_state=42, n_estimators=100)
                        clf.fit(Z_train, train_labels_encoded)

                        # Predict
                        predictions = clf.predict(Z_test)
                        all_predictions.extend(predictions)
                        all_true_labels.extend(test_labels_encoded)

            if all_predictions:
                # Calculate performance metrics
                from sklearn.metrics import (
                    accuracy_score, f1_score, precision_score, recall_score
                )

                accuracy = accuracy_score(all_true_labels, all_predictions)
                f1 = f1_score(all_true_labels, all_predictions, average='weighted')
                precision = precision_score(
                    all_true_labels, all_predictions, average='weighted'
                )
                recall = recall_score(all_true_labels, all_predictions, average='weighted')

                return {
                    "accuracy": accuracy,
                    "f1_score": f1,
                    "precision": precision,
                    "recall": recall,
                    "classification_report": classification_report(
                        all_true_labels, all_predictions
                    ),
                    "n_predictions": len(all_predictions),
                    "success": True
                }

            else:
                return {"error": "No predictions could be made"}

        except Exception as e:
            return {"error": f"Clinical prediction validation failed: {str(e)}"}

    def test_factor_classification(
        self,
        features: np.ndarray,
        labels: np.ndarray,
        feature_type: str
    ) -> Dict:
        """
        Test classification performance using given features.

        Args:
            features: Feature matrix (N x D)
            labels: Classification labels (N,)
            feature_type: Description of feature type (for reporting)

        Returns:
            Dict with keys for each classifier containing:
                - cross_validation: CV scores per metric (mean, std, scores)
                - detailed_metrics: Full metrics dict from ClinicalMetrics
                - feature_type: Type of features used
        """
        results = {}

        # Ensure we have valid data
        if len(np.unique(labels)) < 2:
            return {"error": "Insufficient label diversity for classification"}

        # Get CV settings from config
        cv_settings = self.config.get("cross_validation", {})
        n_folds = cv_settings.get("n_folds", 5)
        stratified = cv_settings.get("stratified", True)
        random_seed = cv_settings.get("random_seed", 42)

        if stratified:
            cv = StratifiedKFold(
                n_splits=n_folds, shuffle=True, random_state=random_seed
            )
        else:
            from sklearn.model_selection import KFold
            cv = KFold(n_splits=n_folds, shuffle=True, random_state=random_seed)

        for model_name, model in self.classification_models.items():
            try:
                # Cross-validation scores
                cv_scores = {}
                for metric in [
                    "accuracy",
                    "precision_macro",
                    "recall_macro",
                    "f1_macro",
                ]:
                    scores = cross_val_score(
                        model, features, labels, cv=cv, scoring=metric
                    )
                    cv_scores[metric] = {
                        "mean": np.mean(scores),
                        "std": np.std(scores),
                        "scores": scores.tolist(),
                    }

                # Train on full data for detailed metrics
                model.fit(features, labels)
                y_pred = model.predict(features)
                y_pred_proba = model.predict_proba(features)

                # Calculate detailed metrics
                detailed_metrics = self.metrics.calculate_detailed_metrics(
                    labels, y_pred, y_pred_proba
                )

                results[model_name] = {
                    "cross_validation": cv_scores,
                    "detailed_metrics": detailed_metrics,
                    "feature_type": feature_type,
                }

            except Exception as e:
                self.logger.warning(f"Classification failed for {model_name}: {str(e)}")
                results[model_name] = {"error": str(e)}

        return results

    def test_cross_cohort_classification(
        self,
        Z_train: np.ndarray,
        Z_test: np.ndarray,
        labels_train: np.ndarray,
        labels_test: np.ndarray,
    ) -> Dict:
        """
        Test classification performance across different cohorts.

        Args:
            Z_train: Training cohort factors (N_train x K)
            Z_test: Test cohort factors (N_test x K)
            labels_train: Training cohort labels (N_train,)
            labels_test: Test cohort labels (N_test,)

        Returns:
            Dict with keys for each classifier containing:
                - cross_cohort_performance: Metrics when training on train cohort,
                  testing on test cohort
                - within_cohort_performance: Metrics when training and testing on
                  test cohort (for comparison)
                - performance_drop: Difference in accuracy between cross-cohort
                  and within-cohort
        """
        results = {}

        # Train on training cohort, test on test cohort
        for model_name, model in self.classification_models.items():
            try:
                # Train on training cohort
                model.fit(Z_train, labels_train)

                # Test on test cohort
                y_pred = model.predict(Z_test)
                y_pred_proba = model.predict_proba(Z_test)

                # Calculate cross-cohort metrics
                cross_cohort_metrics = self.metrics.calculate_detailed_metrics(
                    labels_test, y_pred, y_pred_proba
                )

                # Also test within-cohort performance for comparison
                model_same_cohort = model.__class__(**model.get_params())
                model_same_cohort.fit(Z_test, labels_test)
                y_pred_same = model_same_cohort.predict(Z_test)
                y_pred_proba_same = model_same_cohort.predict_proba(Z_test)

                within_cohort_metrics = self.metrics.calculate_detailed_metrics(
                    labels_test, y_pred_same, y_pred_proba_same
                )

                results[model_name] = {
                    "cross_cohort_performance": cross_cohort_metrics,
                    "within_cohort_performance": within_cohort_metrics,
                    "performance_drop": (
                        cross_cohort_metrics["accuracy"]
                        - within_cohort_metrics["accuracy"]
                    ),
                }

            except Exception as e:
                self.logger.warning(
                    f"Cross-cohort classification failed for {model_name}: {str(e)}"
                )
                results[model_name] = {"error": str(e)}

        return results
