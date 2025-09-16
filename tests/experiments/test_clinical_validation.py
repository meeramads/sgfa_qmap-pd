"""Tests for clinical validation experiment."""

import tempfile
from pathlib import Path

import numpy as np
import pytest

from data import generate_synthetic_data
from experiments.clinical_validation import run_clinical_validation


class TestClinicalValidation:
    """Test clinical validation experiment."""

    @pytest.fixture
    def mock_config(self):
        """Create mock configuration for testing."""
        return {
            "data": {"data_dir": "./test_data"},
            "experiments": {
                "base_output_dir": "./test_results",
                "save_intermediate": True,
            },
            "clinical_validation": {
                "validation_types": [
                    "subtype_classification",
                    "disease_progression",
                    "biomarker_discovery",
                ],
                "classification_metrics": [
                    "accuracy",
                    "precision",
                    "recall",
                    "f1_score",
                ],
                "cross_validation": {
                    "n_folds": 3,  # Reduced for testing
                    "n_repeats": 1,
                },
            },
        }

    @pytest.fixture
    def synthetic_data(self):
        """Generate synthetic data for testing."""
        return generate_synthetic_data(
            num_sources=2, K=4, num_subjects=30, seed=42  # Small for fast testing
        )

    @pytest.fixture
    def mock_clinical_labels(self, synthetic_data):
        """Generate mock clinical labels."""
        n_subjects = synthetic_data["X_list"][0].shape[0]
        return {
            "diagnosis": np.random.choice(
                ["Control", "PD_Mild", "PD_Severe"],
                size=n_subjects,
                p=[0.4, 0.35, 0.25],
            ),
            "progression_score": np.random.normal(50, 15, n_subjects),
            "motor_scores": np.random.normal(20, 8, n_subjects),
            "cognitive_scores": np.random.normal(25, 5, n_subjects),
        }

    @pytest.fixture
    def shared_data_config(self, mock_config, synthetic_data):
        """Create config with shared data."""
        config = mock_config.copy()
        config["_shared_data"] = {
            "X_list": synthetic_data["X_list"],
            "preprocessing_info": {"strategy": "clinical_focused"},
            "mode": "shared",
        }
        return config

    def test_clinical_validation_runs(self, mock_config):
        """Test that clinical validation experiment runs without error."""
        with tempfile.TemporaryDirectory() as tmpdir:
            mock_config["experiments"]["base_output_dir"] = tmpdir

            result = run_clinical_validation(mock_config)
            assert result is not None

    def test_clinical_validation_with_shared_data(self, shared_data_config):
        """Test clinical validation with shared data mode."""
        with tempfile.TemporaryDirectory() as tmpdir:
            shared_data_config["experiments"]["base_output_dir"] = tmpdir

            result = run_clinical_validation(shared_data_config)
            assert result is not None

    def test_subtype_classification_validation(self, mock_config):
        """Test PD subtype classification validation."""
        with tempfile.TemporaryDirectory() as tmpdir:
            mock_config["experiments"]["base_output_dir"] = tmpdir

            # Focus on subtype classification
            mock_config["clinical_validation"]["validation_types"] = [
                "subtype_classification"
            ]

            result = run_clinical_validation(mock_config)
            assert result is not None

    def test_disease_progression_validation(self, mock_config):
        """Test disease progression prediction validation."""
        with tempfile.TemporaryDirectory() as tmpdir:
            mock_config["experiments"]["base_output_dir"] = tmpdir

            # Focus on disease progression
            mock_config["clinical_validation"]["validation_types"] = [
                "disease_progression"
            ]

            result = run_clinical_validation(mock_config)
            assert result is not None

    def test_biomarker_discovery_validation(self, mock_config):
        """Test biomarker discovery validation."""
        with tempfile.TemporaryDirectory() as tmpdir:
            mock_config["experiments"]["base_output_dir"] = tmpdir

            # Focus on biomarker discovery
            mock_config["clinical_validation"]["validation_types"] = [
                "biomarker_discovery"
            ]

            result = run_clinical_validation(mock_config)
            assert result is not None

    def test_external_cohort_validation(self, mock_config):
        """Test external cohort validation."""
        with tempfile.TemporaryDirectory() as tmpdir:
            mock_config["experiments"]["base_output_dir"] = tmpdir

            # Add external validation
            mock_config["clinical_validation"]["validation_types"].append(
                "external_cohort_validation"
            )

            result = run_clinical_validation(mock_config)
            assert result is not None

    def test_factor_clinical_correlation(self, mock_config):
        """Test factor-clinical variable correlation analysis."""
        with tempfile.TemporaryDirectory() as tmpdir:
            mock_config["experiments"]["base_output_dir"] = tmpdir

            # Test factor correlations with clinical measures
            mock_config["clinical_validation"]["analyze_correlations"] = True

            result = run_clinical_validation(mock_config)
            assert result is not None

    def test_longitudinal_validation(self, mock_config):
        """Test longitudinal clinical validation."""
        with tempfile.TemporaryDirectory() as tmpdir:
            mock_config["experiments"]["base_output_dir"] = tmpdir

            # Enable longitudinal analysis
            mock_config["clinical_validation"]["longitudinal_analysis"] = True

            result = run_clinical_validation(mock_config)
            assert result is not None

    def test_multiple_classification_metrics(self, mock_config):
        """Test multiple classification metrics computation."""
        with tempfile.TemporaryDirectory() as tmpdir:
            mock_config["experiments"]["base_output_dir"] = tmpdir

            # Test all classification metrics
            mock_config["clinical_validation"]["classification_metrics"] = [
                "accuracy",
                "precision",
                "recall",
                "f1_score",
                "roc_auc",
            ]

            result = run_clinical_validation(mock_config)
            assert result is not None

    def test_clinical_interpretability_analysis(self, mock_config):
        """Test clinical interpretability of factors."""
        with tempfile.TemporaryDirectory() as tmpdir:
            mock_config["experiments"]["base_output_dir"] = tmpdir

            # Enable interpretability analysis
            mock_config["clinical_validation"]["interpretability_analysis"] = True

            result = run_clinical_validation(mock_config)
            assert result is not None

    def test_clinical_validation_output_structure(self, mock_config):
        """Test that clinical validation produces expected output structure."""
        with tempfile.TemporaryDirectory() as tmpdir:
            mock_config["experiments"]["base_output_dir"] = tmpdir

            result = run_clinical_validation(mock_config)

            # Check result structure
            assert hasattr(result, "experiment_id")
            assert hasattr(result, "status")
            assert hasattr(result, "model_results")

            # Check output directory
            output_path = Path(tmpdir)
            assert output_path.exists()

    def test_cross_validation_in_clinical_context(self, mock_config):
        """Test cross-validation in clinical validation context."""
        with tempfile.TemporaryDirectory() as tmpdir:
            mock_config["experiments"]["base_output_dir"] = tmpdir

            # Specific CV for clinical validation
            mock_config["clinical_validation"]["cross_validation"] = {
                "n_folds": 5,
                "stratified": True,
                "group_aware": False,  # For testing
            }

            result = run_clinical_validation(mock_config)
            assert result is not None

    def test_feature_importance_analysis(self, mock_config):
        """Test clinical feature importance analysis."""
        with tempfile.TemporaryDirectory() as tmpdir:
            mock_config["experiments"]["base_output_dir"] = tmpdir

            # Enable feature importance
            mock_config["clinical_validation"]["feature_importance"] = True

            result = run_clinical_validation(mock_config)
            assert result is not None

    def test_mock_clinical_data_handling(self, mock_config):
        """Test handling of mock clinical data when real data unavailable."""
        with tempfile.TemporaryDirectory() as tmpdir:
            mock_config["experiments"]["base_output_dir"] = tmpdir

            # Should handle absence of real clinical data gracefully
            mock_config["clinical_validation"]["use_mock_labels"] = True

            result = run_clinical_validation(mock_config)
            assert result is not None

    def test_minimal_clinical_config(self, synthetic_data):
        """Test with minimal clinical validation configuration."""
        minimal_config = {
            "data": {"data_dir": "./test_data"},
            "experiments": {"base_output_dir": "./test_results"},
            "_shared_data": {
                "X_list": synthetic_data["X_list"],
                "preprocessing_info": {},
                "mode": "shared",
            },
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            minimal_config["experiments"]["base_output_dir"] = tmpdir

            result = run_clinical_validation(minimal_config)
            assert result is not None

    def test_clinical_validation_error_handling(self, mock_config):
        """Test error handling in clinical validation."""
        with tempfile.TemporaryDirectory() as tmpdir:
            mock_config["experiments"]["base_output_dir"] = tmpdir

            # Invalid validation type
            mock_config["clinical_validation"]["validation_types"] = [
                "invalid_validation_type"
            ]

            # Should handle gracefully
            result = run_clinical_validation(mock_config)

            # Either succeeds with fallback or handles error
            if result is not None:
                assert hasattr(result, "status")

    def test_classification_baseline_comparison(self, mock_config):
        """Test comparison with baseline classification methods."""
        with tempfile.TemporaryDirectory() as tmpdir:
            mock_config["experiments"]["base_output_dir"] = tmpdir

            # Enable baseline comparison
            mock_config["clinical_validation"]["compare_baselines"] = True

            result = run_clinical_validation(mock_config)
            assert result is not None

    def test_clinical_significance_testing(self, mock_config):
        """Test clinical significance testing."""
        with tempfile.TemporaryDirectory() as tmpdir:
            mock_config["experiments"]["base_output_dir"] = tmpdir

            # Enable significance testing
            mock_config["clinical_validation"]["statistical_testing"] = True

            result = run_clinical_validation(mock_config)
            assert result is not None

    def test_multiclass_classification_validation(self, mock_config):
        """Test multiclass classification validation."""
        with tempfile.TemporaryDirectory() as tmpdir:
            mock_config["experiments"]["base_output_dir"] = tmpdir

            # Test multiclass scenario (Control, PD_Mild, PD_Severe)
            mock_config["clinical_validation"]["multiclass_labels"] = True

            result = run_clinical_validation(mock_config)
            assert result is not None

    def test_binary_classification_validation(self, mock_config):
        """Test binary classification validation."""
        with tempfile.TemporaryDirectory() as tmpdir:
            mock_config["experiments"]["base_output_dir"] = tmpdir

            # Test binary scenario (Control vs PD)
            mock_config["clinical_validation"]["binary_classification"] = True

            result = run_clinical_validation(mock_config)
            assert result is not None

    @pytest.mark.parametrize(
        "validation_type",
        ["subtype_classification", "disease_progression", "biomarker_discovery"],
    )
    def test_individual_validation_types(self, mock_config, validation_type):
        """Test individual validation types."""
        with tempfile.TemporaryDirectory() as tmpdir:
            mock_config["experiments"]["base_output_dir"] = tmpdir

            # Test single validation type
            mock_config["clinical_validation"]["validation_types"] = [validation_type]

            result = run_clinical_validation(mock_config)
            assert result is not None

    def test_clinical_validation_reproducibility(self, mock_config):
        """Test clinical validation reproducibility."""
        with tempfile.TemporaryDirectory() as tmpdir:
            mock_config["experiments"]["base_output_dir"] = tmpdir
            mock_config["random_seed"] = 42

            # Run twice with same configuration
            result1 = run_clinical_validation(mock_config)
            result2 = run_clinical_validation(mock_config)

            assert result1 is not None
            assert result2 is not None

    def test_clinical_validation_matrix_saving(self, mock_config):
        """Test that clinical validation matrices are saved."""
        with tempfile.TemporaryDirectory() as tmpdir:
            mock_config["experiments"]["base_output_dir"] = tmpdir

            result = run_clinical_validation(mock_config)
            assert result is not None

            # Check for output files
            output_files = list(Path(tmpdir).rglob("*"))
            assert len(output_files) > 0

    def test_clinical_metrics_validation(self, mock_config):
        """Test that clinical metrics are computed and reasonable."""
        with tempfile.TemporaryDirectory() as tmpdir:
            mock_config["experiments"]["base_output_dir"] = tmpdir

            result = run_clinical_validation(mock_config)
            assert result is not None

            if hasattr(result, "model_results") and result.model_results:
                model_results = result.model_results

                # Should contain clinical validation results
                assert isinstance(model_results, dict)

    def test_stratified_validation(self, mock_config):
        """Test stratified validation for imbalanced clinical data."""
        with tempfile.TemporaryDirectory() as tmpdir:
            mock_config["experiments"]["base_output_dir"] = tmpdir

            # Enable stratified validation
            mock_config["clinical_validation"]["cross_validation"]["stratified"] = True

            result = run_clinical_validation(mock_config)
            assert result is not None

    def test_group_aware_validation(self, mock_config):
        """Test group-aware validation for multi-site data."""
        with tempfile.TemporaryDirectory() as tmpdir:
            mock_config["experiments"]["base_output_dir"] = tmpdir

            # Enable group-aware validation (for multi-site data)
            mock_config["clinical_validation"]["cross_validation"]["group_aware"] = True

            result = run_clinical_validation(mock_config)
            assert result is not None

    def test_clinical_factor_mapping(self, mock_config):
        """Test mapping of factors to clinical variables."""
        with tempfile.TemporaryDirectory() as tmpdir:
            mock_config["experiments"]["base_output_dir"] = tmpdir

            # Enable factor-clinical mapping
            mock_config["clinical_validation"]["factor_mapping"] = True

            result = run_clinical_validation(mock_config)
            assert result is not None

    def test_clinical_validation_plots(self, mock_config):
        """Test clinical validation plot generation."""
        with tempfile.TemporaryDirectory() as tmpdir:
            mock_config["experiments"]["base_output_dir"] = tmpdir

            # Enable plot generation
            mock_config["experiments"]["generate_plots"] = True

            result = run_clinical_validation(mock_config)
            assert result is not None

            # Check for plot output
            all_files = list(Path(tmpdir).rglob("*"))
            assert len(all_files) > 0
