"""Tests for analysis.component_factory module."""

from unittest.mock import Mock, patch, MagicMock

import numpy as np
import pytest

from analysis.component_factory import (
    AnalysisMode,
    AnalysisComponents,
    create_analysis_components,
    integrate_analysis_with_pipeline,
    run_sgfa_with_components,
    prepare_experiment_data,
    quick_sgfa_run,
)


class TestAnalysisMode:
    """Test AnalysisMode enum."""

    def test_mode_values(self):
        """Test that all expected modes exist."""
        assert hasattr(AnalysisMode, 'STRUCTURED')
        assert hasattr(AnalysisMode, 'FALLBACK')
        assert hasattr(AnalysisMode, 'BASIC')

    def test_mode_string_values(self):
        """Test enum string values."""
        assert AnalysisMode.STRUCTURED.value == "structured"
        assert AnalysisMode.FALLBACK.value == "fallback"
        assert AnalysisMode.BASIC.value == "basic"


class TestAnalysisComponents:
    """Test AnalysisComponents dataclass."""

    def test_initialization(self):
        """Test basic initialization."""
        components = AnalysisComponents(
            mode=AnalysisMode.STRUCTURED,
            data_manager=Mock(),
            model_runner=Mock(),
        )

        assert components.mode == AnalysisMode.STRUCTURED
        assert components.data_manager is not None
        assert components.model_runner is not None
        assert components.metadata == {}  # Auto-initialized

    def test_is_structured_property(self):
        """Test is_structured property."""
        structured = AnalysisComponents(mode=AnalysisMode.STRUCTURED)
        fallback = AnalysisComponents(mode=AnalysisMode.FALLBACK)

        assert structured.is_structured is True
        assert fallback.is_structured is False

    def test_success_property(self):
        """Test success property."""
        successful = AnalysisComponents(
            mode=AnalysisMode.STRUCTURED,
            error=None
        )
        failed = AnalysisComponents(
            mode=AnalysisMode.FALLBACK,
            error="Some error"
        )

        assert successful.success is True
        assert failed.success is False

    def test_load_and_prepare_data_no_manager(self):
        """Test load_and_prepare_data when DataManager not available."""
        components = AnalysisComponents(
            mode=AnalysisMode.STRUCTURED,
            data_manager=None
        )

        X_list, info = components.load_and_prepare_data()

        assert X_list == []
        assert info["data_loaded"] is False
        assert "error" in info

    @patch('analysis.component_factory.logger')
    def test_load_and_prepare_data_success(self, mock_logger):
        """Test successful data loading and preparation."""
        # Mock DataManager
        mock_dm = Mock()
        mock_dm.load_data.return_value = {"raw": "data"}
        X_list_mock = [np.array([[1, 2], [3, 4]])]
        hypers_mock = {"K": 5}
        mock_dm.prepare_for_analysis.return_value = (X_list_mock, hypers_mock)

        components = AnalysisComponents(
            mode=AnalysisMode.STRUCTURED,
            data_manager=mock_dm
        )

        X_list, info = components.load_and_prepare_data()

        assert len(X_list) == 1
        assert info["data_loaded"] is True
        assert info["loader"] == "DataManager"
        assert "hyperparameters" in info
        assert "data_characteristics" in info

    @patch('analysis.component_factory.logger')
    def test_load_and_prepare_data_error(self, mock_logger):
        """Test data loading error handling."""
        mock_dm = Mock()
        mock_dm.load_data.side_effect = ValueError("Load failed")

        components = AnalysisComponents(
            mode=AnalysisMode.STRUCTURED,
            data_manager=mock_dm
        )

        X_list, info = components.load_and_prepare_data()

        assert X_list == []
        assert info["status"] == "failed"
        assert "error" in info

    def test_run_analysis_no_components(self):
        """Test run_analysis when components not available."""
        components = AnalysisComponents(
            mode=AnalysisMode.STRUCTURED,
            data_manager=None,
            model_runner=None
        )

        result = components.run_analysis([np.array([[1, 2]])])

        assert result["status"] == "failed"
        assert "error" in result

    @patch('analysis.component_factory.logger')
    def test_run_analysis_success(self, mock_logger):
        """Test successful analysis execution."""
        # Mock components
        mock_dm = Mock()
        mock_dm.prepare_for_analysis.return_value = (
            [np.array([[1, 2]])],
            {"K": 5}
        )

        mock_mr = Mock()
        mock_mr.run_standard_analysis.return_value = {"run_1": "results"}

        components = AnalysisComponents(
            mode=AnalysisMode.STRUCTURED,
            data_manager=mock_dm,
            model_runner=mock_mr
        )

        X_list = [np.array([[1, 2]])]
        result = components.run_analysis(X_list)

        assert result["status"] == "completed"
        assert result["analysis_type"] == "structured_mcmc"
        assert result["structured_framework"] is True


@patch('analysis.component_factory.DataManager')
@patch('analysis.component_factory.ModelRunner')
class TestCreateAnalysisComponents:
    """Test create_analysis_components function."""

    def test_basic_creation(self, mock_mr_class, mock_dm_class):
        """Test basic component creation."""
        config_dict = {
            "dataset": "qmap_pd",
            "K": 10,
            "model": "sparseGFA"
        }

        dm, mr = create_analysis_components(config_dict)

        # Verify DataManager and ModelRunner were instantiated
        mock_dm_class.assert_called_once()
        mock_mr_class.assert_called_once()

    def test_applies_defaults(self, mock_mr_class, mock_dm_class):
        """Test that defaults are applied."""
        config_dict = {}  # Empty config

        dm, mr = create_analysis_components(config_dict)

        # Should have been called (defaults applied)
        mock_dm_class.assert_called_once()
        mock_mr_class.assert_called_once()

    def test_with_results_dir(self, mock_mr_class, mock_dm_class):
        """Test creation with results directory."""
        config_dict = {"K": 10}
        results_dir = "/path/to/results"

        dm, mr = create_analysis_components(config_dict, results_dir)

        # ModelRunner should have been passed results_dir
        assert mock_mr_class.call_args[0][1] == results_dir


@patch('analysis.component_factory.ConfigManager')
@patch('analysis.component_factory.DataManager')
@patch('analysis.component_factory.ModelRunner')
@patch('analysis.component_factory.logger')
class TestIntegrateAnalysisWithPipeline:
    """Test integrate_analysis_with_pipeline function."""

    def test_successful_integration(self, mock_logger, mock_mr_class,
                                    mock_dm_class, mock_cm_class):
        """Test successful pipeline integration."""
        # Mock ConfigManager
        mock_cm = Mock()
        mock_cm.args = Mock()
        mock_cm.setup_analysis_config.return_value = Mock(
            run_standard=True,
            run_cv=False
        )
        mock_cm_class.return_value = mock_cm

        config = {"data": {"data_dir": "/path"}}
        data_dir = "/path/to/data"

        dm, mr, metadata = integrate_analysis_with_pipeline(
            config, data_dir
        )

        assert dm is not None
        assert mr is not None
        assert metadata["analysis_integration_enabled"] is True
        assert metadata["framework_available"] is True

    def test_integration_failure(self, mock_logger, mock_mr_class,
                                mock_dm_class, mock_cm_class):
        """Test integration failure handling."""
        # Make ConfigManager raise an error
        mock_cm_class.side_effect = ValueError("Config error")

        config = {}
        data_dir = "/path"

        dm, mr, metadata = integrate_analysis_with_pipeline(
            config, data_dir
        )

        assert dm is None
        assert mr is None
        assert metadata["analysis_integration_enabled"] is False
        assert metadata["framework_available"] is False
        assert "error" in metadata


@patch('analysis.component_factory.create_analysis_components')
class TestRunSGFAWithComponents:
    """Test run_sgfa_with_components function."""

    @patch('analysis.component_factory.logger')
    def test_successful_run(self, mock_logger, mock_create_components):
        """Test successful SGFA run."""
        # Mock components
        mock_dm = Mock()
        mock_dm.prepare_for_analysis.return_value = (
            [np.array([[1, 2]])],
            {"K": 5}
        )

        mock_mr = Mock()
        mock_mr.run_standard_analysis.return_value = {"run_1": "results"}

        mock_create_components.return_value = (mock_dm, mock_mr)

        X_list = [np.array([[1, 2]])]
        hypers = {"K": 5}
        args_dict = {"model": "sparseGFA"}

        result = run_sgfa_with_components(X_list, hypers, args_dict)

        # Should have returned results
        assert "run_1" in result

    @patch('analysis.component_factory.logger')
    @patch('analysis.component_factory._run_sgfa_fallback')
    def test_fallback_on_error(self, mock_fallback, mock_logger,
                               mock_create_components):
        """Test fallback when main execution fails."""
        # Mock components to raise error
        mock_dm = Mock()
        mock_dm.prepare_for_analysis.side_effect = ValueError("Error")

        mock_mr = Mock()
        mock_create_components.return_value = (mock_dm, mock_mr)

        mock_fallback.return_value = {"fallback": "result"}

        X_list = [np.array([[1, 2]])]
        hypers = {"K": 5}
        args_dict = {}

        result = run_sgfa_with_components(X_list, hypers, args_dict)

        # Should have called fallback
        mock_fallback.assert_called_once()


@patch('analysis.component_factory.create_analysis_components')
class TestPrepareExperimentData:
    """Test prepare_experiment_data function."""

    def test_with_existing_data(self, mock_create_components):
        """Test preparation with existing X_list."""
        mock_dm = Mock()
        mock_dm.prepare_for_analysis.return_value = (
            [np.array([[1, 2]])],
            {"K": 5}
        )
        mock_create_components.return_value = (mock_dm, Mock())

        config_dict = {"K": 10}
        X_list_input = [np.array([[1, 2]])]

        X_list, hypers = prepare_experiment_data(config_dict, X_list_input)

        assert len(X_list) == 1
        assert "K" in hypers

    def test_without_existing_data(self, mock_create_components):
        """Test preparation without X_list (load data)."""
        mock_dm = Mock()
        mock_dm.load_data.return_value = {"raw": "data"}
        mock_dm.prepare_for_analysis.return_value = (
            [np.array([[1, 2]])],
            {"K": 5}
        )
        mock_create_components.return_value = (mock_dm, Mock())

        config_dict = {"K": 10}

        X_list, hypers = prepare_experiment_data(config_dict, X_list=None)

        # Should have called load_data
        mock_dm.load_data.assert_called_once()
        assert len(X_list) == 1


@patch('analysis.component_factory.run_sgfa_with_components')
class TestQuickSGFARun:
    """Test quick_sgfa_run convenience function."""

    def test_basic_run(self, mock_run_sgfa):
        """Test basic quick run."""
        mock_run_sgfa.return_value = {"result": "success"}

        X_list = [np.array([[1, 2]])]
        result = quick_sgfa_run(X_list, K=5)

        # Should have been called with defaults
        mock_run_sgfa.assert_called_once()
        call_args = mock_run_sgfa.call_args

        # Check X_list passed
        assert np.array_equal(call_args[0][0][0], X_list[0])

        # Check hypers contain defaults
        hypers = call_args[0][1]
        assert "percW" in hypers
        assert "Dm" in hypers

    def test_custom_parameters(self, mock_run_sgfa):
        """Test quick run with custom parameters."""
        mock_run_sgfa.return_value = {"result": "success"}

        X_list = [np.array([[1, 2]])]
        result = quick_sgfa_run(
            X_list,
            K=10,
            percW=30.0,
            model="customGFA",
            custom_param="value"
        )

        call_args = mock_run_sgfa.call_args
        args_dict = call_args[0][2]

        assert args_dict["K"] == 10
        assert args_dict["model"] == "customGFA"
        assert args_dict["custom_param"] == "value"


class TestIntegration:
    """Integration tests for component factory."""

    @patch('analysis.component_factory.DataManager')
    @patch('analysis.component_factory.ModelRunner')
    @patch('analysis.component_factory.logger')
    def test_complete_workflow(self, mock_logger, mock_mr_class, mock_dm_class):
        """Test complete analysis workflow."""
        # Setup mocks
        mock_dm = Mock()
        mock_dm.load_data.return_value = {"data": "loaded"}
        mock_dm.prepare_for_analysis.return_value = (
            [np.array([[1, 2], [3, 4]])],
            {"K": 5, "Dm": [2]}
        )
        mock_dm_class.return_value = mock_dm

        mock_mr = Mock()
        mock_mr.run_standard_analysis.return_value = {"run_1": "results"}
        mock_mr_class.return_value = mock_mr

        # Create components
        dm, mr = create_analysis_components({"K": 5})

        # Create AnalysisComponents wrapper
        components = AnalysisComponents(
            mode=AnalysisMode.STRUCTURED,
            data_manager=dm,
            model_runner=mr
        )

        # Load data
        X_list, load_info = components.load_and_prepare_data()
        assert load_info["data_loaded"] is True

        # Run analysis
        result = components.run_analysis(X_list)
        assert result["status"] == "completed"
