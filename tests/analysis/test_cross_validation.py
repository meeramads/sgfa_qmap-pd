"""Tests for cross-validation module."""

import pytest
from unittest.mock import Mock, patch, MagicMock
import numpy as np

from analysis.cross_validation import CVRunner, should_run_cv_analysis, should_run_standard_analysis


@pytest.mark.unit
@pytest.mark.cv
class TestCVRunner:
    """Test CVRunner functionality."""

    def test_cvrunner_init(self, sample_config, temp_dir):
        """Test CVRunner initialization."""
        cv_runner = CVRunner(sample_config, temp_dir)
        
        assert cv_runner.config == sample_config
        assert cv_runner.results_dir == temp_dir

    def test_check_cv_availability_success(self, sample_config, temp_dir):
        """Test CV availability check when modules are available."""
        cv_runner = CVRunner(sample_config, temp_dir)
        
        with patch('analysis.cross_validation.SparseBayesianGFACrossValidator'), \
             patch('analysis.cross_validation.CVConfig'):
            
            result = cv_runner._check_cv_availability()
            assert result is True

    def test_check_cv_availability_failure(self, sample_config, temp_dir):
        """Test CV availability check when modules are missing."""
        cv_runner = CVRunner(sample_config, temp_dir)
        
        with patch('analysis.cross_validation.SparseBayesianGFACrossValidator', side_effect=ImportError):
            result = cv_runner._check_cv_availability()
            assert result is False

    def test_check_neuroimaging_cv_availability_success(self, sample_config, temp_dir):
        """Test neuroimaging CV availability check when available."""
        cv_runner = CVRunner(sample_config, temp_dir)
        
        with patch('analysis.cross_validation.NeuroImagingCrossValidator'), \
             patch('analysis.cross_validation.NeuroImagingCVConfig'), \
             patch('analysis.cross_validation.ParkinsonsConfig'):
            
            result = cv_runner._check_neuroimaging_cv_availability()
            assert result is True

    def test_check_neuroimaging_cv_availability_failure(self, sample_config, temp_dir):
        """Test neuroimaging CV availability check when missing.""" 
        cv_runner = CVRunner(sample_config, temp_dir)
        
        with patch('analysis.cross_validation.NeuroImagingCrossValidator', side_effect=ImportError):
            result = cv_runner._check_neuroimaging_cv_availability()
            assert result is False

    @patch('analysis.cross_validation.logger')
    def test_run_cv_analysis_no_modules(self, mock_logger, sample_config, temp_dir, sample_synthetic_data, sample_hyperparameters):
        """Test CV analysis when no CV modules are available."""
        cv_runner = CVRunner(sample_config, temp_dir)
        
        with patch.object(cv_runner, '_check_cv_availability', return_value=False), \
             patch.object(cv_runner, '_check_neuroimaging_cv_availability', return_value=False):
            
            result = cv_runner.run_cv_analysis(
                sample_synthetic_data['X_list'], 
                sample_hyperparameters, 
                sample_synthetic_data
            )
            
            assert result is None
            mock_logger.error.assert_called_with("Cross-validation requested but no CV module available!")

    def test_run_cv_analysis_neuroimaging_cv(self, sample_config, temp_dir, sample_synthetic_data, sample_hyperparameters):
        """Test neuroimaging CV analysis."""
        sample_config.neuroimaging_cv = True
        cv_runner = CVRunner(sample_config, temp_dir)
        
        mock_results = {'cv_scores': [0.8, 0.9, 0.85]}
        mock_cv_object = Mock()
        
        with patch.object(cv_runner, '_check_neuroimaging_cv_availability', return_value=True), \
             patch.object(cv_runner, '_run_neuroimaging_cv_analysis', return_value=(mock_results, mock_cv_object)):
            
            result = cv_runner.run_cv_analysis(
                sample_synthetic_data['X_list'],
                sample_hyperparameters, 
                sample_synthetic_data
            )
            
            assert result == (mock_results, mock_cv_object)

    def test_run_cv_analysis_basic_cv(self, sample_config, temp_dir, sample_synthetic_data, sample_hyperparameters):
        """Test basic CV analysis."""
        cv_runner = CVRunner(sample_config, temp_dir)
        
        mock_results = {'cv_scores': [0.7, 0.8, 0.75]}
        mock_cv_object = Mock()
        
        with patch.object(cv_runner, '_check_cv_availability', return_value=True), \
             patch.object(cv_runner, '_check_neuroimaging_cv_availability', return_value=False), \
             patch.object(cv_runner, '_run_basic_cv_analysis', return_value=(mock_results, mock_cv_object)):
            
            result = cv_runner.run_cv_analysis(
                sample_synthetic_data['X_list'],
                sample_hyperparameters,
                sample_synthetic_data
            )
            
            assert result == (mock_results, mock_cv_object)

    def test_run_neuroimaging_cv_analysis_standard(self, sample_config, temp_dir, sample_synthetic_data, sample_hyperparameters):
        """Test standard neuroimaging CV analysis."""
        sample_config.nested_cv = False
        sample_config.cv_folds = 3
        sample_config.seed = 123
        
        cv_runner = CVRunner(sample_config, temp_dir)
        
        mock_cv = Mock()
        mock_results = {'scores': [0.8, 0.9]}
        mock_cv.neuroimaging_cross_validate.return_value = mock_results
        
        with patch('analysis.cross_validation.NeuroImagingCrossValidator', return_value=mock_cv), \
             patch('analysis.cross_validation.NeuroImagingCVConfig') as mock_config_cls, \
             patch('analysis.cross_validation.ParkinsonsConfig') as mock_pd_config_cls:
            
            mock_config = Mock()
            mock_config_cls.return_value = mock_config
            mock_pd_config = Mock() 
            mock_pd_config_cls.return_value = mock_pd_config
            
            result = cv_runner._run_neuroimaging_cv_analysis(
                sample_synthetic_data['X_list'],
                sample_hyperparameters,
                sample_synthetic_data
            )
            
            # Check configuration was set up properly
            assert mock_config.outer_cv_folds == 3
            assert mock_config.random_state == 123
            
            # Check CV was called with correct parameters
            mock_cv.neuroimaging_cross_validate.assert_called_once_with(
                sample_synthetic_data['X_list'],
                sample_config,
                sample_hyperparameters, 
                sample_synthetic_data
            )
            
            assert result == (mock_results, mock_cv)

    def test_run_neuroimaging_cv_analysis_nested(self, sample_config, temp_dir, sample_synthetic_data, sample_hyperparameters):
        """Test nested neuroimaging CV analysis."""
        sample_config.nested_cv = True
        cv_runner = CVRunner(sample_config, temp_dir)
        
        mock_cv = Mock()
        mock_results = {'nested_scores': [0.85]}
        mock_cv.nested_neuroimaging_cv.return_value = mock_results
        
        with patch('analysis.cross_validation.NeuroImagingCrossValidator', return_value=mock_cv), \
             patch('analysis.cross_validation.NeuroImagingCVConfig'), \
             patch('analysis.cross_validation.ParkinsonsConfig'):
            
            result = cv_runner._run_neuroimaging_cv_analysis(
                sample_synthetic_data['X_list'],
                sample_hyperparameters,
                sample_synthetic_data
            )
            
            mock_cv.nested_neuroimaging_cv.assert_called_once()
            assert result == (mock_results, mock_cv)

    def test_run_basic_cv_analysis(self, sample_config, temp_dir, sample_synthetic_data, sample_hyperparameters):
        """Test basic CV analysis."""
        sample_config.cv_folds = 4
        sample_config.cv_n_jobs = 2
        sample_config.seed = 456
        
        cv_runner = CVRunner(sample_config, temp_dir)
        
        mock_cv = Mock()
        mock_results = {'basic_scores': [0.7, 0.8]}
        mock_cv.standard_cross_validate.return_value = mock_results
        
        with patch('analysis.cross_validation.SparseBayesianGFACrossValidator', return_value=mock_cv), \
             patch('analysis.cross_validation.CVConfig') as mock_config_cls:
            
            mock_config = Mock()
            mock_config_cls.return_value = mock_config
            
            result = cv_runner._run_basic_cv_analysis(
                sample_synthetic_data['X_list'],
                sample_hyperparameters,
                sample_synthetic_data
            )
            
            # Check configuration
            assert mock_config.outer_cv_folds == 4
            assert mock_config.n_jobs == 2
            assert mock_config.random_state == 456
            
            # Check CV was called
            mock_cv.standard_cross_validate.assert_called_once_with(
                sample_synthetic_data['X_list'],
                sample_config,
                sample_hyperparameters
            )
            
            assert result == (mock_results, mock_cv)


@pytest.mark.unit
class TestCVHelperFunctions:
    """Test CV helper functions."""

    def test_should_run_standard_analysis_default(self, sample_config):
        """Test should_run_standard_analysis with default config.""" 
        result = should_run_standard_analysis(sample_config)
        assert result is True  # cv_only is False by default

    def test_should_run_standard_analysis_cv_only(self, sample_config):
        """Test should_run_standard_analysis when cv_only is True."""
        sample_config.cv_only = True
        result = should_run_standard_analysis(sample_config)
        assert result is False

    def test_should_run_cv_analysis_default(self, sample_config):
        """Test should_run_cv_analysis with default config."""
        result = should_run_cv_analysis(sample_config)
        assert result is False  # No CV flags set by default

    def test_should_run_cv_analysis_run_cv(self, sample_config):
        """Test should_run_cv_analysis when run_cv is True."""
        sample_config.run_cv = True
        result = should_run_cv_analysis(sample_config)
        assert result is True

    def test_should_run_cv_analysis_cv_only(self, sample_config):
        """Test should_run_cv_analysis when cv_only is True."""
        sample_config.cv_only = True 
        result = should_run_cv_analysis(sample_config)
        assert result is True

    def test_should_run_cv_analysis_neuroimaging_cv(self, sample_config):
        """Test should_run_cv_analysis when neuroimaging_cv is True."""
        sample_config.neuroimaging_cv = True
        result = should_run_cv_analysis(sample_config)
        assert result is True

    def test_should_run_cv_analysis_multiple_flags(self, sample_config):
        """Test should_run_cv_analysis with multiple CV flags."""
        sample_config.run_cv = True
        sample_config.neuroimaging_cv = True
        result = should_run_cv_analysis(sample_config)
        assert result is True