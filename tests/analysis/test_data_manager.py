"""Tests for DataManager."""

import pytest
from unittest.mock import Mock, patch, MagicMock
import numpy as np

from analysis.data_manager import DataManager


@pytest.mark.unit
class TestDataManager:
    """Test DataManager functionality."""

    def test_data_manager_init(self, sample_config):
        """Test DataManager initialization."""
        dm = DataManager(sample_config)
        
        assert dm.config == sample_config
        assert dm.preprocessor is None

    def test_load_data_synthetic(self, sample_config, sample_synthetic_data):
        """Test loading synthetic data."""
        sample_config.dataset = 'synthetic'
        
        with patch('analysis.data_manager.generate_synthetic_data') as mock_generate:
            mock_generate.return_value = sample_synthetic_data
            
            dm = DataManager(sample_config)
            result = dm.load_data()
            
            mock_generate.assert_called_once_with(
                sample_config.num_sources,
                sample_config.K,
                sample_config.percW
            )
            assert result == sample_synthetic_data

    def test_load_data_qmap_pd(self, sample_config):
        """Test loading qMAP-PD data."""
        sample_config.dataset = 'qmap_pd'
        sample_config.data_dir = '/fake/data'
        sample_config.clinical_rel = 'clinical.csv'
        sample_config.volumes_rel = 'volumes.csv'
        sample_config.roi_views = True
        
        mock_data = {'X_list': [], 'dataset': 'qmap_pd'}
        
        with patch('analysis.data_manager.load_qmap_pd') as mock_load:
            mock_load.return_value = mock_data
            
            dm = DataManager(sample_config)
            result = dm.load_data()
            
            mock_load.assert_called_once_with(
                data_dir='/fake/data',
                clinical_rel='clinical.csv', 
                volumes_rel='volumes.csv',
                imaging_as_single_view=False,  # not roi_views
                enable_advanced_preprocessing=sample_config.enable_preprocessing,
                enable_spatial_processing=sample_config.enable_spatial_processing,
                **sample_config.preprocessing_params
            )
            assert result == mock_data

    def test_load_data_unknown_dataset(self, sample_config):
        """Test error for unknown dataset."""
        sample_config.dataset = 'unknown'
        
        dm = DataManager(sample_config)
        
        with pytest.raises(ValueError, match="Unknown dataset: unknown"):
            dm.load_data()

    def test_prepare_for_analysis(self, sample_config, sample_synthetic_data):
        """Test data preparation for analysis."""
        dm = DataManager(sample_config)
        
        X_list, hypers = dm.prepare_for_analysis(sample_synthetic_data)
        
        # Check X_list
        assert X_list == sample_synthetic_data['X_list']
        
        # Check hyperparameters
        expected_hypers = {
            'a_sigma': 1, 'b_sigma': 1,
            'nu_local': 1, 'nu_global': 1, 
            'slab_scale': 2, 'slab_df': 4,
            'percW': sample_config.percW,
            'Dm': [20, 15, 10]  # From sample data
        }
        assert hypers == expected_hypers

    def test_log_preprocessing_results(self, sample_config, caplog):
        """Test logging of preprocessing results."""
        dm = DataManager(sample_config)
        
        preprocessing = {
            'feature_reduction': {
                'view_1': {
                    'original': 100,
                    'processed': 50,  
                    'reduction_ratio': 0.5
                },
                'view_2': {
                    'original': 80,
                    'processed': 60,
                    'reduction_ratio': 0.75
                }
            }
        }
        
        dm._log_preprocessing_results(preprocessing)
        
        # Check log messages
        assert "=== Preprocessing Applied ===" in caplog.text
        assert "view_1: 100 -> 50 features (50.00% retained)" in caplog.text
        assert "view_2: 80 -> 60 features (75.00% retained)" in caplog.text


@pytest.mark.unit  
class TestDataManagerIntegration:
    """Integration tests for DataManager."""
    
    @patch('analysis.data_manager.logger')
    def test_load_synthetic_with_logging(self, mock_logger, sample_config):
        """Test synthetic data loading logs appropriately."""
        sample_config.dataset = 'synthetic'
        
        with patch('analysis.data_manager.generate_synthetic_data') as mock_generate:
            mock_generate.return_value = {'X_list': [], 'dataset': 'synthetic'}
            
            dm = DataManager(sample_config)
            dm.load_data()
            
            # Should have logged something about synthetic data
            assert mock_logger.info.called

    def test_qmap_preprocessing_integration(self, sample_config):
        """Test qMAP-PD data loading with preprocessing parameters."""
        sample_config.dataset = 'qmap_pd'
        sample_config.data_dir = '/fake/data'
        sample_config.enable_preprocessing = True
        sample_config.enable_spatial_processing = True
        sample_config.preprocessing_params = {
            'imputation_strategy': 'median',
            'feature_selection': 'variance'
        }
        
        mock_data = {
            'X_list': [],
            'preprocessing': {
                'feature_reduction': {
                    'imaging': {'original': 100, 'processed': 80, 'reduction_ratio': 0.8}
                }
            }
        }
        
        with patch('analysis.data_manager.load_qmap_pd') as mock_load:
            mock_load.return_value = mock_data
            
            dm = DataManager(sample_config)
            result = dm.load_data()
            
            # Check that preprocessing params were passed
            call_kwargs = mock_load.call_args[1]
            assert call_kwargs['enable_advanced_preprocessing'] == True
            assert call_kwargs['enable_spatial_processing'] == True
            assert call_kwargs['imputation_strategy'] == 'median'
            assert call_kwargs['feature_selection'] == 'variance'