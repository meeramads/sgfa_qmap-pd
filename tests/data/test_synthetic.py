"""Tests for synthetic data generation."""

import pytest
import numpy as np
from unittest.mock import patch

from data.synthetic import generate_synthetic_data


@pytest.mark.unit
@pytest.mark.data
class TestSyntheticDataGeneration:
    """Test synthetic data generation functionality."""

    def test_generate_synthetic_data_default_params(self):
        """Test synthetic data generation with default parameters."""
        data = generate_synthetic_data()
        
        # Check structure
        assert isinstance(data, dict)
        required_keys = ['X_list', 'view_names', 'feature_names', 'subject_ids', 
                        'clinical', 'scalers', 'meta', 'ground_truth']
        for key in required_keys:
            assert key in data, f"Missing key: {key}"
        
        # Check dimensions
        assert len(data['X_list']) == 3  # Default num_sources
        assert data['X_list'][0].shape[0] == 150  # Default N subjects
        assert sum(x.shape[1] for x in data['X_list']) == 120  # Total features (60+40+20)
        
        # Check meta information
        assert data['meta']['dataset'] == 'synthetic'
        assert data['meta']['N'] == 150
        assert data['meta']['K_true'] == 3
        
    def test_generate_synthetic_data_custom_params(self):
        """Test synthetic data generation with custom parameters."""
        data = generate_synthetic_data(num_sources=2, K=4, percW=50.0)
        
        # Check custom parameters applied
        assert len(data['X_list']) == 2
        assert len(data['view_names']) == 2
        assert data['ground_truth']['K_true'] == 4
        
        # Check feature dimensions adjusted for 2 sources
        total_features = sum(x.shape[1] for x in data['X_list'])
        assert total_features == 120  # Should distribute 120 features across 2 sources
        
    def test_synthetic_data_ground_truth_structure(self):
        """Test ground truth data structure."""
        data = generate_synthetic_data(K=5)
        
        gt = data['ground_truth']
        N = data['meta']['N']
        K = gt['K_true']
        total_features = sum(data['meta']['Dm'])
        
        # Check ground truth dimensions
        assert gt['Z'].shape == (N, K)
        assert gt['W'].shape == (total_features, K)
        assert len(gt['sigma']) == len(data['X_list'])
        assert gt['lmbZ'].shape == (N, K)
        assert gt['lmbW'].shape == (total_features, K)
        
    def test_synthetic_data_reproducibility(self):
        """Test that synthetic data generation is reproducible."""
        np.random.seed(42)
        data1 = generate_synthetic_data()
        
        np.random.seed(42)
        data2 = generate_synthetic_data()
        
        # Check that results are identical
        np.testing.assert_array_equal(data1['X_list'][0], data2['X_list'][0])
        np.testing.assert_array_equal(data1['ground_truth']['Z'], data2['ground_truth']['Z'])
        
    def test_view_names_and_features(self):
        """Test view names and feature names generation."""
        data = generate_synthetic_data(num_sources=3)
        
        # Check view names
        expected_views = ['view_1', 'view_2', 'view_3']
        assert data['view_names'] == expected_views
        
        # Check feature names structure
        for view in expected_views:
            assert view in data['feature_names']
            assert isinstance(data['feature_names'][view], list)
            assert len(data['feature_names'][view]) > 0
            
    def test_clinical_data_structure(self):
        """Test clinical data structure."""
        data = generate_synthetic_data()
        
        clinical = data['clinical']
        assert len(clinical) == data['meta']['N']
        assert list(clinical.index) == data['subject_ids']
        
    def test_scalers_structure(self):
        """Test scalers structure."""
        data = generate_synthetic_data()
        
        scalers = data['scalers']
        for view_name in data['view_names']:
            assert view_name in scalers
            assert 'mu' in scalers[view_name]
            assert 'sd' in scalers[view_name]
            
            view_idx = data['view_names'].index(view_name)
            n_features = data['X_list'][view_idx].shape[1]
            assert scalers[view_name]['mu'].shape == (1, n_features)
            assert scalers[view_name]['sd'].shape == (1, n_features)

    def test_single_source_generation(self):
        """Test generation with single data source."""
        data = generate_synthetic_data(num_sources=1, K=2)
        
        assert len(data['X_list']) == 1
        assert len(data['view_names']) == 1
        assert data['view_names'] == ['view_1']
        assert data['ground_truth']['K_true'] == 2
        
    def test_large_k_generation(self):
        """Test generation with large number of factors."""
        data = generate_synthetic_data(K=10)
        
        K = data['ground_truth']['K_true']
        assert K == 10
        assert data['ground_truth']['Z'].shape[1] == K
        assert data['ground_truth']['W'].shape[1] == K


@pytest.mark.unit  
@pytest.mark.data
class TestSyntheticDataValidation:
    """Test validation of synthetic data properties."""
    
    def test_data_matrix_properties(self):
        """Test properties of generated data matrices."""
        data = generate_synthetic_data()
        
        for X in data['X_list']:
            # Check no NaN or infinite values
            assert not np.isnan(X).any()
            assert not np.isinf(X).any()
            
            # Check reasonable scale (should be roughly standardized)
            assert np.abs(np.mean(X)) < 2.0  # Mean close to 0
            assert 0.1 < np.std(X) < 10.0    # Std in reasonable range
            
    def test_ground_truth_consistency(self):
        """Test consistency between generated data and ground truth."""
        data = generate_synthetic_data()
        
        X_combined = np.concatenate(data['X_list'], axis=1)
        Z = data['ground_truth']['Z']
        W = data['ground_truth']['W']
        
        # Check dimensions match
        assert X_combined.shape[0] == Z.shape[0]  # Same number of subjects
        assert X_combined.shape[1] == W.shape[0]  # Same number of features
        assert Z.shape[1] == W.shape[1]           # Same number of factors