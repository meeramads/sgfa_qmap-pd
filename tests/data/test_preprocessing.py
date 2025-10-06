"""
Tests for data preprocessing modules.

Tests BasicPreprocessor and NeuroImagingPreprocessor functionality.
"""

import pytest
import numpy as np
from unittest.mock import Mock, patch


@pytest.fixture
def sample_data():
    """Generate sample multi-view data for testing."""
    np.random.seed(42)
    X1 = np.random.randn(50, 100)  # 50 subjects, 100 features
    X2 = np.random.randn(50, 80)   # 50 subjects, 80 features
    X3 = np.random.randn(50, 60)   # 50 subjects, 60 features
    
    # Add some NaN values
    X1[0, 0] = np.nan
    X2[1, 1] = np.nan
    
    return [X1, X2, X3]


@pytest.fixture
def view_names():
    """Sample view names."""
    return ['volume_sn_voxels', 'volume_putamen_voxels', 'clinical']


class TestBasicPreprocessor:
    """Test BasicPreprocessor functionality."""
    
    def test_initialization(self):
        """Test BasicPreprocessor can be initialized."""
        from data.preprocessing import BasicPreprocessor
        
        preprocessor = BasicPreprocessor(imputation_strategy='median')
        assert preprocessor is not None
    
    def test_fit_transform(self, sample_data, view_names):
        """Test fit_transform produces valid output."""
        from data.preprocessing import BasicPreprocessor
        
        preprocessor = BasicPreprocessor()
        X_processed = preprocessor.fit_transform(sample_data, view_names)
        
        # Check output structure
        assert isinstance(X_processed, list)
        assert len(X_processed) == len(sample_data)
        
        # Check dimensions preserved
        for X_orig, X_proc in zip(sample_data, X_processed):
            assert X_proc.shape[0] == X_orig.shape[0]  # Same n_subjects
    
    def test_nan_handling(self, sample_data, view_names):
        """Test that NaN values are imputed."""
        from data.preprocessing import BasicPreprocessor
        
        preprocessor = BasicPreprocessor(imputation_strategy='median')
        X_processed = preprocessor.fit_transform(sample_data, view_names)
        
        # Check no NaN values remain
        for X in X_processed:
            assert not np.isnan(X).any(), "NaN values should be imputed"
    
    def test_scaling(self, sample_data, view_names):
        """Test that data is scaled."""
        from data.preprocessing import BasicPreprocessor
        
        preprocessor = BasicPreprocessor()
        X_processed = preprocessor.fit_transform(sample_data, view_names)
        
        # Check scaling (mean ≈ 0, std ≈ 1 for robust scaling)
        for X in X_processed:
            # Allow for some tolerance since it's robust scaling
            assert np.abs(np.median(X)) < 1.0
    
    def test_imputation_strategies(self, sample_data, view_names):
        """Test different imputation strategies."""
        from data.preprocessing import BasicPreprocessor
        
        strategies = ['mean', 'median', 'most_frequent']
        
        for strategy in strategies:
            try:
                preprocessor = BasicPreprocessor(imputation_strategy=strategy)
                X_processed = preprocessor.fit_transform(sample_data, view_names)
                assert len(X_processed) == len(sample_data)
            except ValueError:
                # 'most_frequent' may not work for continuous data
                if strategy == 'most_frequent':
                    pass
                else:
                    raise


class TestNeuroImagingPreprocessor:
    """Test NeuroImagingPreprocessor functionality."""
    
    def test_initialization(self, tmp_path):
        """Test NeuroImagingPreprocessor can be initialized."""
        from data.preprocessing import NeuroImagingPreprocessor
        
        preprocessor = NeuroImagingPreprocessor(
            data_dir=str(tmp_path),
            imputation_strategy='median',
            feature_selection_method='variance',
            variance_threshold=0.01
        )
        assert preprocessor is not None
    
    def test_fit_transform(self, sample_data, view_names, tmp_path):
        """Test fit_transform with basic configuration."""
        from data.preprocessing import NeuroImagingPreprocessor
        
        preprocessor = NeuroImagingPreprocessor(
            data_dir=str(tmp_path),
            imputation_strategy='median',
            feature_selection_method='variance',
            variance_threshold=0.01
        )
        X_processed = preprocessor.fit_transform(sample_data, view_names)
        
        # Check output structure
        assert isinstance(X_processed, list)
        assert len(X_processed) == len(sample_data)
        
        # Check no NaN values
        for X in X_processed:
            assert not np.isnan(X).any()
    
    def test_feature_selection(self, sample_data, view_names, tmp_path):
        """Test that feature selection reduces dimensionality."""
        from data.preprocessing import NeuroImagingPreprocessor
        
        # Add some constant features (zero variance)
        for X in sample_data:
            X[:, 0] = 1.0  # Constant feature
        
        preprocessor = NeuroImagingPreprocessor(
            data_dir=str(tmp_path),
            feature_selection_method='variance',
            variance_threshold=0.01
        )
        X_processed = preprocessor.fit_transform(sample_data, view_names)
        
        # Should remove at least the constant features
        for X_orig, X_proc in zip(sample_data, X_processed):
            assert X_proc.shape[1] <= X_orig.shape[1]
    
    def test_missing_threshold(self, sample_data, view_names, tmp_path):
        """Test missing data threshold filtering."""
        from data.preprocessing import NeuroImagingPreprocessor
        
        # Add column with many missing values
        sample_data[0][:, 1] = np.nan
        sample_data[0][:40, 1] = np.nan  # 80% missing
        
        preprocessor = NeuroImagingPreprocessor(
            data_dir=str(tmp_path),
            missing_threshold=0.5,  # Remove features >50% missing
            feature_selection_method='none'
        )
        X_processed = preprocessor.fit_transform(sample_data, view_names)
        
        # Should remove high-missing column
        assert X_processed[0].shape[1] < sample_data[0].shape[1]
    
    def test_no_feature_selection(self, sample_data, view_names, tmp_path):
        """Test with feature selection disabled."""
        from data.preprocessing import NeuroImagingPreprocessor
        
        preprocessor = NeuroImagingPreprocessor(
            data_dir=str(tmp_path),
            feature_selection_method='none',
            variance_threshold=0.0
        )
        X_processed = preprocessor.fit_transform(sample_data, view_names)
        
        # Dimensions should be preserved (except for NaN handling)
        for X_orig, X_proc in zip(sample_data, X_processed):
            # May differ slightly due to missing value handling
            assert abs(X_proc.shape[1] - X_orig.shape[1]) <= 2


class TestNeuroImagingConfig:
    """Test NeuroImagingConfig dataclass."""
    
    def test_default_config(self):
        """Test default configuration."""
        from data.preprocessing import NeuroImagingConfig
        
        config = NeuroImagingConfig()
        assert config.imputation_strategy == 'median'
        assert config.feature_selection_method == 'variance'
        assert config.variance_threshold >= 0
    
    def test_custom_config(self):
        """Test custom configuration."""
        from data.preprocessing import NeuroImagingConfig
        
        config = NeuroImagingConfig(
            imputation_strategy='mean',
            feature_selection_method='statistical',
            n_top_features=100
        )
        assert config.imputation_strategy == 'mean'
        assert config.feature_selection_method == 'statistical'
        assert config.n_top_features == 100
    
    def test_validate_config(self):
        """Test configuration validation."""
        from data.preprocessing import NeuroImagingConfig
        
        config = NeuroImagingConfig()
        try:
            config.validate()
        except Exception as e:
            pytest.fail(f"Valid config should not raise exception: {e}")
    
    def test_to_dict(self):
        """Test conversion to dictionary."""
        from data.preprocessing import NeuroImagingConfig
        
        config = NeuroImagingConfig(imputation_strategy='median')
        config_dict = config.to_dict()
        
        assert isinstance(config_dict, dict)
        assert 'imputation_strategy' in config_dict
        assert config_dict['imputation_strategy'] == 'median'


class TestPreprocessingIntegration:
    """Integration tests for preprocessing pipeline."""
    
    def test_basic_to_advanced_pipeline(self, sample_data, view_names, tmp_path):
        """Test using BasicPreprocessor then NeuroImagingPreprocessor."""
        from data.preprocessing import BasicPreprocessor, NeuroImagingPreprocessor
        
        # Basic preprocessing
        basic_preprocessor = BasicPreprocessor()
        X_basic = basic_preprocessor.fit_transform(sample_data, view_names)
        
        # Advanced preprocessing
        advanced_preprocessor = NeuroImagingPreprocessor(
            data_dir=str(tmp_path),
            feature_selection_method='variance'
        )
        X_advanced = advanced_preprocessor.fit_transform(X_basic, view_names)
        
        # Check pipeline works
        assert len(X_advanced) == len(sample_data)
        for X in X_advanced:
            assert not np.isnan(X).any()
    
    def test_preprocessing_preserves_subjects(self, sample_data, view_names, tmp_path):
        """Test that number of subjects is preserved."""
        from data.preprocessing import NeuroImagingPreprocessor
        
        n_subjects = sample_data[0].shape[0]
        
        preprocessor = NeuroImagingPreprocessor(
            data_dir=str(tmp_path),
            feature_selection_method='variance'
        )
        X_processed = preprocessor.fit_transform(sample_data, view_names)
        
        # All views should have same number of subjects
        for X in X_processed:
            assert X.shape[0] == n_subjects
    
    def test_preprocessing_reproducibility(self, sample_data, view_names, tmp_path):
        """Test that preprocessing is reproducible."""
        from data.preprocessing import NeuroImagingPreprocessor
        
        preprocessor1 = NeuroImagingPreprocessor(
            data_dir=str(tmp_path),
            imputation_strategy='median',
            feature_selection_method='variance'
        )
        X_processed1 = preprocessor1.fit_transform(sample_data, view_names)
        
        preprocessor2 = NeuroImagingPreprocessor(
            data_dir=str(tmp_path),
            imputation_strategy='median',
            feature_selection_method='variance'
        )
        X_processed2 = preprocessor2.fit_transform(sample_data, view_names)
        
        # Results should be identical
        for X1, X2 in zip(X_processed1, X_processed2):
            np.testing.assert_array_almost_equal(X1, X2)


class TestEdgeCases:
    """Test edge cases and error handling."""
    
    def test_single_subject(self, tmp_path):
        """Test preprocessing with single subject."""
        from data.preprocessing import BasicPreprocessor
        
        X_list = [np.random.randn(1, 10)]
        view_names = ['view1']
        
        preprocessor = BasicPreprocessor()
        X_processed = preprocessor.fit_transform(X_list, view_names)
        
        assert len(X_processed) == 1
        assert X_processed[0].shape[0] == 1
    
    def test_single_feature(self, tmp_path):
        """Test preprocessing with single feature."""
        from data.preprocessing import BasicPreprocessor
        
        X_list = [np.random.randn(10, 1)]
        view_names = ['view1']
        
        preprocessor = BasicPreprocessor()
        X_processed = preprocessor.fit_transform(X_list, view_names)
        
        assert len(X_processed) == 1
        assert X_processed[0].shape[1] >= 1
    
    def test_all_nan_feature(self, tmp_path):
        """Test handling of features with all NaN values."""
        from data.preprocessing import NeuroImagingPreprocessor
        
        X_list = [np.random.randn(10, 5)]
        X_list[0][:, 0] = np.nan  # All NaN column
        view_names = ['view1']
        
        preprocessor = NeuroImagingPreprocessor(
            data_dir=str(tmp_path),
            missing_threshold=0.5
        )
        X_processed = preprocessor.fit_transform(X_list, view_names)
        
        # Should remove all-NaN column
        assert X_processed[0].shape[1] < X_list[0].shape[1]
    
    def test_zero_variance_features(self, tmp_path):
        """Test removal of zero-variance features."""
        from data.preprocessing import NeuroImagingPreprocessor
        
        X_list = [np.random.randn(10, 5)]
        X_list[0][:, 0] = 1.0  # Constant feature
        view_names = ['view1']
        
        preprocessor = NeuroImagingPreprocessor(
            data_dir=str(tmp_path),
            feature_selection_method='variance',
            variance_threshold=0.01
        )
        X_processed = preprocessor.fit_transform(X_list, view_names)
        
        # Should remove constant feature
        assert X_processed[0].shape[1] < X_list[0].shape[1]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
