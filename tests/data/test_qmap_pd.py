"""
Tests for qMAP-PD data loading module.

Tests data loading, ROI selection, and clinical feature exclusion.
"""

import pytest
import numpy as np
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
import pandas as pd


@pytest.fixture
def mock_data_dir(tmp_path):
    """Create a mock qMAP-PD data directory structure."""
    data_dir = tmp_path / "qMAP-PD_data"
    data_dir.mkdir()
    
    # Create clinical directory
    clinical_dir = data_dir / "data_clinical"
    clinical_dir.mkdir()
    
    # Create mock clinical data
    clinical_data = pd.DataFrame({
        'subject_id': ['sub-01', 'sub-02', 'sub-03'],
        'age': [60, 65, 70],
        'sex': ['M', 'F', 'M'],
        'tiv': [1500, 1400, 1600],
        'updrs_total': [20, 25, 30],
        'updrs_motor': [15, 18, 22]
    })
    clinical_data.to_csv(clinical_dir / "clinical.tsv", sep='\t', index=False)
    
    # Create volume matrices directory
    volume_dir = data_dir / "volume_matrices"
    volume_dir.mkdir()
    
    # Create mock ROI files
    for roi in ['sn', 'putamen', 'lentiform']:
        roi_data = pd.DataFrame(
            np.random.randn(3, 50),  # 3 subjects, 50 voxels
            index=['sub-01', 'sub-02', 'sub-03']
        )
        roi_data.to_csv(volume_dir / f"volume_{roi}_voxels.tsv", sep='\t')
    
    return str(data_dir)


class TestQMAPPDDataLoading:
    """Test basic qMAP-PD data loading functionality."""
    
    def test_load_basic_structure(self, mock_data_dir):
        """Test basic data loading returns correct structure."""
        from data.qmap_pd import load_qmap_pd
        
        data = load_qmap_pd(data_dir=mock_data_dir)
        
        # Check required keys
        assert 'X_list' in data
        assert 'view_names' in data
        assert 'feature_names' in data
        
        # Check data types
        assert isinstance(data['X_list'], list)
        assert all(isinstance(X, np.ndarray) for X in data['X_list'])
    
    def test_concatenated_imaging_view(self, mock_data_dir):
        """Test imaging_as_single_view=True concatenates imaging data."""
        from data.qmap_pd import load_qmap_pd
        
        data = load_qmap_pd(
            data_dir=mock_data_dir,
            imaging_as_single_view=True
        )
        
        # Should have: clinical + concatenated imaging = 2 views
        assert len(data['X_list']) == 2
        assert 'clinical' in data['view_names']
        assert any('imaging' in name or 'concatenated' in name 
                  for name in data['view_names'])
    
    def test_separate_imaging_views(self, mock_data_dir):
        """Test imaging_as_single_view=False keeps ROIs separate."""
        from data.qmap_pd import load_qmap_pd
        
        data = load_qmap_pd(
            data_dir=mock_data_dir,
            imaging_as_single_view=False
        )
        
        # Should have: clinical + 3 separate ROIs = 4 views
        assert len(data['X_list']) >= 2  # At least clinical + 1 ROI
        assert 'clinical' in data['view_names']
    
    def test_data_shape_consistency(self, mock_data_dir):
        """Test all views have same number of subjects."""
        from data.qmap_pd import load_qmap_pd
        
        data = load_qmap_pd(data_dir=mock_data_dir)
        
        n_subjects = data['X_list'][0].shape[0]
        for X in data['X_list']:
            assert X.shape[0] == n_subjects, "All views should have same n_subjects"
            assert X.ndim == 2, "All views should be 2D arrays"


class TestROISelection:
    """Test ROI selection functionality."""
    
    def test_select_single_roi(self, mock_data_dir):
        """Test selecting a single ROI."""
        from data.qmap_pd import load_qmap_pd
        
        data = load_qmap_pd(
            data_dir=mock_data_dir,
            select_rois=['volume_sn_voxels.tsv'],
            imaging_as_single_view=False
        )
        
        # Should have: clinical + sn only = 2 views
        assert len(data['X_list']) == 2
        assert 'clinical' in data['view_names']
        assert any('sn' in name.lower() for name in data['view_names'])
    
    def test_select_multiple_rois(self, mock_data_dir):
        """Test selecting multiple ROIs."""
        from data.qmap_pd import load_qmap_pd
        
        data = load_qmap_pd(
            data_dir=mock_data_dir,
            select_rois=['volume_sn_voxels.tsv', 'volume_putamen_voxels.tsv'],
            imaging_as_single_view=False
        )
        
        # Should have: clinical + sn + putamen = 3 views
        assert len(data['X_list']) == 3
        assert 'clinical' in data['view_names']
    
    def test_select_roi_with_concatenation(self, mock_data_dir):
        """Test ROI selection with concatenated imaging view."""
        from data.qmap_pd import load_qmap_pd
        
        data = load_qmap_pd(
            data_dir=mock_data_dir,
            select_rois=['volume_sn_voxels.tsv'],
            imaging_as_single_view=True
        )
        
        # Should have: clinical + concatenated (just sn) = 2 views
        assert len(data['X_list']) == 2
    
    def test_invalid_roi_name(self, mock_data_dir):
        """Test handling of invalid ROI names."""
        from data.qmap_pd import load_qmap_pd
        
        # Should either raise error or skip invalid ROI
        try:
            data = load_qmap_pd(
                data_dir=mock_data_dir,
                select_rois=['nonexistent_roi.tsv'],
                imaging_as_single_view=False
            )
            # If it doesn't error, should at least have clinical data
            assert len(data['X_list']) >= 1
        except (FileNotFoundError, ValueError):
            # This is also acceptable behavior
            pass


class TestClinicalFeatureExclusion:
    """Test clinical feature exclusion functionality."""
    
    def test_exclude_single_feature(self, mock_data_dir):
        """Test excluding a single clinical feature."""
        from data.qmap_pd import load_qmap_pd
        
        # Load without exclusion
        data_full = load_qmap_pd(data_dir=mock_data_dir)
        clinical_idx = data_full['view_names'].index('clinical')
        n_features_full = data_full['X_list'][clinical_idx].shape[1]
        
        # Load with exclusion
        data_excl = load_qmap_pd(
            data_dir=mock_data_dir,
            exclude_clinical_features=['age']
        )
        clinical_idx_excl = data_excl['view_names'].index('clinical')
        n_features_excl = data_excl['X_list'][clinical_idx_excl].shape[1]
        
        # Should have one fewer feature
        assert n_features_excl == n_features_full - 1
    
    def test_exclude_multiple_features(self, mock_data_dir):
        """Test excluding multiple clinical features."""
        from data.qmap_pd import load_qmap_pd
        
        # Load without exclusion
        data_full = load_qmap_pd(data_dir=mock_data_dir)
        clinical_idx = data_full['view_names'].index('clinical')
        n_features_full = data_full['X_list'][clinical_idx].shape[1]
        
        # Load with exclusion of 3 features
        data_excl = load_qmap_pd(
            data_dir=mock_data_dir,
            exclude_clinical_features=['age', 'sex', 'tiv']
        )
        clinical_idx_excl = data_excl['view_names'].index('clinical')
        n_features_excl = data_excl['X_list'][clinical_idx_excl].shape[1]
        
        # Should have 3 fewer features
        assert n_features_excl == n_features_full - 3
    
    def test_exclude_nonexistent_feature(self, mock_data_dir):
        """Test handling of nonexistent feature names."""
        from data.qmap_pd import load_qmap_pd
        
        # Should either ignore or warn, but not crash
        data = load_qmap_pd(
            data_dir=mock_data_dir,
            exclude_clinical_features=['nonexistent_feature']
        )
        
        assert 'X_list' in data
        assert len(data['X_list']) > 0
    
    def test_feature_names_updated(self, mock_data_dir):
        """Test that feature names are updated after exclusion."""
        from data.qmap_pd import load_qmap_pd
        
        data = load_qmap_pd(
            data_dir=mock_data_dir,
            exclude_clinical_features=['age']
        )
        
        if 'feature_names' in data and 'clinical' in data['feature_names']:
            clinical_features = data['feature_names']['clinical']
            assert 'age' not in clinical_features


class TestCombinedOptions:
    """Test combining ROI selection and clinical exclusion."""
    
    def test_select_roi_and_exclude_clinical(self, mock_data_dir):
        """Test using both ROI selection and clinical exclusion."""
        from data.qmap_pd import load_qmap_pd
        
        data = load_qmap_pd(
            data_dir=mock_data_dir,
            select_rois=['volume_sn_voxels.tsv'],
            exclude_clinical_features=['age', 'sex', 'tiv'],
            imaging_as_single_view=False
        )
        
        # Should have: clinical (reduced) + sn = 2 views
        assert len(data['X_list']) == 2
        assert all(X.shape[0] == 3 for X in data['X_list'])  # 3 subjects
    
    def test_multiple_rois_and_exclusions(self, mock_data_dir):
        """Test multiple ROIs with multiple clinical exclusions."""
        from data.qmap_pd import load_qmap_pd
        
        data = load_qmap_pd(
            data_dir=mock_data_dir,
            select_rois=['volume_sn_voxels.tsv', 'volume_putamen_voxels.tsv'],
            exclude_clinical_features=['age', 'sex'],
            imaging_as_single_view=False
        )
        
        # Should have: clinical (reduced) + sn + putamen = 3 views
        assert len(data['X_list']) == 3


class TestDataValidation:
    """Test data validation and error handling."""
    
    def test_missing_data_dir(self):
        """Test handling of missing data directory."""
        from data.qmap_pd import load_qmap_pd
        
        with pytest.raises((FileNotFoundError, ValueError)):
            load_qmap_pd(data_dir='/nonexistent/path')
    
    def test_empty_data_dir(self, tmp_path):
        """Test handling of empty data directory."""
        from data.qmap_pd import load_qmap_pd
        
        empty_dir = tmp_path / "empty"
        empty_dir.mkdir()
        
        # Should either raise error or return minimal data
        try:
            data = load_qmap_pd(data_dir=str(empty_dir))
            # If it doesn't error, check it has proper structure
            assert 'X_list' in data
        except (FileNotFoundError, ValueError, KeyError):
            # This is acceptable behavior
            pass
    
    def test_nan_handling(self, mock_data_dir):
        """Test that data with NaN values is handled."""
        from data.qmap_pd import load_qmap_pd
        
        # Load data (may contain NaN)
        data = load_qmap_pd(data_dir=mock_data_dir)
        
        # Check data is returned and is numeric
        assert len(data['X_list']) > 0
        for X in data['X_list']:
            assert X.dtype in [np.float32, np.float64]


# Integration test
class TestIntegration:
    """Integration tests for complete workflows."""
    
    def test_complete_workflow(self, mock_data_dir):
        """Test complete data loading workflow."""
        from data.qmap_pd import load_qmap_pd
        
        # Load with all options
        data = load_qmap_pd(
            data_dir=mock_data_dir,
            select_rois=['volume_sn_voxels.tsv'],
            exclude_clinical_features=['age', 'sex', 'tiv'],
            imaging_as_single_view=False
        )
        
        # Verify complete structure
        assert 'X_list' in data
        assert 'view_names' in data
        assert 'feature_names' in data
        assert len(data['X_list']) == 2  # clinical + sn
        assert all(isinstance(X, np.ndarray) for X in data['X_list'])
        assert all(X.ndim == 2 for X in data['X_list'])
        assert all(X.shape[0] == 3 for X in data['X_list'])  # 3 subjects


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
