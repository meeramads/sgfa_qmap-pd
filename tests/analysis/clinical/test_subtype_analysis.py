"""Tests for analysis.clinical.subtype_analysis module."""

from unittest.mock import Mock, patch

import numpy as np
import pytest

from analysis.clinical.subtype_analysis import PDSubtypeAnalyzer


class TestPDSubtypeAnalyzerInit:
    """Test PDSubtypeAnalyzer initialization."""

    def test_default_initialization(self):
        """Test initialization with defaults."""
        analyzer = PDSubtypeAnalyzer()

        assert analyzer.n_subtypes_range == [2, 3, 4, 5]
        assert analyzer.logger is not None

    def test_custom_initialization(self):
        """Test initialization with custom parameters."""
        analyzer = PDSubtypeAnalyzer(
            n_subtypes_range=[3, 4],
            logger=Mock()
        )

        assert analyzer.n_subtypes_range == [3, 4]


class TestDiscoverSubtypes:
    """Test discover_subtypes method."""

    @patch('analysis.clinical.subtype_analysis.KMeans')
    def test_basic_subtype_discovery(self, mock_kmeans):
        """Test basic subtype discovery."""
        analyzer = PDSubtypeAnalyzer(n_subtypes_range=[2, 3])

        # Mock KMeans
        mock_model = Mock()
        mock_model.labels_ = np.array([0, 1, 0, 1, 0])
        mock_model.inertia_ = 10.5
        mock_kmeans.return_value = mock_model

        Z = np.random.randn(50, 5)

        result = analyzer.discover_subtypes(Z)

        assert "optimal_n_subtypes" in result or "n_subtypes" in result
        assert "subtype_labels" in result or "labels" in result

    @patch('analysis.clinical.subtype_analysis.KMeans')
    def test_with_clinical_data(self, mock_kmeans):
        """Test subtype discovery with clinical data."""
        analyzer = PDSubtypeAnalyzer(n_subtypes_range=[2])

        mock_model = Mock()
        mock_model.labels_ = np.array([0, 1, 0, 1])
        mock_kmeans.return_value = mock_model

        Z = np.random.randn(20, 3)
        clinical_data = {
            "diagnosis": np.array([0, 1, 0, 1] * 5),
            "age": np.random.randint(50, 80, 20)
        }

        result = analyzer.discover_subtypes(Z, clinical_data=clinical_data)

        assert result is not None


class TestEdgeCases:
    """Test edge cases."""

    def test_single_subtype(self):
        """Test with n_subtypes_range=[1]."""
        analyzer = PDSubtypeAnalyzer(n_subtypes_range=[1])

        Z = np.random.randn(10, 3)

        result = analyzer.discover_subtypes(Z)
        # Should handle gracefully
        assert result is not None

    def test_more_subtypes_than_samples(self):
        """Test when requesting more subtypes than samples."""
        analyzer = PDSubtypeAnalyzer(n_subtypes_range=[20])

        Z = np.random.randn(5, 3)  # Only 5 samples

        with pytest.raises((ValueError, Exception)):
            analyzer.discover_subtypes(Z)
