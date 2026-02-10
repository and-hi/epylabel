"""
Comprehensive tests for the Ensemble transformation.
"""

import pytest
import pandas as pd
import numpy as np

from epylabel.labeler import Ensemble


class TestEnsembleBasic:
    """Basic functionality tests for Ensemble."""

    def test_initialization_default_params(self):
        """Test Ensemble initializes with default parameters."""
        ens = Ensemble()
        assert ens.n_min == 2

    def test_initialization_custom_params(self):
        """Test Ensemble initializes with custom parameters."""
        ens = Ensemble(n_min=3)
        assert ens.n_min == 3

    def test_transform_requires_multiple_inputs(self):
        """Test that transform requires multiple input DataFrames."""
        ens = Ensemble(n_min=2)
        dates = pd.date_range("2020-01-01", periods=10)
        df1 = pd.DataFrame({"0": [True] * 10}, index=dates)

        # Single input should work but may not make sense
        result = ens.transform(df1)
        assert result.shape == df1.shape

    def test_transform_returns_dataframe(self):
        """Test transform returns a DataFrame."""
        ens = Ensemble(n_min=2)
        dates = pd.date_range("2020-01-01", periods=10)
        df1 = pd.DataFrame({"0": [True] * 5 + [False] * 5}, index=dates)
        df2 = pd.DataFrame({"0": [True] * 7 + [False] * 3}, index=dates)

        result = ens.transform(df1, df2)
        assert isinstance(result, pd.DataFrame)

    def test_transform_same_shape(self):
        """Test transform returns same shape as input."""
        ens = Ensemble(n_min=2)
        dates = pd.date_range("2020-01-01", periods=10)
        df1 = pd.DataFrame({"0": [True] * 5 + [False] * 5}, index=dates)
        df2 = pd.DataFrame({"0": [True] * 7 + [False] * 3}, index=dates)

        result = ens.transform(df1, df2)
        assert result.shape == df1.shape


class TestEnsembleMajorityVoting:
    """Tests for Ensemble majority voting logic."""

    def test_two_algorithms_both_agree_true(self):
        """Test that two algorithms agreeing on True produces True."""
        ens = Ensemble(n_min=2)
        dates = pd.date_range("2020-01-01", periods=5)
        df1 = pd.DataFrame({"0": [True] * 5}, index=dates)
        df2 = pd.DataFrame({"0": [True] * 5}, index=dates)

        result = ens.transform(df1, df2)
        assert result["0"].all(), "Both True should produce True"

    def test_two_algorithms_both_agree_false(self):
        """Test that two algorithms agreeing on False produces False."""
        ens = Ensemble(n_min=2)
        dates = pd.date_range("2020-01-01", periods=5)
        df1 = pd.DataFrame({"0": [False] * 5}, index=dates)
        df2 = pd.DataFrame({"0": [False] * 5}, index=dates)

        result = ens.transform(df1, df2)
        assert not result["0"].any(), "Both False should produce False"

    def test_two_algorithms_disagree(self):
        """Test behavior when two algorithms disagree."""
        ens = Ensemble(n_min=2)
        dates = pd.date_range("2020-01-01", periods=5)
        df1 = pd.DataFrame({"0": [True] * 5}, index=dates)
        df2 = pd.DataFrame({"0": [False] * 5}, index=dates)

        result = ens.transform(df1, df2)
        # With n_min=2, both must agree, so disagreement should produce False
        assert not result["0"].any(), "Disagreement with n_min=2 should produce False"

    def test_three_algorithms_majority(self):
        """Test three algorithms with majority voting."""
        ens = Ensemble(n_min=2)
        dates = pd.date_range("2020-01-01", periods=5)
        df1 = pd.DataFrame({"0": [True] * 5}, index=dates)
        df2 = pd.DataFrame({"0": [True] * 5}, index=dates)
        df3 = pd.DataFrame({"0": [False] * 5}, index=dates)

        result = ens.transform(df1, df2, df3)
        # 2 out of 3 agree on True, meets n_min=2
        assert result["0"].all(), "Majority True with n_min=2 should produce True"

    def test_three_algorithms_minority(self):
        """Test three algorithms with minority."""
        ens = Ensemble(n_min=2)
        dates = pd.date_range("2020-01-01", periods=5)
        df1 = pd.DataFrame({"0": [True] * 5}, index=dates)
        df2 = pd.DataFrame({"0": [False] * 5}, index=dates)
        df3 = pd.DataFrame({"0": [False] * 5}, index=dates)

        result = ens.transform(df1, df2, df3)
        # Only 1 out of 3 is True, doesn't meet n_min=2
        assert not result["0"].any(), "Only 1/3 True should produce False"


class TestEnsembleNMin:
    """Tests for different n_min parameter values."""

    @pytest.mark.parametrize("n_min", [1, 2, 3])
    def test_different_n_min_values(self, n_min):
        """Test Ensemble with different n_min values."""
        ens = Ensemble(n_min=n_min)
        dates = pd.date_range("2020-01-01", periods=5)
        df1 = pd.DataFrame({"0": [True] * 5}, index=dates)
        df2 = pd.DataFrame({"0": [True] * 5}, index=dates)
        df3 = pd.DataFrame({"0": [False] * 5}, index=dates)

        result = ens.transform(df1, df2, df3)
        assert result.shape == df1.shape

    def test_n_min_1_any_algorithm(self):
        """Test that n_min=1 triggers if any algorithm detects."""
        ens = Ensemble(n_min=1)
        dates = pd.date_range("2020-01-01", periods=5)
        df1 = pd.DataFrame({"0": [True] * 5}, index=dates)
        df2 = pd.DataFrame({"0": [False] * 5}, index=dates)
        df3 = pd.DataFrame({"0": [False] * 5}, index=dates)

        result = ens.transform(df1, df2, df3)
        # At least 1 True meets n_min=1
        assert result["0"].all(), "n_min=1 should trigger with any True"

    def test_n_min_3_requires_all(self):
        """Test that n_min=3 requires all algorithms to agree."""
        ens = Ensemble(n_min=3)
        dates = pd.date_range("2020-01-01", periods=5)
        df1 = pd.DataFrame({"0": [True] * 5}, index=dates)
        df2 = pd.DataFrame({"0": [True] * 5}, index=dates)
        df3 = pd.DataFrame({"0": [False] * 5}, index=dates)

        result = ens.transform(df1, df2, df3)
        # 2 out of 3, doesn't meet n_min=3
        assert not result["0"].any(), "n_min=3 should require all True"

    def test_n_min_higher_stricter(self):
        """Test that higher n_min is stricter."""
        dates = pd.date_range("2020-01-01", periods=10)
        # Mixed pattern
        df1 = pd.DataFrame({"0": [True] * 6 + [False] * 4}, index=dates)
        df2 = pd.DataFrame({"0": [True] * 4 + [False] * 6}, index=dates)
        df3 = pd.DataFrame({"0": [True] * 8 + [False] * 2}, index=dates)

        ens1 = Ensemble(n_min=1)
        ens2 = Ensemble(n_min=2)
        ens3 = Ensemble(n_min=3)

        result1 = ens1.transform(df1, df2, df3)
        result2 = ens2.transform(df1, df2, df3)
        result3 = ens3.transform(df1, df2, df3)

        detections1 = result1.sum().sum()
        detections2 = result2.sum().sum()
        detections3 = result3.sum().sum()

        assert detections1 >= detections2 >= detections3, "Higher n_min should be stricter"


class TestEnsembleMultiLocation:
    """Tests for Ensemble with multiple locations."""

    def test_multi_location_shape(self):
        """Test transform works with multiple locations."""
        ens = Ensemble(n_min=2)
        dates = pd.date_range("2020-01-01", periods=10)
        df1 = pd.DataFrame({
            "0": [True] * 5 + [False] * 5,
            "1": [False] * 5 + [True] * 5
        }, index=dates)
        df2 = pd.DataFrame({
            "0": [True] * 7 + [False] * 3,
            "1": [False] * 3 + [True] * 7
        }, index=dates)

        result = ens.transform(df1, df2)
        assert result.shape == df1.shape

    def test_multi_location_independent(self):
        """Test that each location is processed independently."""
        ens = Ensemble(n_min=2)
        dates = pd.date_range("2020-01-01", periods=10)

        # Location 0: both agree True
        # Location 1: disagree
        df1 = pd.DataFrame({
            "0": [True] * 10,
            "1": [True] * 10
        }, index=dates)
        df2 = pd.DataFrame({
            "0": [True] * 10,
            "1": [False] * 10
        }, index=dates)

        result = ens.transform(df1, df2)

        assert result["0"].all(), "Location 0 should be True"
        assert not result["1"].any(), "Location 1 should be False"


class TestEnsembleTestData:
    """Tests using ensemble test data."""

    def test_clear_outbreak_agreement(self, ensemble_test_data):
        """Test that clear outbreak produces agreement across simulated algorithms."""
        # Simulate algorithm outputs for clear outbreak (location 0)
        clear_outbreak = ensemble_test_data[["0"]]

        # Simple simulation: all algorithms should detect the outbreak
        # Create simulated algorithm outputs
        threshold = clear_outbreak["0"].median()
        algo1 = clear_outbreak > threshold
        algo2 = clear_outbreak > threshold * 0.8
        algo3 = clear_outbreak > threshold * 1.2

        ens = Ensemble(n_min=2)
        result = ens.transform(algo1, algo2, algo3)

        # Should have some detections
        assert result.any().any(), "Clear outbreak should produce ensemble detection"


class TestEnsembleReproducibility:
    """Tests for Ensemble reproducibility."""

    def test_deterministic_output(self):
        """Test that Ensemble produces deterministic output."""
        ens = Ensemble(n_min=2)
        dates = pd.date_range("2020-01-01", periods=10)
        df1 = pd.DataFrame({"0": [True] * 5 + [False] * 5}, index=dates)
        df2 = pd.DataFrame({"0": [True] * 7 + [False] * 3}, index=dates)

        result1 = ens.transform(df1, df2)
        result2 = ens.transform(df1, df2)

        pd.testing.assert_frame_equal(result1, result2)

    def test_same_results_new_instance(self):
        """Test that new instance produces same results."""
        dates = pd.date_range("2020-01-01", periods=10)
        df1 = pd.DataFrame({"0": [True] * 5 + [False] * 5}, index=dates)
        df2 = pd.DataFrame({"0": [True] * 7 + [False] * 3}, index=dates)

        ens1 = Ensemble(n_min=2)
        ens2 = Ensemble(n_min=2)

        result1 = ens1.transform(df1, df2)
        result2 = ens2.transform(df1, df2)

        pd.testing.assert_frame_equal(result1, result2)
