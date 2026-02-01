"""
Comprehensive tests for the Changerate transformation.
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime

from epylabel.labeler import Changerate


class TestChangerateBasic:
    """Basic functionality tests for Changerate."""

    def test_initialization_default_params(self):
        """Test Changerate initializes with default parameters."""
        cr = Changerate()
        assert cr.n_days == 7
        assert cr.changerate_ceiling == 3

    def test_initialization_custom_params(self):
        """Test Changerate initializes with custom parameters."""
        cr = Changerate(n_days=14, changerate_ceiling=5)
        assert cr.n_days == 14
        assert cr.changerate_ceiling == 5

    def test_transform_returns_dataframe(self, simple_wave_data):
        """Test transform returns a DataFrame."""
        cr = Changerate()
        result = cr.transform(simple_wave_data)
        assert isinstance(result, pd.DataFrame)

    def test_transform_same_shape(self, simple_wave_data):
        """Test transform returns same shape as input."""
        cr = Changerate()
        result = cr.transform(simple_wave_data)
        assert result.shape == simple_wave_data.shape

    def test_transform_same_index(self, simple_wave_data):
        """Test transform preserves index."""
        cr = Changerate()
        result = cr.transform(simple_wave_data)
        pd.testing.assert_index_equal(result.index, simple_wave_data.index)

    def test_transform_same_columns(self, simple_wave_data):
        """Test transform preserves columns."""
        cr = Changerate()
        result = cr.transform(simple_wave_data)
        pd.testing.assert_index_equal(result.columns, simple_wave_data.columns)


class TestChangerateValues:
    """Tests for Changerate value computation."""

    def test_constant_values_zero_changerate(self):
        """Test that constant values produce zero change rate."""
        dates = pd.date_range("2020-01-01", periods=30)
        df = pd.DataFrame({"0": np.ones(30) * 100}, index=dates)

        cr = Changerate(n_days=7, changerate_ceiling=10)
        result = cr.transform(df)

        # After initial NaN period, values should be near zero
        assert np.allclose(result.iloc[7:].values, 0, atol=1e-10)

    def test_increasing_values_positive_changerate(self):
        """Test that increasing values produce positive change rate."""
        dates = pd.date_range("2020-01-01", periods=30)
        # Linear increase
        df = pd.DataFrame({"0": np.arange(30) * 10.0}, index=dates)

        cr = Changerate(n_days=7, changerate_ceiling=100)
        result = cr.transform(df)

        # Change rate should be positive
        assert (result.iloc[7:] > 0).all().all()

    def test_decreasing_values_negative_changerate(self):
        """Test that decreasing values produce negative change rate."""
        dates = pd.date_range("2020-01-01", periods=30)
        # Linear decrease
        df = pd.DataFrame({"0": 100 - np.arange(30) * 3.0}, index=dates)

        cr = Changerate(n_days=7, changerate_ceiling=100)
        result = cr.transform(df)

        # Change rate should be negative
        assert (result.iloc[7:] < 0).all().all()

    def test_ceiling_applied(self):
        """Test that ceiling is applied to change rate."""
        dates = pd.date_range("2020-01-01", periods=30)
        # Exponential growth - should exceed ceiling
        df = pd.DataFrame({"0": np.exp(np.arange(30) * 0.5)}, index=dates)

        ceiling = 5
        cr = Changerate(n_days=7, changerate_ceiling=ceiling)
        result = cr.transform(df)

        # Values should not exceed ceiling
        assert (result.iloc[7:] <= ceiling).all().all()

    def test_nan_handling_initial_period(self):
        """Test that initial period produces NaN due to rolling window."""
        dates = pd.date_range("2020-01-01", periods=30)
        df = pd.DataFrame({"0": np.random.rand(30) * 100}, index=dates)

        n_days = 7
        cr = Changerate(n_days=n_days, changerate_ceiling=10)
        result = cr.transform(df)

        # First n_days-1 values should be NaN
        assert result.iloc[:n_days-1].isna().all().all()


class TestChangerateMultiLocation:
    """Tests for Changerate with multiple locations."""

    def test_multi_location_shape(self, multi_location_data):
        """Test transform works with multiple locations."""
        cr = Changerate()
        result = cr.transform(multi_location_data)
        assert result.shape == multi_location_data.shape

    def test_multi_location_independent(self, multi_location_data):
        """Test that each location is processed independently."""
        cr = Changerate()
        result = cr.transform(multi_location_data)

        # Process each location separately and compare
        for col in multi_location_data.columns:
            single_loc = multi_location_data[[col]]
            single_result = cr.transform(single_loc)
            pd.testing.assert_frame_equal(result[[col]], single_result)


class TestChangerateEdgeCases:
    """Edge case tests for Changerate."""

    def test_all_zeros(self, edge_cases_data):
        """Test handling of all-zero data."""
        cr = Changerate()
        # Column 0 is all zeros
        zero_data = edge_cases_data[["0"]]
        result = cr.transform(zero_data)
        # Should not raise error and should produce finite values or NaN
        assert not np.isinf(result.values).any()

    def test_single_spike(self, edge_cases_data):
        """Test handling of single spike in data."""
        cr = Changerate()
        # Column 2 has single spike
        spike_data = edge_cases_data[["2"]]
        result = cr.transform(spike_data)
        # Should produce high positive followed by high negative change rate
        assert result.max().max() > 0
        assert result.min().min() < 0

    def test_step_function(self, edge_cases_data):
        """Test handling of step function."""
        cr = Changerate()
        # Column 3 is step function
        step_data = edge_cases_data[["3"]]
        result = cr.transform(step_data)
        # Should detect the step
        assert result.max().max() > 0

    def test_short_series(self, short_series_data):
        """Test handling of short time series."""
        cr = Changerate(n_days=7)
        result = cr.transform(short_series_data)
        assert result.shape == short_series_data.shape


class TestChangerateNDays:
    """Tests for different n_days parameter values."""

    @pytest.mark.parametrize("n_days", [3, 7, 14, 21, 28])
    def test_different_n_days(self, simple_wave_data, n_days):
        """Test Changerate with different rolling window sizes."""
        cr = Changerate(n_days=n_days, changerate_ceiling=10)
        result = cr.transform(simple_wave_data)

        # Should produce valid output
        assert result.shape == simple_wave_data.shape
        # First n_days-1 values should be NaN
        assert result.iloc[:n_days-1].isna().all().all()

    def test_n_days_affects_smoothing(self, simple_wave_data):
        """Test that larger n_days produces smoother results."""
        cr_short = Changerate(n_days=3, changerate_ceiling=100)
        cr_long = Changerate(n_days=14, changerate_ceiling=100)

        result_short = cr_short.transform(simple_wave_data)
        result_long = cr_long.transform(simple_wave_data)

        # Longer window should have lower variance (more smoothing)
        var_short = result_short.dropna().var().values[0]
        var_long = result_long.dropna().var().values[0]
        assert var_long < var_short


class TestChangerateCeiling:
    """Tests for different changerate_ceiling parameter values."""

    @pytest.mark.parametrize("ceiling", [1, 3, 5, 10, 100])
    def test_different_ceilings(self, simple_wave_data, ceiling):
        """Test Changerate with different ceiling values."""
        cr = Changerate(n_days=7, changerate_ceiling=ceiling)
        result = cr.transform(simple_wave_data)

        # Values should not exceed ceiling
        assert (result.dropna() <= ceiling).all().all()

    def test_ceiling_zero_produces_zeros(self):
        """Test that ceiling of 0 caps all positive values to 0."""
        dates = pd.date_range("2020-01-01", periods=30)
        df = pd.DataFrame({"0": np.exp(np.arange(30) * 0.1)}, index=dates)

        cr = Changerate(n_days=7, changerate_ceiling=0)
        result = cr.transform(df)

        # All positive values should be capped to 0
        assert (result.dropna() <= 0).all().all()
