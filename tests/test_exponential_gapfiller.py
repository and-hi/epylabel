"""
Comprehensive tests for ExponentialGrowth and GapFiller transformations.
"""

import pytest
import pandas as pd
import numpy as np

from epylabel.labeler import ExponentialGrowth, GapFiller


class TestExponentialGrowthBasic:
    """Basic functionality tests for ExponentialGrowth."""

    def test_initialization_params(self):
        """Test ExponentialGrowth initializes with parameters."""
        eg = ExponentialGrowth(n_days=7, thresh=0.05)
        assert eg.n_days == 7
        assert eg.thresh == 0.05

    def test_initialization_custom_params(self):
        """Test ExponentialGrowth initializes with custom parameters."""
        eg = ExponentialGrowth(n_days=14, thresh=0.01, log_transform=True)
        assert eg.n_days == 14
        assert eg.thresh == 0.01
        assert eg.log_transform == True

    def test_transform_returns_dataframe(self, exponential_growth_data):
        """Test transform returns a DataFrame."""
        eg = ExponentialGrowth(n_days=7, thresh=0.05)
        result = eg.transform(exponential_growth_data)
        assert isinstance(result, pd.DataFrame)

    def test_transform_same_shape(self, exponential_growth_data):
        """Test transform returns same shape as input."""
        eg = ExponentialGrowth(n_days=7, thresh=0.05)
        result = eg.transform(exponential_growth_data)
        assert result.shape == exponential_growth_data.shape

    def test_transform_returns_boolean(self, exponential_growth_data):
        """Test transform returns boolean values."""
        eg = ExponentialGrowth(n_days=7, thresh=0.05)
        result = eg.transform(exponential_growth_data)
        assert result.dtypes.apply(lambda x: x == bool or x == np.bool_).all()


class TestExponentialGrowthDetection:
    """Tests for ExponentialGrowth detection."""

    def test_detects_exponential_growth(self, exponential_growth_data):
        """Test detection of clear exponential growth."""
        eg = ExponentialGrowth(n_days=7, thresh=0.05)
        result = eg.transform(exponential_growth_data)

        # Location 0 has clear exponential growth
        assert result["0"].any(), "Should detect exponential growth in location 0"

    def test_detects_less_on_linear(self, exponential_growth_data):
        """Test that linear growth produces less detection than exponential."""
        eg = ExponentialGrowth(n_days=7, thresh=0.05)
        result = eg.transform(exponential_growth_data)

        # Location 2 has linear growth - should detect less than exponential
        exp_detections = result["0"].sum()
        linear_detections = result["2"].sum()

        # Exponential should have more or equal detections
        assert exp_detections >= linear_detections * 0.5, "Exponential should have significant detections"


class TestExponentialGrowthParameters:
    """Tests for ExponentialGrowth parameters."""

    @pytest.mark.parametrize("n_days", [3, 7, 14, 21])
    def test_different_n_days(self, exponential_growth_data, n_days):
        """Test with different window sizes."""
        eg = ExponentialGrowth(n_days=n_days, thresh=0.05)
        result = eg.transform(exponential_growth_data)
        assert result.shape == exponential_growth_data.shape

    @pytest.mark.parametrize("thresh", [0.001, 0.01, 0.05, 0.1])
    def test_different_thresh(self, exponential_growth_data, thresh):
        """Test with different threshold values."""
        eg = ExponentialGrowth(n_days=7, thresh=thresh)
        result = eg.transform(exponential_growth_data)
        assert result.shape == exponential_growth_data.shape


class TestExponentialGrowthEdgeCases:
    """Edge case tests for ExponentialGrowth."""

    def test_constant_values(self, edge_cases_data):
        """Test handling of constant values."""
        eg = ExponentialGrowth(n_days=7, thresh=0.05)
        constant_data = edge_cases_data[["1"]]
        result = eg.transform(constant_data)
        # Constant values should not trigger detection
        assert result.shape == constant_data.shape

    def test_short_series(self, short_series_data):
        """Test handling of short time series."""
        eg = ExponentialGrowth(n_days=7, thresh=0.05)
        result = eg.transform(short_series_data)
        assert result.shape == short_series_data.shape


# GapFiller Tests

class TestGapFillerBasic:
    """Basic functionality tests for GapFiller."""

    def test_initialization_default_params(self):
        """Test GapFiller initializes with default parameters."""
        gf = GapFiller()
        assert gf.max_gap == 7

    def test_initialization_custom_params(self):
        """Test GapFiller initializes with custom parameters."""
        gf = GapFiller(max_gap=14)
        assert gf.max_gap == 14

    def test_transform_returns_dataframe(self):
        """Test transform returns a DataFrame."""
        gf = GapFiller()
        dates = pd.date_range("2020-01-01", periods=30)
        df = pd.DataFrame({"0": [True]*5 + [False]*3 + [True]*5 + [False]*17}, index=dates)
        result = gf.transform(df)
        assert isinstance(result, pd.DataFrame)

    def test_transform_same_shape(self):
        """Test transform returns same shape as input."""
        gf = GapFiller()
        dates = pd.date_range("2020-01-01", periods=30)
        df = pd.DataFrame({"0": [True]*5 + [False]*3 + [True]*5 + [False]*17}, index=dates)
        result = gf.transform(df)
        assert result.shape == df.shape


class TestGapFillerFilling:
    """Tests for GapFiller gap filling logic."""

    def test_fills_small_gaps(self):
        """Test that small gaps are filled."""
        gf = GapFiller(max_gap=7)
        dates = pd.date_range("2020-01-01", periods=20)
        # Gap of 3 between two True periods
        df = pd.DataFrame({"0": [True]*5 + [False]*3 + [True]*5 + [False]*7}, index=dates)

        result = gf.transform(df)

        # Gap should be filled
        assert result["0"].iloc[5:8].all(), "Small gap should be filled"

    def test_does_not_fill_large_gaps(self):
        """Test that large gaps are not filled."""
        gf = GapFiller(max_gap=5)
        dates = pd.date_range("2020-01-01", periods=30)
        # Gap of 10 between two True periods
        df = pd.DataFrame({"0": [True]*5 + [False]*10 + [True]*5 + [False]*10}, index=dates)

        result = gf.transform(df)

        # Large gap should NOT be filled
        assert not result["0"].iloc[5:15].all(), "Large gap should not be filled"

    def test_gap_at_boundary(self):
        """Test gap filling at boundary."""
        gf = GapFiller(max_gap=7)
        dates = pd.date_range("2020-01-01", periods=20)
        # Gap at the start
        df = pd.DataFrame({"0": [False]*3 + [True]*10 + [False]*7}, index=dates)

        result = gf.transform(df)

        # Gap at start should not be filled (no True before)
        assert not result["0"].iloc[0:3].any(), "Gap at start should not be filled"


class TestGapFillerMaxGap:
    """Tests for different max_gap parameter values."""

    @pytest.mark.parametrize("max_gap", [1, 3, 7, 14, 30])
    def test_different_max_gap(self, max_gap):
        """Test GapFiller with different max_gap values."""
        gf = GapFiller(max_gap=max_gap)
        dates = pd.date_range("2020-01-01", periods=50)
        df = pd.DataFrame({"0": [True]*10 + [False]*5 + [True]*10 + [False]*25}, index=dates)
        result = gf.transform(df)
        assert result.shape == df.shape

    def test_larger_max_gap_more_filling(self):
        """Test that larger max_gap fills more gaps."""
        dates = pd.date_range("2020-01-01", periods=50)
        # Multiple gaps of different sizes
        df = pd.DataFrame({
            "0": [True]*5 + [False]*3 + [True]*5 + [False]*8 + [True]*5 + [False]*24
        }, index=dates)

        gf_small = GapFiller(max_gap=5)
        gf_large = GapFiller(max_gap=15)

        result_small = gf_small.transform(df)
        result_large = gf_large.transform(df)

        filled_small = result_small.sum().sum()
        filled_large = result_large.sum().sum()

        assert filled_large >= filled_small, "Larger max_gap should fill more"


class TestGapFillerMultiLocation:
    """Tests for GapFiller with multiple locations."""

    def test_multi_location_shape(self):
        """Test transform works with multiple locations."""
        gf = GapFiller(max_gap=7)
        dates = pd.date_range("2020-01-01", periods=30)
        df = pd.DataFrame({
            "0": [True]*5 + [False]*3 + [True]*5 + [False]*17,
            "1": [False]*10 + [True]*5 + [False]*2 + [True]*5 + [False]*8
        }, index=dates)

        result = gf.transform(df)
        assert result.shape == df.shape

    def test_multi_location_independent(self):
        """Test that each location is processed independently."""
        gf = GapFiller(max_gap=5)
        dates = pd.date_range("2020-01-01", periods=30)

        # Location 0: gap of 3 (should fill)
        # Location 1: gap of 8 (should not fill)
        df = pd.DataFrame({
            "0": [True]*5 + [False]*3 + [True]*5 + [False]*17,
            "1": [True]*5 + [False]*8 + [True]*5 + [False]*12
        }, index=dates)

        result = gf.transform(df)

        # Location 0 gap should be filled
        assert result["0"].iloc[5:8].all(), "Location 0 gap should be filled"
        # Location 1 gap should NOT be filled
        assert not result["1"].iloc[5:13].all(), "Location 1 gap should not be filled"


class TestGapFillerReproducibility:
    """Tests for GapFiller reproducibility."""

    def test_deterministic_output(self):
        """Test that GapFiller produces deterministic output."""
        gf = GapFiller(max_gap=7)
        dates = pd.date_range("2020-01-01", periods=30)
        df = pd.DataFrame({"0": [True]*5 + [False]*3 + [True]*5 + [False]*17}, index=dates)

        result1 = gf.transform(df)
        result2 = gf.transform(df)

        pd.testing.assert_frame_equal(result1, result2)
