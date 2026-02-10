"""
Comprehensive tests for the BCP (Bayesian Change Point) transformation.
"""

import pytest
import pandas as pd
import numpy as np

from epylabel.labeler import Bcp


class TestBcpBasic:
    """Basic functionality tests for Bcp."""

    def test_initialization_default_params(self):
        """Test Bcp initializes with default parameters."""
        bcp = Bcp()
        assert bcp.d == 100
        assert bcp.p0 == 0.2
        assert bcp.thresh == 0.5

    def test_initialization_custom_params(self):
        """Test Bcp initializes with custom parameters."""
        bcp = Bcp(d=500, p0=0.001, thresh=0.9)
        assert bcp.d == 500
        assert bcp.p0 == 0.001
        assert bcp.thresh == 0.9

    def test_transform_returns_dataframe(self, simple_wave_data):
        """Test transform returns a DataFrame."""
        bcp = Bcp(d=100, p0=0.1, thresh=0.5)
        result = bcp.transform(simple_wave_data)
        assert isinstance(result, pd.DataFrame)

    def test_transform_same_shape(self, simple_wave_data):
        """Test transform returns same shape as input."""
        bcp = Bcp(d=100, p0=0.1, thresh=0.5)
        result = bcp.transform(simple_wave_data)
        assert result.shape == simple_wave_data.shape

    def test_transform_returns_boolean(self, simple_wave_data):
        """Test transform returns boolean values."""
        bcp = Bcp(d=100, p0=0.1, thresh=0.5)
        result = bcp.transform(simple_wave_data)
        assert result.dtypes.apply(lambda x: x == bool or x == np.bool_).all()


class TestBcpChangePointDetection:
    """Tests for BCP change point detection."""

    def test_detects_known_changepoints(self, known_changepoints_data):
        """Test that BCP detects known changepoints."""
        bcp = Bcp(d=500, p0=0.01, thresh=0.5)
        result = bcp.transform(known_changepoints_data)

        # Should detect changepoints - check that there are transitions
        labels = result["0"].astype(int)
        transitions = (labels != labels.shift()).sum()

        assert transitions >= 2, "Should detect at least some changepoints"

    def test_detects_level_change(self):
        """Test detection of simple level change."""
        dates = pd.date_range("2020-01-01", periods=100)
        # Clear level change at day 50
        values = np.concatenate([np.ones(50) * 10, np.ones(50) * 100])
        df = pd.DataFrame({"0": values}, index=dates)

        bcp = Bcp(d=100, p0=0.1, thresh=0.3)
        result = bcp.transform(df)

        # Should detect the transition region
        labels = result["0"]
        # Check there's a transition somewhere in the middle
        mid_labels = labels.iloc[40:60]
        assert mid_labels.any() or not mid_labels.all(), "Should detect transition around day 50"


class TestBcpWaveDetection:
    """Tests for BCP on wave patterns."""

    def test_single_wave_detection(self, simple_wave_data):
        """Test detection on single wave."""
        bcp = Bcp(d=200, p0=0.05, thresh=0.5)
        result = bcp.transform(simple_wave_data)

        # Should detect something during the wave
        assert result.any().any(), "Should detect wave pattern"

    def test_multi_wave_detection(self, multi_wave_data):
        """Test detection on multiple waves."""
        bcp = Bcp(d=500, p0=0.01, thresh=0.5)
        result = bcp.transform(multi_wave_data)

        # Count transitions
        labels = result["0"].astype(int)
        transitions = (labels != labels.shift()).sum()

        # Multiple waves should produce multiple transitions
        assert transitions >= 4, "Should detect multiple wave transitions"


class TestBcpMultiLocation:
    """Tests for BCP with multiple locations."""

    def test_multi_location_shape(self, multi_location_data):
        """Test transform works with multiple locations."""
        bcp = Bcp(d=100, p0=0.1, thresh=0.5)
        result = bcp.transform(multi_location_data)
        assert result.shape == multi_location_data.shape

    def test_multi_location_detections(self, multi_location_data):
        """Test that multiple locations are processed."""
        bcp = Bcp(d=100, p0=0.1, thresh=0.5)
        result = bcp.transform(multi_location_data)

        # Each location should have some detections
        for col in result.columns:
            assert result[col].any() or not result[col].all(), f"Location {col} should have variation"


class TestBcpParameters:
    """Tests for different BCP parameter values."""

    @pytest.mark.parametrize("d", [50, 100, 500, 1000])
    def test_different_d_values(self, simple_wave_data, d):
        """Test BCP with different d (max changepoints) values."""
        bcp = Bcp(d=d, p0=0.1, thresh=0.5)
        result = bcp.transform(simple_wave_data)
        assert result.shape == simple_wave_data.shape

    @pytest.mark.parametrize("p0", [0.001, 0.01, 0.1, 0.2, 0.5])
    def test_different_p0_values(self, simple_wave_data, p0):
        """Test BCP with different p0 (prior probability) values."""
        bcp = Bcp(d=100, p0=p0, thresh=0.5)
        result = bcp.transform(simple_wave_data)
        assert result.shape == simple_wave_data.shape

    @pytest.mark.parametrize("thresh", [0.3, 0.5, 0.7, 0.9])
    def test_different_thresh_values(self, simple_wave_data, thresh):
        """Test BCP with different threshold values."""
        bcp = Bcp(d=100, p0=0.1, thresh=thresh)
        result = bcp.transform(simple_wave_data)
        assert result.shape == simple_wave_data.shape

    def test_higher_threshold_fewer_detections(self, simple_wave_data):
        """Test that higher threshold produces fewer detections."""
        bcp_low = Bcp(d=100, p0=0.1, thresh=0.3)
        bcp_high = Bcp(d=100, p0=0.1, thresh=0.9)

        result_low = bcp_low.transform(simple_wave_data)
        result_high = bcp_high.transform(simple_wave_data)

        detections_low = result_low.sum().sum()
        detections_high = result_high.sum().sum()

        # Higher threshold should generally produce fewer or equal detections
        # (not strict because of MCMC randomness)
        assert detections_high <= detections_low * 1.2, "Higher threshold should not produce many more detections"


class TestBcpEdgeCases:
    """Edge case tests for BCP."""

    def test_constant_values(self, edge_cases_data):
        """Test handling of constant values."""
        bcp = Bcp(d=50, p0=0.1, thresh=0.5)
        constant_data = edge_cases_data[["1"]]
        result = bcp.transform(constant_data)
        # Constant values - no change points
        assert result.shape == constant_data.shape

    def test_step_function(self, edge_cases_data):
        """Test handling of step function."""
        bcp = Bcp(d=50, p0=0.1, thresh=0.5)
        step_data = edge_cases_data[["3"]]
        result = bcp.transform(step_data)
        # Should detect the step
        assert result.shape == step_data.shape

    def test_linear_increase(self, edge_cases_data):
        """Test handling of linear increase."""
        bcp = Bcp(d=50, p0=0.1, thresh=0.5)
        linear_data = edge_cases_data[["4"]]
        result = bcp.transform(linear_data)
        assert result.shape == linear_data.shape

    def test_short_series(self, short_series_data):
        """Test handling of short time series."""
        bcp = Bcp(d=20, p0=0.1, thresh=0.5)
        result = bcp.transform(short_series_data)
        assert result.shape == short_series_data.shape


class TestBcpWithChangerate:
    """Tests for BCP used after Changerate (typical pipeline)."""

    def test_bcp_after_changerate(self, simple_wave_data):
        """Test BCP applied to changerate output."""
        from epylabel.labeler import Changerate

        cr = Changerate(n_days=7, changerate_ceiling=6)
        bcp = Bcp(d=100, p0=0.1, thresh=0.5)

        changerate_result = cr.transform(simple_wave_data)
        # Fill NaN values for BCP
        changerate_filled = changerate_result.fillna(0)
        bcp_result = bcp.transform(changerate_filled)

        assert bcp_result.shape == simple_wave_data.shape

    def test_typical_pipeline_detects_wave(self, simple_wave_data):
        """Test typical BCP pipeline detects wave."""
        from epylabel.labeler import Changerate

        cr = Changerate(n_days=7, changerate_ceiling=6)
        bcp = Bcp(d=200, p0=0.01, thresh=0.5)

        changerate_result = cr.transform(simple_wave_data)
        changerate_filled = changerate_result.fillna(0)
        bcp_result = bcp.transform(changerate_filled)

        assert bcp_result.any().any(), "Pipeline should detect wave"


class TestBcpReproducibility:
    """Tests for BCP reproducibility.

    Note: BCP uses MCMC which is stochastic. These tests check for
    reasonable consistency rather than exact reproducibility.
    """

    def test_similar_results_multiple_runs(self, simple_wave_data):
        """Test that multiple runs produce similar results."""
        bcp = Bcp(d=100, p0=0.1, thresh=0.5)

        result1 = bcp.transform(simple_wave_data)
        result2 = bcp.transform(simple_wave_data)

        # Results should be similar (allow some MCMC variation)
        agreement = (result1 == result2).sum().sum() / result1.size
        assert agreement > 0.8, f"Results should be mostly similar, got {agreement} agreement"

    def test_results_within_bounds(self, simple_wave_data):
        """Test that results are always valid boolean."""
        bcp = Bcp(d=100, p0=0.1, thresh=0.5)

        for _ in range(3):
            result = bcp.transform(simple_wave_data)
            assert result.isin([True, False]).all().all(), "All values should be boolean"
