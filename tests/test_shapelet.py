"""
Comprehensive tests for the Shapelet transformation.
"""

import pytest
import pandas as pd
import numpy as np

from epylabel.labeler import Shapelet


class TestShapeletBasic:
    """Basic functionality tests for Shapelet."""

    def test_initialization_default_params(self):
        """Test Shapelet initializes with default parameters."""
        sp = Shapelet()
        assert sp.n_days == 7
        assert sp.x_max == 3
        assert sp.thresh == 0.8

    def test_initialization_custom_params(self):
        """Test Shapelet initializes with custom parameters."""
        sp = Shapelet(n_days=14, x_max=5, thresh=0.9)
        assert sp.n_days == 14
        assert sp.x_max == 5
        assert sp.thresh == 0.9

    def test_transform_returns_dataframe(self, simple_wave_data):
        """Test transform returns a DataFrame."""
        sp = Shapelet()
        result = sp.transform(simple_wave_data)
        assert isinstance(result, pd.DataFrame)

    def test_transform_same_shape(self, simple_wave_data):
        """Test transform returns same shape as input."""
        sp = Shapelet()
        result = sp.transform(simple_wave_data)
        assert result.shape == simple_wave_data.shape

    def test_transform_returns_boolean(self, simple_wave_data):
        """Test transform returns boolean values."""
        sp = Shapelet()
        result = sp.transform(simple_wave_data)
        assert result.dtypes.apply(lambda x: x == bool or x == np.bool_).all()


class TestShapeletDetection:
    """Tests for Shapelet outbreak detection."""

    def test_detects_exponential_growth(self, shapelet_test_data):
        """Test that Shapelet detects exponential growth pattern."""
        sp = Shapelet(n_days=7, x_max=3, thresh=0.7)
        result = sp.transform(shapelet_test_data)

        # Location 0 has clear exponential surge - should be detected
        assert result["0"].any(), "Should detect exponential growth in location 0"

    def test_no_detection_on_decline(self, shapelet_test_data):
        """Test that Shapelet does not detect declining pattern."""
        sp = Shapelet(n_days=7, x_max=3, thresh=0.8)
        result = sp.transform(shapelet_test_data)

        # Location 1 has declining pattern - should not trigger
        # (or trigger much less than location 0)
        detected_0 = result["0"].sum()
        detected_1 = result["1"].sum()
        assert detected_0 > detected_1, "Should detect more in growth than decline"

    def test_no_detection_on_noise(self, shapelet_test_data):
        """Test that Shapelet does not detect pure noise."""
        sp = Shapelet(n_days=7, x_max=3, thresh=0.8)
        result = sp.transform(shapelet_test_data)

        # Location 3 is pure noise - should have minimal detection
        noise_detection_rate = result["3"].mean()
        assert noise_detection_rate < 0.3, f"Too many false positives on noise: {noise_detection_rate}"


class TestShapeletWaveDetection:
    """Tests for Shapelet on wave patterns."""

    def test_single_wave_detection(self, simple_wave_data):
        """Test detection of single wave."""
        sp = Shapelet(n_days=7, x_max=3, thresh=0.7)
        result = sp.transform(simple_wave_data)

        # Should detect something during the growth phase
        assert result.any().any(), "Should detect growth phase of wave"

    def test_multi_wave_detection(self, multi_wave_data):
        """Test detection of multiple waves."""
        sp = Shapelet(n_days=7, x_max=3, thresh=0.7)
        result = sp.transform(multi_wave_data)

        # Should detect multiple periods
        # Count distinct outbreak periods
        labels = result["0"].astype(int)
        changes = (labels != labels.shift()).cumsum()
        n_periods = labels.groupby(changes).first().sum()

        assert n_periods >= 2, "Should detect at least 2 outbreak periods"


class TestShapeletMultiLocation:
    """Tests for Shapelet with multiple locations."""

    def test_multi_location_shape(self, multi_location_data):
        """Test transform works with multiple locations."""
        sp = Shapelet()
        result = sp.transform(multi_location_data)
        assert result.shape == multi_location_data.shape

    def test_multi_location_independent(self, multi_location_data):
        """Test that each location is processed independently."""
        sp = Shapelet()
        result = sp.transform(multi_location_data)

        # Each location should have different detection timing
        # (since waves are staggered in multi_location_data)
        first_detection = {}
        for col in result.columns:
            detected = result[col]
            if detected.any():
                first_detection[col] = detected.idxmax()

        # At least some locations should have different first detection times
        unique_times = len(set(first_detection.values()))
        assert unique_times > 1, "Different locations should detect at different times"


class TestShapeletThreshold:
    """Tests for different threshold parameter values."""

    @pytest.mark.parametrize("thresh", [0.5, 0.7, 0.8, 0.9, 0.95])
    def test_different_thresholds(self, simple_wave_data, thresh):
        """Test Shapelet with different threshold values."""
        sp = Shapelet(n_days=7, x_max=3, thresh=thresh)
        result = sp.transform(simple_wave_data)
        assert result.shape == simple_wave_data.shape

    def test_higher_threshold_fewer_detections(self, simple_wave_data):
        """Test that higher threshold produces fewer detections."""
        sp_low = Shapelet(n_days=7, x_max=3, thresh=0.5)
        sp_high = Shapelet(n_days=7, x_max=3, thresh=0.95)

        result_low = sp_low.transform(simple_wave_data)
        result_high = sp_high.transform(simple_wave_data)

        detections_low = result_low.sum().sum()
        detections_high = result_high.sum().sum()

        assert detections_high <= detections_low, "Higher threshold should produce fewer detections"


class TestShapeletNDays:
    """Tests for different n_days parameter values."""

    @pytest.mark.parametrize("n_days", [3, 7, 14, 21])
    def test_different_n_days(self, simple_wave_data, n_days):
        """Test Shapelet with different window sizes."""
        sp = Shapelet(n_days=n_days, x_max=3, thresh=0.7)
        result = sp.transform(simple_wave_data)
        assert result.shape == simple_wave_data.shape


class TestShapeletEdgeCases:
    """Edge case tests for Shapelet."""

    def test_all_zeros(self, edge_cases_data):
        """Test handling of all-zero data."""
        sp = Shapelet()
        zero_data = edge_cases_data[["0"]]
        result = sp.transform(zero_data)
        # All zeros should not produce detections
        assert not result.any().any(), "All zeros should not trigger detection"

    def test_constant_values(self, edge_cases_data):
        """Test handling of constant values."""
        sp = Shapelet()
        constant_data = edge_cases_data[["1"]]
        result = sp.transform(constant_data)
        # Constant values should not produce detections
        assert not result.any().any(), "Constant values should not trigger detection"

    def test_single_spike(self, edge_cases_data):
        """Test handling of single spike."""
        sp = Shapelet(n_days=7, thresh=0.5)
        spike_data = edge_cases_data[["2"]]
        result = sp.transform(spike_data)
        # Single spike might or might not trigger - just check no error
        assert result.shape == spike_data.shape

    def test_short_series(self, short_series_data):
        """Test handling of short time series."""
        sp = Shapelet(n_days=7)
        result = sp.transform(short_series_data)
        assert result.shape == short_series_data.shape


class TestShapeletReproducibility:
    """Tests for Shapelet reproducibility."""

    def test_deterministic_output(self, simple_wave_data):
        """Test that Shapelet produces deterministic output."""
        sp = Shapelet()

        result1 = sp.transform(simple_wave_data)
        result2 = sp.transform(simple_wave_data)

        pd.testing.assert_frame_equal(result1, result2)

    def test_same_results_new_instance(self, simple_wave_data):
        """Test that new instance produces same results."""
        sp1 = Shapelet(n_days=7, x_max=3, thresh=0.8)
        sp2 = Shapelet(n_days=7, x_max=3, thresh=0.8)

        result1 = sp1.transform(simple_wave_data)
        result2 = sp2.transform(simple_wave_data)

        pd.testing.assert_frame_equal(result1, result2)
