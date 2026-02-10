"""
Comprehensive tests for the WaveFinder transformation.
"""

import pytest
import pandas as pd
import numpy as np

from epylabel.labeler import WaveFinder


class TestWaveFinderBasic:
    """Basic functionality tests for WaveFinder."""

    def test_initialization_default_params(self):
        """Test WaveFinder initializes with default parameters."""
        wf = WaveFinder()
        assert wf.abs_prominence_threshold == 5
        assert wf.prominence_height_threshold == 0.01
        assert wf.t_sep_a == 35

    def test_initialization_custom_params(self):
        """Test WaveFinder initializes with custom parameters."""
        wf = WaveFinder(
            abs_prominence_threshold=10,
            prominence_height_threshold=0.05,
            t_sep_a=50
        )
        assert wf.abs_prominence_threshold == 10
        assert wf.prominence_height_threshold == 0.05
        assert wf.t_sep_a == 50

    def test_transform_returns_dataframe(self, simple_wave_data):
        """Test transform returns a DataFrame."""
        wf = WaveFinder()
        result = wf.transform(simple_wave_data)
        assert isinstance(result, pd.DataFrame)

    def test_transform_same_shape(self, simple_wave_data):
        """Test transform returns same shape as input."""
        wf = WaveFinder()
        result = wf.transform(simple_wave_data)
        assert result.shape == simple_wave_data.shape

    def test_transform_returns_boolean(self, simple_wave_data):
        """Test transform returns boolean values."""
        wf = WaveFinder()
        result = wf.transform(simple_wave_data)
        assert result.dtypes.apply(lambda x: x == bool or x == np.bool_).all()


class TestWaveFinderDetection:
    """Tests for WaveFinder wave detection."""

    def test_detects_single_wave(self, simple_wave_data):
        """Test detection of single clear wave."""
        wf = WaveFinder(abs_prominence_threshold=5)
        result = wf.transform(simple_wave_data)

        # Should detect the wave
        assert result.any().any(), "Should detect the wave"

        # Should have contiguous labeled period
        labels = result["0"]
        assert labels.sum() > 10, "Wave should be labeled for multiple days"

    def test_detects_multiple_waves(self, multi_wave_data):
        """Test detection of multiple waves."""
        wf = WaveFinder(abs_prominence_threshold=5, t_sep_a=35)
        result = wf.transform(multi_wave_data)

        # Count distinct wave periods
        labels = result["0"].astype(int)
        changes = labels.diff().abs()
        n_transitions = changes.sum()

        # Multiple waves should produce multiple transitions
        assert n_transitions >= 4, f"Should detect multiple waves, got {n_transitions} transitions"

    def test_does_not_detect_noise(self, noisy_data):
        """Test that pure noise does not produce excessive detections."""
        wf = WaveFinder(abs_prominence_threshold=20)  # High threshold
        result = wf.transform(noisy_data)

        # Should not label too much as wave
        label_proportion = result.sum().sum() / result.size
        assert label_proportion < 0.5, f"Too many labels on noisy data: {label_proportion}"


class TestWaveFinderPeakTrough:
    """Tests for WaveFinder peak and trough detection."""

    def test_identifies_peak(self, simple_wave_data):
        """Test that peak is identified within wave."""
        wf = WaveFinder(abs_prominence_threshold=5)
        result = wf.transform(simple_wave_data)

        # The peak of the wave should be labeled
        peak_idx = simple_wave_data["0"].idxmax()
        # Check area around peak
        peak_region = result["0"].loc[peak_idx - pd.Timedelta(days=5):peak_idx + pd.Timedelta(days=5)]
        assert peak_region.any(), "Peak region should be labeled"


class TestWaveFinderMultiLocation:
    """Tests for WaveFinder with multiple locations."""

    def test_multi_location_shape(self, multi_location_data):
        """Test transform works with multiple locations."""
        wf = WaveFinder()
        result = wf.transform(multi_location_data)
        assert result.shape == multi_location_data.shape

    def test_multi_location_independent(self, multi_location_data):
        """Test that each location is processed independently."""
        wf = WaveFinder()
        result = wf.transform(multi_location_data)

        # Different locations should have different detection patterns
        # (waves are staggered in test data)
        detection_sums = {col: result[col].sum() for col in result.columns}

        # At least some locations should have different detection counts
        unique_counts = len(set(detection_sums.values()))
        assert unique_counts > 1, "Different locations should have different detections"


class TestWaveFinderParameters:
    """Tests for different WaveFinder parameter values."""

    @pytest.mark.parametrize("threshold", [1, 5, 10, 20, 50])
    def test_different_prominence_thresholds(self, simple_wave_data, threshold):
        """Test WaveFinder with different prominence thresholds."""
        wf = WaveFinder(abs_prominence_threshold=threshold)
        result = wf.transform(simple_wave_data)
        assert result.shape == simple_wave_data.shape

    @pytest.mark.parametrize("t_sep", [14, 35, 50, 100])
    def test_different_t_sep_values(self, simple_wave_data, t_sep):
        """Test WaveFinder with different t_sep_a values."""
        wf = WaveFinder(t_sep_a=t_sep)
        result = wf.transform(simple_wave_data)
        assert result.shape == simple_wave_data.shape

    def test_higher_threshold_fewer_detections(self, simple_wave_data):
        """Test that higher prominence threshold produces fewer detections."""
        wf_low = WaveFinder(abs_prominence_threshold=1)
        wf_high = WaveFinder(abs_prominence_threshold=50)

        result_low = wf_low.transform(simple_wave_data)
        result_high = wf_high.transform(simple_wave_data)

        detections_low = result_low.sum().sum()
        detections_high = result_high.sum().sum()

        assert detections_high <= detections_low, "Higher threshold should produce fewer detections"


class TestWaveFinderEdgeCases:
    """Edge case tests for WaveFinder."""

    def test_constant_values(self, edge_cases_data):
        """Test handling of constant values."""
        wf = WaveFinder()
        constant_data = edge_cases_data[["1"]]
        result = wf.transform(constant_data)
        # Constant values should not produce wave detections
        assert result.shape == constant_data.shape

    def test_single_spike(self, edge_cases_data):
        """Test handling of single spike."""
        wf = WaveFinder(abs_prominence_threshold=100)
        spike_data = edge_cases_data[["2"]]
        result = wf.transform(spike_data)
        # Single spike might be detected as small wave
        assert result.shape == spike_data.shape

    def test_linear_increase(self, edge_cases_data):
        """Test handling of linear increase."""
        wf = WaveFinder()
        linear_data = edge_cases_data[["4"]]
        result = wf.transform(linear_data)
        # Linear increase is not a wave pattern
        assert result.shape == linear_data.shape

    def test_exponential_growth(self, edge_cases_data):
        """Test handling of exponential growth without decline."""
        wf = WaveFinder()
        exp_data = edge_cases_data[["5"]]
        result = wf.transform(exp_data)
        assert result.shape == exp_data.shape

    def test_short_series(self, short_series_data):
        """Test handling of short time series."""
        wf = WaveFinder(t_sep_a=10)  # Shorter separation for short series
        result = wf.transform(short_series_data)
        assert result.shape == short_series_data.shape


class TestWaveFinderCovidLike:
    """Tests for WaveFinder on COVID-like patterns."""

    def test_covid_sample_detection(self, real_covid_sample_data):
        """Test detection on COVID-like sample data."""
        wf = WaveFinder(abs_prominence_threshold=50, t_sep_a=35)
        result = wf.transform(real_covid_sample_data)

        # Should detect multiple waves in COVID-like data
        labels = result["0"].astype(int)
        transitions = (labels != labels.shift()).sum()

        assert transitions >= 6, f"Should detect multiple COVID waves, got {transitions} transitions"

    def test_covid_sample_coverage(self, real_covid_sample_data):
        """Test that COVID-like data has reasonable outbreak coverage."""
        wf = WaveFinder(abs_prominence_threshold=30, t_sep_a=35)
        result = wf.transform(real_covid_sample_data)

        # COVID data should have significant outbreak periods
        label_proportion = result.sum().sum() / result.size
        assert 0.2 < label_proportion < 0.8, f"Unexpected outbreak coverage: {label_proportion}"


class TestWaveFinderReproducibility:
    """Tests for WaveFinder reproducibility."""

    def test_deterministic_output(self, simple_wave_data):
        """Test that WaveFinder produces deterministic output."""
        wf = WaveFinder()

        result1 = wf.transform(simple_wave_data)
        result2 = wf.transform(simple_wave_data)

        pd.testing.assert_frame_equal(result1, result2)

    def test_same_results_new_instance(self, simple_wave_data):
        """Test that new instance produces same results."""
        wf1 = WaveFinder(abs_prominence_threshold=5, t_sep_a=35)
        wf2 = WaveFinder(abs_prominence_threshold=5, t_sep_a=35)

        result1 = wf1.transform(simple_wave_data)
        result2 = wf2.transform(simple_wave_data)

        pd.testing.assert_frame_equal(result1, result2)
