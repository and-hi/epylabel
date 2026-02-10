"""
Comprehensive tests for the Pipeline class.
"""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path

from epylabel.pipeline import Pipeline
from epylabel.labeler import Changerate, Shapelet, WaveFinder, Ensemble, GapFiller


class TestPipelineBasic:
    """Basic functionality tests for Pipeline."""

    def test_pipeline_single_step(self, simple_wave_data):
        """Test pipeline with single transformation."""
        cr = Changerate(n_days=7)
        pipeline = Pipeline([cr])

        result = pipeline.transform(simple_wave_data)

        # Should be same as running transformation directly
        expected = cr.transform(simple_wave_data)
        pd.testing.assert_frame_equal(result, expected)

    def test_pipeline_two_steps(self, simple_wave_data):
        """Test pipeline with two transformations."""
        cr = Changerate(n_days=7, changerate_ceiling=6)
        sp = Shapelet(n_days=7, thresh=0.5)

        pipeline = Pipeline([cr, sp])
        result = pipeline.transform(simple_wave_data)

        # Should be same as running sequentially
        intermediate = cr.transform(simple_wave_data)
        expected = sp.transform(intermediate)
        pd.testing.assert_frame_equal(result, expected)

    def test_pipeline_returns_dataframe(self, simple_wave_data):
        """Test that pipeline returns DataFrame."""
        pipeline = Pipeline([Changerate()])
        result = pipeline.transform(simple_wave_data)
        assert isinstance(result, pd.DataFrame)


class TestPipelineChaining:
    """Tests for pipeline chaining behavior."""

    def test_changerate_to_shapelet(self, simple_wave_data):
        """Test Changerate -> Shapelet pipeline."""
        pipeline = Pipeline([
            Changerate(n_days=7, changerate_ceiling=6),
            Shapelet(n_days=7, thresh=0.5)
        ])

        result = pipeline.transform(simple_wave_data)
        assert result.shape == simple_wave_data.shape
        assert result.dtypes.apply(lambda x: x == bool or x == np.bool_).all()

    def test_shapelet_to_gapfiller(self, simple_wave_data):
        """Test Shapelet -> GapFiller pipeline."""
        pipeline = Pipeline([
            Shapelet(n_days=7, thresh=0.6),
            GapFiller(max_gap=5)
        ])

        result = pipeline.transform(simple_wave_data)
        assert result.shape == simple_wave_data.shape

    def test_wavefinder_to_gapfiller(self, simple_wave_data):
        """Test WaveFinder -> GapFiller pipeline."""
        pipeline = Pipeline([
            WaveFinder(abs_prominence_threshold=5),
            GapFiller(max_gap=7)
        ])

        result = pipeline.transform(simple_wave_data)
        assert result.shape == simple_wave_data.shape


class TestPipelineMultiLocation:
    """Tests for pipeline with multiple locations."""

    def test_pipeline_multi_location(self, multi_location_data):
        """Test pipeline works with multiple locations."""
        pipeline = Pipeline([
            Changerate(n_days=7),
            Shapelet(n_days=7, thresh=0.5)
        ])

        result = pipeline.transform(multi_location_data)
        assert result.shape == multi_location_data.shape

    def test_pipeline_preserves_columns(self, multi_location_data):
        """Test pipeline preserves column names."""
        pipeline = Pipeline([Changerate()])
        result = pipeline.transform(multi_location_data)
        pd.testing.assert_index_equal(result.columns, multi_location_data.columns)


class TestPipelineEdgeCases:
    """Edge case tests for Pipeline."""

    def test_empty_pipeline(self, simple_wave_data):
        """Test empty pipeline raises error or returns input."""
        # Depending on implementation, this might raise or return input
        try:
            pipeline = Pipeline([])
            result = pipeline.transform(simple_wave_data)
            # If it doesn't raise, should return input unchanged
            pd.testing.assert_frame_equal(result, simple_wave_data)
        except (ValueError, IndexError):
            pass  # Expected behavior

    def test_pipeline_with_short_data(self, short_series_data):
        """Test pipeline with short time series."""
        pipeline = Pipeline([
            Changerate(n_days=7),
            Shapelet(n_days=7, thresh=0.5)
        ])

        result = pipeline.transform(short_series_data)
        assert result.shape == short_series_data.shape


class TestPipelineReproducibility:
    """Tests for pipeline reproducibility."""

    def test_pipeline_deterministic(self, simple_wave_data):
        """Test that pipeline produces deterministic results."""
        pipeline = Pipeline([
            Changerate(n_days=7),
            Shapelet(n_days=7, thresh=0.7)
        ])

        result1 = pipeline.transform(simple_wave_data)
        result2 = pipeline.transform(simple_wave_data)

        pd.testing.assert_frame_equal(result1, result2)

    def test_new_pipeline_same_results(self, simple_wave_data):
        """Test that new pipeline instance produces same results."""
        pipeline1 = Pipeline([
            Changerate(n_days=7, changerate_ceiling=6),
            Shapelet(n_days=7, x_max=3, thresh=0.8)
        ])
        pipeline2 = Pipeline([
            Changerate(n_days=7, changerate_ceiling=6),
            Shapelet(n_days=7, x_max=3, thresh=0.8)
        ])

        result1 = pipeline1.transform(simple_wave_data)
        result2 = pipeline2.transform(simple_wave_data)

        pd.testing.assert_frame_equal(result1, result2)


class TestTypicalPipelines:
    """Tests for typical pipeline configurations used in the paper."""

    def test_bcp_pipeline(self, simple_wave_data):
        """Test typical BCP pipeline (Changerate -> BCP)."""
        from epylabel.labeler import Bcp

        pipeline = Pipeline([
            Changerate(n_days=7, changerate_ceiling=6),
        ])

        # Get changerate result
        cr_result = pipeline.transform(simple_wave_data)

        # BCP needs filled values
        bcp = Bcp(d=100, p0=0.1, thresh=0.5)
        bcp_result = bcp.transform(cr_result.fillna(0))

        assert bcp_result.shape == simple_wave_data.shape
        assert bcp_result.dtypes.apply(lambda x: x == bool or x == np.bool_).all()

    def test_full_ensemble_pipeline(self, simple_wave_data):
        """Test full pipeline as used in paper."""
        from epylabel.labeler import Bcp

        # Individual pipelines
        cr = Changerate(n_days=7, changerate_ceiling=6)
        bcp = Bcp(d=100, p0=0.1, thresh=0.5)
        sp = Shapelet(n_days=7, x_max=3, thresh=0.8)
        wf = WaveFinder(abs_prominence_threshold=5, t_sep_a=35)
        ens = Ensemble(n_min=2)

        # Run individual algorithms
        cr_result = cr.transform(simple_wave_data)
        bcp_labels = bcp.transform(cr_result.fillna(0))
        sp_labels = sp.transform(simple_wave_data)
        wf_labels = wf.transform(simple_wave_data)

        # Ensemble
        ensemble_result = ens.transform(bcp_labels, sp_labels, wf_labels)

        assert ensemble_result.shape == simple_wave_data.shape
        assert ensemble_result.dtypes.apply(lambda x: x == bool or x == np.bool_).all()
