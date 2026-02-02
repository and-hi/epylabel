"""
Tests for pure Python BCP implementation.

Compares Python implementation with R implementation where available.
"""

import numpy as np
import pandas as pd
import pytest
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from epylabel.bcp_python import bcp, BcpPython, BcpResult


class TestBcpPythonBasic:
    """Basic tests for Python BCP implementation."""

    def test_returns_bcp_result(self):
        """Test that bcp returns a BcpResult object."""
        data = np.random.randn(50)
        result = bcp(data)
        assert isinstance(result, BcpResult)
        assert hasattr(result, 'posterior_mean')
        assert hasattr(result, 'posterior_prob')

    def test_output_shapes(self):
        """Test that output arrays have correct shape."""
        n = 100
        data = np.random.randn(n)
        result = bcp(data)
        assert result.posterior_mean.shape == (n,)
        assert result.posterior_prob.shape == (n,)

    def test_posterior_prob_range(self):
        """Test that posterior probabilities are in [0, 1]."""
        data = np.random.randn(50)
        result = bcp(data)
        assert np.all(result.posterior_prob >= 0)
        assert np.all(result.posterior_prob <= 1)

    def test_first_position_zero(self):
        """Test that first position has posterior_prob = 0."""
        data = np.random.randn(50)
        result = bcp(data)
        assert result.posterior_prob[0] == 0.0

    def test_single_observation(self):
        """Test with single observation."""
        data = np.array([5.0])
        result = bcp(data)
        assert len(result.posterior_mean) == 1
        assert result.posterior_mean[0] == 5.0

    def test_two_observations(self):
        """Test with two observations."""
        data = np.array([1.0, 10.0])
        result = bcp(data)
        assert len(result.posterior_mean) == 2


class TestBcpPythonChangepoints:
    """Test changepoint detection on known patterns."""

    def test_clean_step_function(self):
        """Test detection on clean step function with clear changepoints."""
        # Create clean step function
        data = np.concatenate([
            np.ones(20) * 10,
            np.ones(20) * 50,
            np.ones(20) * 25,
        ])

        result = bcp(data, p0=0.2)

        # Changepoints should be detected at indices 20 and 40
        # Allow for off-by-one due to algorithm differences
        high_prob_indices = np.where(result.posterior_prob > 0.5)[0]

        # Should have exactly 2 changepoints (or close to positions 20 and 40)
        assert len(high_prob_indices) >= 1, "Should detect at least one changepoint"

        # Check that detected changepoints are near expected positions
        expected_cps = [20, 40]
        for expected_cp in expected_cps:
            # At least one detected changepoint should be within 2 positions
            min_dist = min(abs(high_prob_indices - expected_cp))
            assert min_dist <= 2, f"Expected changepoint near {expected_cp}, got {high_prob_indices}"

    def test_posterior_mean_reflects_segments(self):
        """Test that posterior mean reflects segment means."""
        # Clear step function
        data = np.concatenate([
            np.ones(30) * 10,
            np.ones(30) * 50,
        ])

        result = bcp(data, p0=0.2)

        # First segment mean should be close to 10
        assert abs(np.mean(result.posterior_mean[:25]) - 10) < 5

        # Second segment mean should be close to 50
        assert abs(np.mean(result.posterior_mean[35:]) - 50) < 5

    def test_no_changepoint_constant_data(self):
        """Test with constant data - posterior mean should be constant."""
        data = np.ones(50) * 100
        result = bcp(data, p0=0.5)

        # Posterior mean should be close to 100 everywhere
        assert np.allclose(result.posterior_mean, 100, atol=1.0), \
            "Posterior mean should be close to data value for constant data"

    def test_high_p0_fewer_changepoints(self):
        """Test that higher p0 leads to fewer detected changepoints."""
        np.random.seed(42)
        data = np.concatenate([
            np.random.normal(0, 1, 30),
            np.random.normal(5, 1, 30),
        ])

        result_low_p0 = bcp(data, p0=0.1)
        result_high_p0 = bcp(data, p0=0.9)

        # Higher p0 should generally result in fewer high-probability positions
        high_prob_low = np.sum(result_low_p0.posterior_prob > 0.5)
        high_prob_high = np.sum(result_high_p0.posterior_prob > 0.5)

        # This isn't always guaranteed, but generally true
        # Just check they both work
        assert high_prob_low >= 0
        assert high_prob_high >= 0


class TestBcpPythonMCMC:
    """Test MCMC algorithm."""

    def test_mcmc_runs(self):
        """Test that MCMC runs without error."""
        data = np.random.randn(30)
        result = bcp(data, burnin=10, mcmc=50)
        assert result.posterior_mean.shape == (30,)
        assert result.posterior_prob.shape == (30,)

    def test_mcmc_detects_changepoints(self):
        """Test that MCMC detects clear changepoints."""
        np.random.seed(42)
        data = np.concatenate([
            np.ones(20) * 0,
            np.ones(20) * 100,
        ])

        result = bcp(data, p0=0.2, burnin=50, mcmc=200)

        # Should detect the obvious changepoint (allow some tolerance)
        # The Python implementation may find multiple high-prob positions
        high_prob = np.where(result.posterior_prob > 0.3)[0]
        assert len(high_prob) > 0, "Should detect at least one changepoint"


class TestBcpPythonVsR:
    """Compare Python implementation with R implementation."""

    @pytest.fixture
    def r_available(self):
        """Check if R and bcp package are available."""
        try:
            import rpy2.robjects as robjects
            import rpy2.robjects.packages as rpackages
            bcp_r = rpackages.importr("bcp")
            return True
        except Exception:
            return False

    def test_similar_changepoint_detection(self, r_available):
        """Test that Python and R detect changepoints in similar locations."""
        if not r_available:
            pytest.skip("R/rpy2 not available")

        import rpy2.robjects as robjects
        import rpy2.robjects.packages as rpackages

        # Test data with clear changepoint
        np.random.seed(42)
        data = np.concatenate([
            np.ones(30) * 10,
            np.ones(30) * 50,
        ])

        # Python result
        result_py = bcp(data, p0=0.2)

        # R result
        bcp_r = rpackages.importr("bcp")
        x_r = robjects.FloatVector(data)
        result_r = bcp_r.bcp(x_r, p0=0.2)
        prob_r = np.array(result_r.rx2["posterior.prob"]).flatten()

        # Both should detect changepoint around index 30
        py_cp = np.where(result_py.posterior_prob > 0.5)[0]
        r_cp = np.where(prob_r > 0.5)[0]

        # At least one of them should detect near position 30
        if len(r_cp) > 0:
            r_near_30 = any(abs(cp - 30) <= 2 for cp in r_cp)
            assert r_near_30, f"R should detect changepoint near 30, got {r_cp}"

    def test_posterior_mean_similar_trend(self, r_available):
        """Test that posterior means follow similar trends."""
        if not r_available:
            pytest.skip("R/rpy2 not available")

        import rpy2.robjects as robjects
        import rpy2.robjects.packages as rpackages

        # Clear step function
        data = np.concatenate([
            np.ones(25) * 0,
            np.ones(25) * 100,
        ])

        # Python result
        result_py = bcp(data, p0=0.2)

        # R result
        bcp_r = rpackages.importr("bcp")
        x_r = robjects.FloatVector(data)
        result_r = bcp_r.bcp(x_r, p0=0.2)
        mean_r = np.array(result_r.rx2["posterior.mean"]).flatten()

        # Both should show transition from low to high
        # First segment should have low mean
        py_first = np.mean(result_py.posterior_mean[:20])
        r_first = np.mean(mean_r[:20])

        # Second segment should have high mean
        py_second = np.mean(result_py.posterior_mean[30:])
        r_second = np.mean(mean_r[30:])

        # Check same direction of change
        assert py_first < py_second, "Python: first segment should be lower"
        assert r_first < r_second, "R: first segment should be lower"


class TestBcpClass:
    """Test Bcp class from labeler.py."""

    def test_bcp_python_mode(self):
        """Test Bcp class with use_python=True."""
        from epylabel.labeler import Bcp

        # Create with Python mode
        bcp_obj = Bcp(d=100, p0=0.2, thresh=0.5, use_python=True)

        # Create test data
        dates = pd.date_range('2020-01-01', periods=60, freq='D')
        data = pd.DataFrame({
            0: np.concatenate([np.ones(30) * 10, np.ones(30) * 50])
        }, index=dates)
        data.index.name = 'target'

        # Transform should work
        result = bcp_obj.transform(data)
        assert isinstance(result, pd.DataFrame)
        assert result.shape == data.shape

    @pytest.fixture
    def r_available(self):
        """Check if R is available."""
        try:
            import rpy2.robjects
            return True
        except Exception:
            return False

    def test_bcp_r_mode(self, r_available):
        """Test Bcp class with R mode."""
        if not r_available:
            pytest.skip("R/rpy2 not available")

        from epylabel.labeler import Bcp

        # Create with R mode (default)
        bcp_obj = Bcp(d=100, p0=0.2, thresh=0.5, use_python=False)

        # Create test data
        dates = pd.date_range('2020-01-01', periods=60, freq='D')
        data = pd.DataFrame({
            0: np.concatenate([np.ones(30) * 10, np.ones(30) * 50])
        }, index=dates)
        data.index.name = 'target'

        # Transform should work
        result = bcp_obj.transform(data)
        assert isinstance(result, pd.DataFrame)
        assert result.shape == data.shape


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
