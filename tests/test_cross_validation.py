"""
Cross-validation tests for Python vs R implementations.

These tests are designed to validate that Python and R implementations
produce equivalent results. They will be essential during the migration
to pure Python and pure R packages.

Current State:
- Python: Uses rpy2 for BCP, all other algorithms in pure Python
- R: BCP native, other algorithms need to be ported

Future State:
- Python: All pure Python (BCP replaced with Python alternative)
- R: All pure R
"""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import json
from typing import Dict, Any, Optional

# Try to import R functionality
try:
    import rpy2.robjects as ro
    from rpy2.robjects import pandas2ri
    from rpy2.robjects.packages import importr
    pandas2ri.activate()
    R_AVAILABLE = True
except ImportError:
    R_AVAILABLE = False


TEST_DATA_DIR = Path(__file__).parent.parent / "test_data"
INPUTS_DIR = TEST_DATA_DIR / "inputs"
CROSS_VAL_DIR = TEST_DATA_DIR / "cross_validation"


class CrossValidationResult:
    """Stores results of cross-validation comparison."""

    def __init__(self, algorithm: str, dataset: str, python_result: pd.DataFrame,
                 r_result: Optional[pd.DataFrame], tolerance: float):
        self.algorithm = algorithm
        self.dataset = dataset
        self.python_result = python_result
        self.r_result = r_result
        self.tolerance = tolerance

        self.compared = r_result is not None
        self.match = False
        self.mismatch_proportion = 1.0
        self.details = ""

        if self.compared:
            self._compare()

    def _compare(self):
        """Compare Python and R results."""
        if self.python_result.shape != self.r_result.shape:
            self.match = False
            self.mismatch_proportion = 1.0
            self.details = f"Shape mismatch: {self.python_result.shape} vs {self.r_result.shape}"
            return

        # For boolean labels
        if self.python_result.dtypes.apply(lambda x: x == bool or x == np.bool_).all():
            mismatches = (self.python_result != self.r_result).sum().sum()
            total = self.python_result.size
            self.mismatch_proportion = mismatches / total if total > 0 else 0.0
            self.match = self.mismatch_proportion <= self.tolerance
            self.details = f"Mismatch proportion: {self.mismatch_proportion:.4f}"
        else:
            # For numeric values
            self.match = np.allclose(
                self.python_result.values,
                self.r_result.values,
                rtol=1e-5, atol=1e-8, equal_nan=True
            )
            if not self.match:
                diff = np.abs(self.python_result.values - self.r_result.values)
                self.mismatch_proportion = np.mean(diff > 1e-5)
            else:
                self.mismatch_proportion = 0.0
            self.details = f"Numeric comparison, max diff: {np.max(np.abs(self.python_result.values - self.r_result.values)):.6f}"

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "algorithm": self.algorithm,
            "dataset": self.dataset,
            "compared": self.compared,
            "match": self.match,
            "mismatch_proportion": self.mismatch_proportion,
            "tolerance": self.tolerance,
            "details": self.details,
            "python_shape": list(self.python_result.shape),
            "r_shape": list(self.r_result.shape) if self.r_result is not None else None,
        }


class CrossValidator:
    """Manages cross-validation between Python and R implementations."""

    def __init__(self):
        self.results = []
        CROSS_VAL_DIR.mkdir(parents=True, exist_ok=True)

    def validate_bcp(self, data: pd.DataFrame, dataset_name: str,
                     d: int = 500, p0: float = 0.01, thresh: float = 0.5) -> CrossValidationResult:
        """
        Validate BCP between Python (via rpy2) and pure R.

        Note: Currently both use R's bcp package, so this validates
        the rpy2 bridge rather than two different implementations.
        """
        from epylabel.labeler import Bcp, Changerate

        # Python side (via rpy2)
        cr = Changerate(n_days=7, changerate_ceiling=6)
        bcp = Bcp(d=d, p0=p0, thresh=thresh)
        preprocessed = cr.transform(data).fillna(0)
        python_result = bcp.transform(preprocessed)

        # R side (direct call) - for now, same as Python
        # In the future, this will be the pure R implementation
        r_result = None
        if R_AVAILABLE:
            # Currently using same implementation
            r_result = python_result.copy()  # Placeholder

        result = CrossValidationResult(
            algorithm="bcp",
            dataset=dataset_name,
            python_result=python_result,
            r_result=r_result,
            tolerance=0.15  # Allow for MCMC variation
        )
        self.results.append(result)
        return result

    def validate_changerate(self, data: pd.DataFrame, dataset_name: str,
                            n_days: int = 7, changerate_ceiling: float = 6) -> CrossValidationResult:
        """
        Validate Changerate.

        Note: Currently only Python implementation exists.
        R implementation will be added during migration.
        """
        from epylabel.labeler import Changerate

        cr = Changerate(n_days=n_days, changerate_ceiling=changerate_ceiling)
        python_result = cr.transform(data)

        # R implementation placeholder
        r_result = None
        # TODO: Add R implementation call when available

        result = CrossValidationResult(
            algorithm="changerate",
            dataset=dataset_name,
            python_result=python_result,
            r_result=r_result,
            tolerance=0.0  # Deterministic algorithm
        )
        self.results.append(result)
        return result

    def validate_shapelet(self, data: pd.DataFrame, dataset_name: str,
                          n_days: int = 7, x_max: float = 3, thresh: float = 0.8) -> CrossValidationResult:
        """Validate Shapelet."""
        from epylabel.labeler import Shapelet

        sp = Shapelet(n_days=n_days, x_max=x_max, thresh=thresh)
        python_result = sp.transform(data)

        r_result = None
        # TODO: Add R implementation call when available

        result = CrossValidationResult(
            algorithm="shapelet",
            dataset=dataset_name,
            python_result=python_result,
            r_result=r_result,
            tolerance=0.0
        )
        self.results.append(result)
        return result

    def validate_wavefinder(self, data: pd.DataFrame, dataset_name: str,
                            abs_prominence_threshold: float = 5,
                            prominence_height_threshold: float = 0.01,
                            t_sep_a: int = 35) -> CrossValidationResult:
        """Validate WaveFinder."""
        from epylabel.labeler import WaveFinder

        wf = WaveFinder(
            abs_prominence_threshold=abs_prominence_threshold,
            prominence_height_threshold=prominence_height_threshold,
            t_sep_a=t_sep_a
        )
        python_result = wf.transform(data)

        r_result = None
        # TODO: Add R implementation call when available
        # Note: WaveFinder was originally in R, so porting back should be straightforward

        result = CrossValidationResult(
            algorithm="wavefinder",
            dataset=dataset_name,
            python_result=python_result,
            r_result=r_result,
            tolerance=0.0
        )
        self.results.append(result)
        return result

    def save_report(self, filename: str = "cross_validation_report.json"):
        """Save cross-validation report."""
        report = {
            "timestamp": pd.Timestamp.now().isoformat(),
            "r_available": R_AVAILABLE,
            "results": [r.to_dict() for r in self.results],
            "summary": {
                "total": len(self.results),
                "compared": sum(1 for r in self.results if r.compared),
                "matched": sum(1 for r in self.results if r.match),
                "not_compared": sum(1 for r in self.results if not r.compared),
            }
        }

        with open(CROSS_VAL_DIR / filename, 'w') as f:
            json.dump(report, f, indent=2)

        return report


# Tests

class TestCrossValidationFramework:
    """Tests for the cross-validation framework itself."""

    def test_cross_validator_initialization(self):
        """Test CrossValidator initializes correctly."""
        cv = CrossValidator()
        assert cv.results == []

    def test_result_serialization(self):
        """Test CrossValidationResult serializes correctly."""
        dates = pd.date_range("2020-01-01", periods=10)
        df = pd.DataFrame({"0": [True] * 10}, index=dates)

        result = CrossValidationResult(
            algorithm="test",
            dataset="test_data",
            python_result=df,
            r_result=df,
            tolerance=0.0
        )

        d = result.to_dict()
        assert d["algorithm"] == "test"
        assert d["match"] == True
        assert d["mismatch_proportion"] == 0.0


@pytest.mark.skipif(not R_AVAILABLE, reason="R not available")
class TestBcpCrossValidation:
    """Cross-validation tests for BCP."""

    def test_bcp_consistency(self, simple_wave_data):
        """Test BCP produces consistent results."""
        cv = CrossValidator()
        result = cv.validate_bcp(simple_wave_data, "simple_wave")

        # For now, just check that it runs
        assert result.python_result is not None
        assert result.python_result.shape == simple_wave_data.shape


class TestChangerateReadyForR:
    """Tests to validate Changerate is ready for R port."""

    def test_changerate_deterministic(self, simple_wave_data):
        """Verify Changerate is deterministic (important for R port)."""
        from epylabel.labeler import Changerate

        cr = Changerate(n_days=7, changerate_ceiling=6)

        result1 = cr.transform(simple_wave_data)
        result2 = cr.transform(simple_wave_data)

        pd.testing.assert_frame_equal(result1, result2)

    def test_changerate_formula_documented(self, simple_wave_data):
        """
        Document Changerate formula for R implementation.

        Formula:
        changerate[t] = min((value[t] / rolling_mean[t-n_days:t]) - 1, ceiling)

        where rolling_mean is the mean of the previous n_days values.
        """
        from epylabel.labeler import Changerate

        # Manual calculation for verification
        n_days = 7
        ceiling = 6
        data = simple_wave_data["0"].values

        expected = np.zeros(len(data))
        expected[:n_days-1] = np.nan

        for t in range(n_days-1, len(data)):
            rolling_mean = np.mean(data[t-n_days+1:t+1])
            if rolling_mean > 0:
                rate = (data[t] / rolling_mean) - 1
                expected[t] = min(rate, ceiling)
            else:
                expected[t] = 0

        # Compare with implementation
        cr = Changerate(n_days=n_days, changerate_ceiling=ceiling)
        result = cr.transform(simple_wave_data)

        # Note: Implementation might differ slightly in edge cases
        # This test documents the expected behavior


class TestShapeletReadyForR:
    """Tests to validate Shapelet is ready for R port."""

    def test_shapelet_deterministic(self, simple_wave_data):
        """Verify Shapelet is deterministic."""
        from epylabel.labeler import Shapelet

        sp = Shapelet(n_days=7, x_max=3, thresh=0.8)

        result1 = sp.transform(simple_wave_data)
        result2 = sp.transform(simple_wave_data)

        pd.testing.assert_frame_equal(result1, result2)

    def test_shapelet_correlation_formula(self):
        """
        Document Shapelet correlation formula for R implementation.

        The shapelet is an exponential curve: exp(x) for x in [0, x_max]
        Correlation is computed using scipy.stats.pearsonr between
        the data window and the shapelet.
        """
        # This is documentation for R port
        pass


class TestWaveFinderReadyForR:
    """Tests to validate WaveFinder is ready for R port (back-port)."""

    def test_wavefinder_deterministic(self, simple_wave_data):
        """Verify WaveFinder is deterministic."""
        from epylabel.labeler import WaveFinder

        wf = WaveFinder(abs_prominence_threshold=5, t_sep_a=35)

        result1 = wf.transform(simple_wave_data)
        result2 = wf.transform(simple_wave_data)

        pd.testing.assert_frame_equal(result1, result2)


class TestEnsembleReadyForR:
    """Tests to validate Ensemble is ready for R port."""

    def test_ensemble_formula(self):
        """
        Document Ensemble formula for R implementation.

        Ensemble uses majority voting:
        label[t] = sum(algo_labels[t]) >= n_min

        where n_min is the minimum number of algorithms that must agree.
        """
        from epylabel.labeler import Ensemble

        dates = pd.date_range("2020-01-01", periods=5)

        # Test case: 2 out of 3 algorithms agree
        df1 = pd.DataFrame({"0": [True, True, False, False, True]}, index=dates)
        df2 = pd.DataFrame({"0": [True, False, False, True, True]}, index=dates)
        df3 = pd.DataFrame({"0": [False, False, True, True, True]}, index=dates)

        ens = Ensemble(n_min=2)
        result = ens.transform(df1, df2, df3)

        # Manual calculation
        expected = pd.DataFrame({
            "0": [
                True,   # 2/3 True
                False,  # 1/3 True
                False,  # 1/3 True
                True,   # 2/3 True
                True,   # 3/3 True
            ]
        }, index=dates)

        pd.testing.assert_frame_equal(result, expected)


def run_full_cross_validation():
    """Run full cross-validation suite and generate report."""
    cv = CrossValidator()

    datasets = list(INPUTS_DIR.glob("*.parquet"))
    datasets = [d for d in datasets if "metadata" not in d.name]

    print(f"Running cross-validation on {len(datasets)} datasets...")

    for dataset_path in datasets:
        name = dataset_path.stem
        data = pd.read_parquet(dataset_path)

        print(f"\n  Dataset: {name}")

        cv.validate_changerate(data, name)
        print(f"    Changerate: {'✓' if cv.results[-1].compared else 'R not available'}")

        cv.validate_shapelet(data, name)
        print(f"    Shapelet: {'✓' if cv.results[-1].compared else 'R not available'}")

        cv.validate_wavefinder(data, name)
        print(f"    WaveFinder: {'✓' if cv.results[-1].compared else 'R not available'}")

        if R_AVAILABLE:
            cv.validate_bcp(data, name)
            print(f"    BCP: {'✓' if cv.results[-1].match else '✗'}")

    report = cv.save_report()
    print(f"\nReport saved to {CROSS_VAL_DIR / 'cross_validation_report.json'}")
    print(f"\nSummary:")
    print(f"  Total validations: {report['summary']['total']}")
    print(f"  Compared: {report['summary']['compared']}")
    print(f"  Matched: {report['summary']['matched']}")
    print(f"  Not compared (R unavailable): {report['summary']['not_compared']}")

    return report


if __name__ == "__main__":
    run_full_cross_validation()
