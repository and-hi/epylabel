"""
Golden Master (Snapshot) Tests for epylabel.

These tests compare algorithm outputs against saved reference outputs.
They are essential for:
1. Detecting unintended changes in algorithm behavior
2. Validating cross-language implementations (Python vs R)
3. Ensuring reproducibility across versions

Usage:
    pytest tests/test_golden_master.py                    # Run tests
    pytest tests/test_golden_master.py --update-golden    # Update reference outputs
"""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import json
from datetime import datetime

from epylabel.labeler import (
    Bcp, Changerate, Ensemble, ExponentialGrowth,
    GapFiller, Shapelet, WaveFinder
)
from epylabel.pipeline import Pipeline
from epylabel.metrics import summary


# Paths
TEST_DATA_DIR = Path(__file__).parent.parent / "test_data"
INPUTS_DIR = TEST_DATA_DIR / "inputs"
EXPECTED_DIR = TEST_DATA_DIR / "expected_outputs" / "python"


def ensure_expected_dir():
    """Ensure expected output directory exists."""
    EXPECTED_DIR.mkdir(parents=True, exist_ok=True)
    for alg in ["changerate", "bcp", "shapelet", "wavefinder", "exponential_growth", "gapfiller", "ensemble"]:
        (EXPECTED_DIR / alg).mkdir(exist_ok=True)


class GoldenMasterManager:
    """Manages golden master (reference) outputs."""

    def __init__(self, algorithm_name: str):
        self.algorithm_name = algorithm_name
        self.output_dir = EXPECTED_DIR / algorithm_name
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def get_reference_path(self, dataset_name: str) -> Path:
        """Get path to reference output file."""
        return self.output_dir / f"{dataset_name}_labels.parquet"

    def get_metadata_path(self, dataset_name: str) -> Path:
        """Get path to metadata file."""
        return self.output_dir / f"{dataset_name}_metadata.json"

    def save_reference(self, dataset_name: str, labels: pd.DataFrame, params: dict):
        """Save reference output."""
        labels.to_parquet(self.get_reference_path(dataset_name))
        metadata = {
            "algorithm": self.algorithm_name,
            "dataset": dataset_name,
            "params": params,
            "generated_at": datetime.now().isoformat(),
            "shape": list(labels.shape),
            "columns": list(labels.columns),
        }
        with open(self.get_metadata_path(dataset_name), 'w') as f:
            json.dump(metadata, f, indent=2)

    def load_reference(self, dataset_name: str) -> pd.DataFrame:
        """Load reference output."""
        path = self.get_reference_path(dataset_name)
        if not path.exists():
            return None
        return pd.read_parquet(path)

    def has_reference(self, dataset_name: str) -> bool:
        """Check if reference exists."""
        return self.get_reference_path(dataset_name).exists()


def compare_boolean_labels(actual: pd.DataFrame, expected: pd.DataFrame, tolerance: float = 0.0) -> tuple:
    """
    Compare two boolean label DataFrames.

    Returns:
        (match, mismatch_proportion, details)
    """
    if actual.shape != expected.shape:
        return False, 1.0, f"Shape mismatch: {actual.shape} vs {expected.shape}"

    if not actual.columns.equals(expected.columns):
        return False, 1.0, f"Column mismatch: {list(actual.columns)} vs {list(expected.columns)}"

    mismatches = (actual != expected).sum().sum()
    total = actual.size
    mismatch_prop = mismatches / total if total > 0 else 0.0

    if mismatch_prop <= tolerance:
        return True, mismatch_prop, f"Match within tolerance ({mismatch_prop:.4f} <= {tolerance})"
    else:
        return False, mismatch_prop, f"Mismatch: {mismatch_prop:.4f} > {tolerance}"


# Test fixtures
@pytest.fixture
def update_golden(request):
    """Check if we should update golden masters."""
    return request.config.getoption("--update-golden", default=False)


def pytest_addoption(parser):
    """Add command line option for updating golden masters."""
    try:
        parser.addoption("--update-golden", action="store_true", default=False,
                         help="Update golden master reference outputs")
    except ValueError:
        pass  # Option already added


# Golden Master Tests

class TestChangerateGoldenMaster:
    """Golden master tests for Changerate."""

    PARAMS = {"n_days": 7, "changerate_ceiling": 6}
    TOLERANCE = 0.0  # Exact match required for deterministic algorithm

    @pytest.fixture
    def manager(self):
        return GoldenMasterManager("changerate")

    @pytest.fixture
    def algorithm(self):
        return Changerate(**self.PARAMS)

    @pytest.mark.parametrize("dataset_name", [
        "simple_wave", "multi_wave", "multi_location", "edge_cases"
    ])
    def test_golden_master(self, manager, algorithm, dataset_name, update_golden):
        """Test Changerate output matches golden master."""
        input_data = pd.read_parquet(INPUTS_DIR / f"{dataset_name}.parquet")
        actual = algorithm.transform(input_data)

        if update_golden or not manager.has_reference(dataset_name):
            manager.save_reference(dataset_name, actual, self.PARAMS)
            pytest.skip(f"Updated golden master for {dataset_name}")

        expected = manager.load_reference(dataset_name)
        match, mismatch_prop, details = compare_boolean_labels(actual, expected, self.TOLERANCE)

        # For numeric comparison (Changerate outputs floats, not bools)
        if not match:
            # Use numeric comparison
            match = np.allclose(actual.values, expected.values, rtol=1e-5, atol=1e-8, equal_nan=True)
            details = "Numeric comparison used"

        assert match, f"Changerate golden master mismatch for {dataset_name}: {details}"


class TestShapeletGoldenMaster:
    """Golden master tests for Shapelet."""

    PARAMS = {"n_days": 7, "x_max": 3, "thresh": 0.8}
    TOLERANCE = 0.0

    @pytest.fixture
    def manager(self):
        return GoldenMasterManager("shapelet")

    @pytest.fixture
    def algorithm(self):
        return Shapelet(**self.PARAMS)

    @pytest.mark.parametrize("dataset_name", [
        "simple_wave", "multi_wave", "shapelet_test_cases"
    ])
    def test_golden_master(self, manager, algorithm, dataset_name, update_golden):
        """Test Shapelet output matches golden master."""
        input_data = pd.read_parquet(INPUTS_DIR / f"{dataset_name}.parquet")
        actual = algorithm.transform(input_data)

        if update_golden or not manager.has_reference(dataset_name):
            manager.save_reference(dataset_name, actual, self.PARAMS)
            pytest.skip(f"Updated golden master for {dataset_name}")

        expected = manager.load_reference(dataset_name)
        match, mismatch_prop, details = compare_boolean_labels(actual, expected, self.TOLERANCE)

        assert match, f"Shapelet golden master mismatch for {dataset_name}: {details}"


class TestWaveFinderGoldenMaster:
    """Golden master tests for WaveFinder."""

    PARAMS = {"abs_prominence_threshold": 5, "prominence_height_threshold": 0.01, "t_sep_a": 35}
    TOLERANCE = 0.0

    @pytest.fixture
    def manager(self):
        return GoldenMasterManager("wavefinder")

    @pytest.fixture
    def algorithm(self):
        return WaveFinder(**self.PARAMS)

    @pytest.mark.parametrize("dataset_name", [
        "simple_wave", "multi_wave", "real_covid_sample"
    ])
    def test_golden_master(self, manager, algorithm, dataset_name, update_golden):
        """Test WaveFinder output matches golden master."""
        input_data = pd.read_parquet(INPUTS_DIR / f"{dataset_name}.parquet")
        actual = algorithm.transform(input_data)

        if update_golden or not manager.has_reference(dataset_name):
            manager.save_reference(dataset_name, actual, self.PARAMS)
            pytest.skip(f"Updated golden master for {dataset_name}")

        expected = manager.load_reference(dataset_name)
        match, mismatch_prop, details = compare_boolean_labels(actual, expected, self.TOLERANCE)

        assert match, f"WaveFinder golden master mismatch for {dataset_name}: {details}"


class TestBcpGoldenMaster:
    """Golden master tests for BCP.

    Note: BCP uses MCMC and is stochastic. We use a higher tolerance
    and check for structural similarity rather than exact match.
    """

    PARAMS = {"d": 500, "p0": 0.01, "thresh": 0.5}
    TOLERANCE = 0.15  # Allow 15% mismatch due to MCMC randomness

    @pytest.fixture
    def manager(self):
        return GoldenMasterManager("bcp")

    @pytest.fixture
    def algorithm(self):
        return Bcp(**self.PARAMS)

    @pytest.mark.parametrize("dataset_name", [
        "simple_wave", "known_changepoints"
    ])
    def test_golden_master(self, manager, algorithm, dataset_name, update_golden):
        """Test BCP output is structurally similar to golden master."""
        input_data = pd.read_parquet(INPUTS_DIR / f"{dataset_name}.parquet")

        # For BCP, we need Changerate preprocessing
        cr = Changerate(n_days=7, changerate_ceiling=6)
        preprocessed = cr.transform(input_data).fillna(0)
        actual = algorithm.transform(preprocessed)

        if update_golden or not manager.has_reference(dataset_name):
            manager.save_reference(dataset_name, actual, self.PARAMS)
            pytest.skip(f"Updated golden master for {dataset_name}")

        expected = manager.load_reference(dataset_name)
        match, mismatch_prop, details = compare_boolean_labels(actual, expected, self.TOLERANCE)

        assert match, f"BCP golden master mismatch for {dataset_name}: {details} (mismatch: {mismatch_prop:.2%})"


class TestExponentialGrowthGoldenMaster:
    """Golden master tests for ExponentialGrowth."""

    PARAMS = {"n_days": 7, "p_thresh": 0.05, "slope_thresh": 0}
    TOLERANCE = 0.0

    @pytest.fixture
    def manager(self):
        return GoldenMasterManager("exponential_growth")

    @pytest.fixture
    def algorithm(self):
        return ExponentialGrowth(**self.PARAMS)

    @pytest.mark.parametrize("dataset_name", [
        "exponential_growth_cases", "simple_wave"
    ])
    def test_golden_master(self, manager, algorithm, dataset_name, update_golden):
        """Test ExponentialGrowth output matches golden master."""
        input_data = pd.read_parquet(INPUTS_DIR / f"{dataset_name}.parquet")
        actual = algorithm.transform(input_data)

        if update_golden or not manager.has_reference(dataset_name):
            manager.save_reference(dataset_name, actual, self.PARAMS)
            pytest.skip(f"Updated golden master for {dataset_name}")

        expected = manager.load_reference(dataset_name)
        match, mismatch_prop, details = compare_boolean_labels(actual, expected, self.TOLERANCE)

        assert match, f"ExponentialGrowth golden master mismatch for {dataset_name}: {details}"


class TestPipelineGoldenMaster:
    """Golden master tests for full pipelines."""

    @pytest.fixture
    def manager(self):
        return GoldenMasterManager("ensemble")

    def test_full_pipeline_golden_master(self, manager, update_golden):
        """Test full BCP pipeline produces consistent results."""
        dataset_name = "simple_wave"
        input_data = pd.read_parquet(INPUTS_DIR / f"{dataset_name}.parquet")

        # Run full pipeline
        cr = Changerate(n_days=7, changerate_ceiling=6)
        bcp = Bcp(d=200, p0=0.01, thresh=0.5)
        sp = Shapelet(n_days=7, x_max=3, thresh=0.8)
        wf = WaveFinder(abs_prominence_threshold=5)
        ens = Ensemble(n_min=2)

        # Get individual labels
        cr_result = cr.transform(input_data)
        bcp_labels = bcp.transform(cr_result.fillna(0))
        sp_labels = sp.transform(input_data)
        wf_labels = wf.transform(input_data)

        # Ensemble
        actual = ens.transform(bcp_labels, sp_labels, wf_labels)

        params = {
            "changerate": {"n_days": 7, "changerate_ceiling": 6},
            "bcp": {"d": 200, "p0": 0.01, "thresh": 0.5},
            "shapelet": {"n_days": 7, "x_max": 3, "thresh": 0.8},
            "wavefinder": {"abs_prominence_threshold": 5},
            "ensemble": {"n_min": 2}
        }

        if update_golden or not manager.has_reference(f"pipeline_{dataset_name}"):
            manager.save_reference(f"pipeline_{dataset_name}", actual, params)
            pytest.skip(f"Updated golden master for pipeline_{dataset_name}")

        expected = manager.load_reference(f"pipeline_{dataset_name}")
        # Higher tolerance for pipeline due to BCP stochasticity
        match, mismatch_prop, details = compare_boolean_labels(actual, expected, tolerance=0.2)

        assert match, f"Pipeline golden master mismatch: {details}"


# Utility function to generate all golden masters
def generate_all_golden_masters():
    """Generate all golden master reference outputs."""
    ensure_expected_dir()

    datasets = list(INPUTS_DIR.glob("*.parquet"))
    datasets = [d for d in datasets if "metadata" not in d.name]

    print(f"Generating golden masters for {len(datasets)} datasets...")

    # Changerate
    cr = Changerate(n_days=7, changerate_ceiling=6)
    manager = GoldenMasterManager("changerate")
    for dataset_path in datasets:
        name = dataset_path.stem
        data = pd.read_parquet(dataset_path)
        result = cr.transform(data)
        manager.save_reference(name, result, {"n_days": 7, "changerate_ceiling": 6})
        print(f"  Changerate: {name}")

    # Shapelet
    sp = Shapelet(n_days=7, x_max=3, thresh=0.8)
    manager = GoldenMasterManager("shapelet")
    for dataset_path in datasets:
        name = dataset_path.stem
        data = pd.read_parquet(dataset_path)
        result = sp.transform(data)
        manager.save_reference(name, result, {"n_days": 7, "x_max": 3, "thresh": 0.8})
        print(f"  Shapelet: {name}")

    # WaveFinder
    wf = WaveFinder(abs_prominence_threshold=5, prominence_height_threshold=0.01, t_sep_a=35)
    manager = GoldenMasterManager("wavefinder")
    for dataset_path in datasets:
        name = dataset_path.stem
        data = pd.read_parquet(dataset_path)
        try:
            result = wf.transform(data)
            manager.save_reference(name, result, {"abs_prominence_threshold": 5, "prominence_height_threshold": 0.01, "t_sep_a": 35})
            print(f"  WaveFinder: {name}")
        except Exception as e:
            print(f"  WaveFinder: {name} - SKIPPED ({e})")

    # ExponentialGrowth
    eg = ExponentialGrowth(n_days=7, thresh=0.05)
    manager = GoldenMasterManager("exponential_growth")
    for dataset_path in datasets:
        name = dataset_path.stem
        data = pd.read_parquet(dataset_path)
        try:
            result = eg.transform(data)
            manager.save_reference(name, result, {"n_days": 7, "thresh": 0.05})
            print(f"  ExponentialGrowth: {name}")
        except Exception as e:
            print(f"  ExponentialGrowth: {name} - SKIPPED ({e})")

    print("Done generating golden masters!")


if __name__ == "__main__":
    generate_all_golden_masters()
