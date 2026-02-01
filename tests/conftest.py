"""
Pytest configuration and fixtures for epylabel tests.
"""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path

# Test data directory
TEST_DATA_DIR = Path(__file__).parent.parent / "test_data"
INPUTS_DIR = TEST_DATA_DIR / "inputs"
EXPECTED_DIR = TEST_DATA_DIR / "expected_outputs" / "python"


@pytest.fixture(scope="session")
def test_data_dir():
    """Return the test data directory path."""
    return TEST_DATA_DIR


@pytest.fixture(scope="session")
def inputs_dir():
    """Return the inputs directory path."""
    return INPUTS_DIR


@pytest.fixture(scope="session")
def expected_dir():
    """Return the expected outputs directory path."""
    return EXPECTED_DIR


@pytest.fixture(scope="session")
def simple_wave_data():
    """Load simple wave test data."""
    return pd.read_parquet(INPUTS_DIR / "simple_wave.parquet")


@pytest.fixture(scope="session")
def multi_wave_data():
    """Load multi-wave test data."""
    return pd.read_parquet(INPUTS_DIR / "multi_wave.parquet")


@pytest.fixture(scope="session")
def noisy_data():
    """Load noisy test data."""
    return pd.read_parquet(INPUTS_DIR / "noisy.parquet")


@pytest.fixture(scope="session")
def edge_cases_data():
    """Load edge cases test data."""
    return pd.read_parquet(INPUTS_DIR / "edge_cases.parquet")


@pytest.fixture(scope="session")
def multi_location_data():
    """Load multi-location test data."""
    return pd.read_parquet(INPUTS_DIR / "multi_location.parquet")


@pytest.fixture(scope="session")
def real_covid_sample_data():
    """Load real COVID sample test data."""
    return pd.read_parquet(INPUTS_DIR / "real_covid_sample.parquet")


@pytest.fixture(scope="session")
def short_series_data():
    """Load short series test data."""
    return pd.read_parquet(INPUTS_DIR / "short_series.parquet")


@pytest.fixture(scope="session")
def known_changepoints_data():
    """Load known changepoints test data."""
    return pd.read_parquet(INPUTS_DIR / "known_changepoints.parquet")


@pytest.fixture(scope="session")
def exponential_growth_data():
    """Load exponential growth test data."""
    return pd.read_parquet(INPUTS_DIR / "exponential_growth_cases.parquet")


@pytest.fixture(scope="session")
def shapelet_test_data():
    """Load shapelet test data."""
    return pd.read_parquet(INPUTS_DIR / "shapelet_test_cases.parquet")


@pytest.fixture(scope="session")
def ensemble_test_data():
    """Load ensemble test data."""
    return pd.read_parquet(INPUTS_DIR / "ensemble_test_cases.parquet")


@pytest.fixture(scope="session")
def all_test_datasets(inputs_dir):
    """Load all test datasets as a dictionary."""
    datasets = {}
    for f in inputs_dir.glob("*.parquet"):
        if "metadata" not in f.name:
            datasets[f.stem] = pd.read_parquet(f)
    return datasets


# Comparison helpers
def compare_labels(actual: pd.DataFrame, expected: pd.DataFrame, tolerance: float = 0.0) -> bool:
    """
    Compare two label DataFrames.

    Args:
        actual: Actual labels
        expected: Expected labels
        tolerance: Allowed proportion of different labels (0.0 = exact match)

    Returns:
        True if labels match within tolerance
    """
    if actual.shape != expected.shape:
        return False

    if tolerance == 0.0:
        return actual.equals(expected)

    # For boolean labels, calculate mismatch proportion
    mismatches = (actual != expected).sum().sum()
    total = actual.size
    mismatch_prop = mismatches / total

    return mismatch_prop <= tolerance


def compare_numeric(actual: pd.DataFrame, expected: pd.DataFrame, rtol: float = 1e-5, atol: float = 1e-8) -> bool:
    """
    Compare two numeric DataFrames with tolerance.

    Args:
        actual: Actual values
        expected: Expected values
        rtol: Relative tolerance
        atol: Absolute tolerance

    Returns:
        True if values match within tolerance
    """
    return np.allclose(actual.values, expected.values, rtol=rtol, atol=atol, equal_nan=True)


@pytest.fixture
def compare_labels_fixture():
    """Provide the compare_labels function as a fixture."""
    return compare_labels


@pytest.fixture
def compare_numeric_fixture():
    """Provide the compare_numeric function as a fixture."""
    return compare_numeric
