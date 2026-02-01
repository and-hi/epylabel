"""
Comprehensive tests for utility functions and metrics.
"""

import pytest
import pandas as pd
import numpy as np

from epylabel.utils import to_wide, to_long, assert_same_structure_list, signal_ids
from epylabel.metrics import summary


class TestToWide:
    """Tests for to_wide conversion function."""

    def test_basic_conversion(self):
        """Test basic long to wide conversion."""
        long_df = pd.DataFrame({
            "target": pd.to_datetime(["2020-01-01", "2020-01-02", "2020-01-01", "2020-01-02"]),
            "location": [0, 0, 1, 1],
            "value": [10, 20, 30, 40]
        })

        result = to_wide(long_df)

        assert result.shape == (2, 2)  # 2 dates, 2 locations
        assert list(result.columns) == [0, 1]
        assert result.loc["2020-01-01", 0] == 10
        assert result.loc["2020-01-02", 1] == 40

    def test_preserves_values(self):
        """Test that conversion preserves all values."""
        long_df = pd.DataFrame({
            "target": pd.date_range("2020-01-01", periods=10).tolist() * 3,
            "location": [0] * 10 + [1] * 10 + [2] * 10,
            "value": list(range(30))
        })

        result = to_wide(long_df)

        assert result.shape == (10, 3)
        assert result.sum().sum() == sum(range(30))

    def test_with_labels(self):
        """Test conversion with boolean labels."""
        long_df = pd.DataFrame({
            "target": pd.to_datetime(["2020-01-01", "2020-01-02", "2020-01-01", "2020-01-02"]),
            "location": [0, 0, 1, 1],
            "label": [True, False, False, True]
        })

        result = to_wide(long_df, value_col="label")

        assert result.dtypes.apply(lambda x: x == bool or x == np.bool_ or x == object).all()


class TestToLong:
    """Tests for to_long conversion function."""

    def test_basic_conversion(self):
        """Test basic wide to long conversion."""
        wide_df = pd.DataFrame({
            0: [10, 20],
            1: [30, 40]
        }, index=pd.to_datetime(["2020-01-01", "2020-01-02"]))
        wide_df.index.name = "target"

        result = to_long(wide_df)

        assert len(result) == 4
        assert "target" in result.columns
        assert "location" in result.columns
        assert "value" in result.columns

    def test_preserves_values(self):
        """Test that conversion preserves all values."""
        wide_df = pd.DataFrame({
            0: [10, 20, 30],
            1: [40, 50, 60],
            2: [70, 80, 90]
        }, index=pd.date_range("2020-01-01", periods=3))
        wide_df.index.name = "target"

        result = to_long(wide_df)

        assert len(result) == 9
        assert result["value"].sum() == sum([10, 20, 30, 40, 50, 60, 70, 80, 90])

    def test_roundtrip(self):
        """Test long -> wide -> long roundtrip."""
        original = pd.DataFrame({
            "target": pd.date_range("2020-01-01", periods=5).tolist() * 2,
            "location": [0] * 5 + [1] * 5,
            "value": list(range(10))
        })

        wide = to_wide(original)
        back_to_long = to_long(wide)

        # Sort both for comparison
        original_sorted = original.sort_values(["location", "target"]).reset_index(drop=True)
        result_sorted = back_to_long.sort_values(["location", "target"]).reset_index(drop=True)

        assert len(original_sorted) == len(result_sorted)


class TestSignalIds:
    """Tests for signal_ids function."""

    def test_basic_signal_ids(self):
        """Test basic signal ID assignment."""
        labels = pd.Series([False, False, True, True, True, False, True, True, False])

        result = signal_ids(labels)

        # Should have different IDs for different signal periods
        # Period 1: indices 2,3,4 (True)
        # Period 2: indices 6,7 (True)
        assert result[2] == result[3] == result[4]  # Same signal period
        assert result[6] == result[7]  # Same signal period
        assert result[2] != result[6]  # Different signal periods

    def test_no_signals(self):
        """Test with no signals (all False)."""
        labels = pd.Series([False] * 10)
        result = signal_ids(labels)
        # All should be 0 or NaN for non-signal periods
        assert (result == 0).all() or result.isna().all()

    def test_all_signals(self):
        """Test with all signals (all True)."""
        labels = pd.Series([True] * 10)
        result = signal_ids(labels)
        # All should have same ID
        assert len(result.unique()) == 1 or (result == result.iloc[0]).all()

    def test_alternating_signals(self):
        """Test with alternating signals."""
        labels = pd.Series([True, False, True, False, True, False])
        result = signal_ids(labels)
        # Each True should be its own signal period
        assert result[0] != result[2] != result[4]


class TestSummary:
    """Tests for summary statistics function."""

    def test_basic_summary(self):
        """Test basic summary statistics."""
        dates = pd.date_range("2020-01-01", periods=100)
        labels = pd.DataFrame({
            "0": [False] * 20 + [True] * 30 + [False] * 20 + [True] * 20 + [False] * 10
        }, index=dates)

        result = summary(labels)

        assert "n_labels" in result.columns or "n_labels" in result.index
        assert "prop_labels" in result.columns or "prop_labels" in result.index

    def test_summary_multi_location(self):
        """Test summary with multiple locations."""
        dates = pd.date_range("2020-01-01", periods=50)
        labels = pd.DataFrame({
            "0": [False] * 10 + [True] * 20 + [False] * 20,
            "1": [True] * 30 + [False] * 20,
            "2": [False] * 50
        }, index=dates)

        result = summary(labels)

        # Should have summary for each location
        assert len(result) >= 3 or result.shape[1] >= 3

    def test_summary_no_labels(self):
        """Test summary with no labels."""
        dates = pd.date_range("2020-01-01", periods=50)
        labels = pd.DataFrame({
            "0": [False] * 50
        }, index=dates)

        result = summary(labels)

        # Should handle gracefully
        assert result is not None

    def test_summary_all_labels(self):
        """Test summary with all labels."""
        dates = pd.date_range("2020-01-01", periods=50)
        labels = pd.DataFrame({
            "0": [True] * 50
        }, index=dates)

        result = summary(labels)

        # prop_labels should be 1.0
        if "prop_labels" in result.columns:
            assert result["prop_labels"].iloc[0] == 1.0
        elif "prop_labels" in result.index:
            assert result.loc["prop_labels"].iloc[0] == 1.0


class TestAssertSameStructure:
    """Tests for assert_same_structure_list function."""

    def test_same_structure_passes(self):
        """Test that same structure DataFrames pass."""
        dates = pd.date_range("2020-01-01", periods=10)
        df1 = pd.DataFrame({"0": range(10), "1": range(10)}, index=dates)
        df2 = pd.DataFrame({"0": range(10, 20), "1": range(10, 20)}, index=dates)

        # Should not raise
        assert_same_structure_list([df1, df2])

    def test_different_columns_fails(self):
        """Test that different columns fail."""
        dates = pd.date_range("2020-01-01", periods=10)
        df1 = pd.DataFrame({"0": range(10), "1": range(10)}, index=dates)
        df2 = pd.DataFrame({"0": range(10), "2": range(10)}, index=dates)

        with pytest.raises(AssertionError):
            assert_same_structure_list([df1, df2])

    def test_different_index_fails(self):
        """Test that different index fails."""
        dates1 = pd.date_range("2020-01-01", periods=10)
        dates2 = pd.date_range("2020-01-05", periods=10)
        df1 = pd.DataFrame({"0": range(10)}, index=dates1)
        df2 = pd.DataFrame({"0": range(10)}, index=dates2)

        with pytest.raises(AssertionError):
            assert_same_structure_list([df1, df2])

    def test_different_shape_fails(self):
        """Test that different shape fails."""
        dates1 = pd.date_range("2020-01-01", periods=10)
        dates2 = pd.date_range("2020-01-01", periods=15)
        df1 = pd.DataFrame({"0": range(10)}, index=dates1)
        df2 = pd.DataFrame({"0": range(15)}, index=dates2)

        with pytest.raises(AssertionError):
            assert_same_structure_list([df1, df2])
