#!/usr/bin/env python3
"""
Generate comprehensive test datasets for epylabel validation.

These datasets are designed to:
1. Test all edge cases and boundary conditions
2. Provide reproducible test cases for cross-language validation
3. Cover various outbreak patterns found in real epidemiological data
"""

import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime, timedelta

# Set seed for reproducibility
np.random.seed(42)

OUTPUT_DIR = Path(__file__).parent / "inputs"
OUTPUT_DIR.mkdir(exist_ok=True)


def create_date_index(start_date: str, n_days: int) -> pd.DatetimeIndex:
    """Create a datetime index starting from start_date."""
    return pd.date_range(start=start_date, periods=n_days, freq='D')


def save_dataset(df: pd.DataFrame, name: str, description: str):
    """Save dataset with metadata."""
    output_path = OUTPUT_DIR / f"{name}.parquet"
    df.to_parquet(output_path)
    print(f"  Created: {name}.parquet - {description}")
    return df


def generate_simple_wave():
    """
    Generate a simple single wave pattern.
    Clear exponential growth followed by decline.
    """
    n_days = 200
    dates = create_date_index("2020-01-01", n_days)

    # Create a simple wave: baseline -> growth -> peak -> decline -> baseline
    baseline = 10
    peak_value = 500
    peak_day = 100
    growth_rate = 0.08
    decline_rate = 0.06

    values = np.zeros(n_days)
    for i in range(n_days):
        if i < 50:
            values[i] = baseline + np.random.normal(0, 2)
        elif i < peak_day:
            days_from_start = i - 50
            values[i] = baseline * np.exp(growth_rate * days_from_start) + np.random.normal(0, 5)
        else:
            days_from_peak = i - peak_day
            values[i] = peak_value * np.exp(-decline_rate * days_from_peak) + np.random.normal(0, 5)

    values = np.maximum(values, 0)  # No negative values

    # Create wide format DataFrame
    df = pd.DataFrame({"0": values}, index=dates)
    df.index.name = "target"

    return save_dataset(df, "simple_wave", "Single clear wave pattern")


def generate_multi_wave():
    """
    Generate multiple distinct waves.
    Similar to COVID-19 pandemic waves.
    """
    n_days = 500
    dates = create_date_index("2020-01-01", n_days)

    baseline = 10
    values = np.ones(n_days) * baseline

    # Wave 1: days 50-120
    wave1_peak = 80
    for i in range(50, 120):
        if i < wave1_peak:
            values[i] += 200 * np.exp(0.1 * (i - 50)) / np.exp(0.1 * 30)
        else:
            values[i] += 200 * np.exp(-0.08 * (i - wave1_peak))

    # Wave 2: days 180-280 (larger)
    wave2_peak = 230
    for i in range(180, 280):
        if i < wave2_peak:
            values[i] += 400 * np.exp(0.08 * (i - 180)) / np.exp(0.08 * 50)
        else:
            values[i] += 400 * np.exp(-0.06 * (i - wave2_peak))

    # Wave 3: days 320-420
    wave3_peak = 370
    for i in range(320, 420):
        if i < wave3_peak:
            values[i] += 300 * np.exp(0.07 * (i - 320)) / np.exp(0.07 * 50)
        else:
            values[i] += 300 * np.exp(-0.07 * (i - wave3_peak))

    # Add noise
    values += np.random.normal(0, 5, n_days)
    values = np.maximum(values, 0)

    df = pd.DataFrame({"0": values}, index=dates)
    df.index.name = "target"

    return save_dataset(df, "multi_wave", "Multiple distinct waves (3 waves)")


def generate_noisy_data():
    """
    Generate high-noise data to test robustness.
    """
    n_days = 300
    dates = create_date_index("2020-01-01", n_days)

    # Underlying signal
    signal = 50 + 30 * np.sin(np.linspace(0, 4 * np.pi, n_days))

    # Add significant noise (SNR ~ 1)
    noise = np.random.normal(0, 30, n_days)
    values = signal + noise
    values = np.maximum(values, 0)

    df = pd.DataFrame({"0": values}, index=dates)
    df.index.name = "target"

    return save_dataset(df, "noisy", "High variance noisy data")


def generate_edge_cases():
    """
    Generate edge case scenarios.
    """
    n_days = 100
    dates = create_date_index("2020-01-01", n_days)

    # Location 0: All zeros
    loc0 = np.zeros(n_days)

    # Location 1: Constant value
    loc1 = np.ones(n_days) * 50

    # Location 2: Single spike
    loc2 = np.ones(n_days) * 10
    loc2[50] = 500

    # Location 3: Step function
    loc3 = np.concatenate([np.ones(50) * 10, np.ones(50) * 100])

    # Location 4: Linear increase
    loc4 = np.linspace(0, 200, n_days)

    # Location 5: Exponential growth (no decline)
    loc5 = 10 * np.exp(0.03 * np.arange(n_days))

    df = pd.DataFrame({
        "0": loc0,
        "1": loc1,
        "2": loc2,
        "3": loc3,
        "4": loc4,
        "5": loc5
    }, index=dates)
    df.index.name = "target"

    return save_dataset(df, "edge_cases", "Various edge case scenarios")


def generate_multi_location():
    """
    Generate data with multiple locations (simulating counties/states).
    """
    n_days = 200
    n_locations = 10
    dates = create_date_index("2020-01-01", n_days)

    data = {}
    for loc in range(n_locations):
        # Each location has slightly different wave timing and magnitude
        offset = loc * 10  # Stagger waves
        magnitude = 100 + loc * 50  # Different magnitudes

        values = np.ones(n_days) * 10
        peak_day = 80 + offset

        for i in range(30 + offset, min(150 + offset, n_days)):
            if i < peak_day:
                values[i] += magnitude * np.exp(0.1 * (i - 30 - offset)) / np.exp(0.1 * 50)
            else:
                values[i] += magnitude * np.exp(-0.08 * (i - peak_day))

        values += np.random.normal(0, 5, n_days)
        values = np.maximum(values, 0)
        data[str(loc)] = values

    df = pd.DataFrame(data, index=dates)
    df.index.name = "target"

    return save_dataset(df, "multi_location", f"Multiple locations ({n_locations} locations)")


def generate_real_covid_sample():
    """
    Generate synthetic data that mimics real COVID-19 characteristics.
    Based on typical German COVID-19 incidence patterns.
    """
    n_days = 1000  # ~3 years
    dates = create_date_index("2020-03-01", n_days)

    values = np.ones(n_days) * 5  # Baseline

    # Wave 1: Spring 2020 (days 0-80)
    for i in range(0, 80):
        if i < 40:
            values[i] += 150 * np.exp(0.15 * i) / np.exp(0.15 * 40)
        else:
            values[i] += 150 * np.exp(-0.1 * (i - 40))

    # Summer plateau (days 80-200)
    values[80:200] += np.random.uniform(5, 20, 120)

    # Wave 2: Fall/Winter 2020 (days 200-350)
    for i in range(200, 350):
        if i < 280:
            values[i] += 300 * np.exp(0.05 * (i - 200)) / np.exp(0.05 * 80)
        else:
            values[i] += 300 * np.exp(-0.03 * (i - 280))

    # Wave 3: Alpha (days 350-450)
    for i in range(350, 450):
        if i < 400:
            values[i] += 250 * np.exp(0.06 * (i - 350)) / np.exp(0.06 * 50)
        else:
            values[i] += 250 * np.exp(-0.05 * (i - 400))

    # Summer 2021 (days 450-550)
    values[450:550] += np.random.uniform(10, 40, 100)

    # Wave 4: Delta (days 550-700)
    for i in range(550, 700):
        if i < 630:
            values[i] += 400 * np.exp(0.04 * (i - 550)) / np.exp(0.04 * 80)
        else:
            values[i] += 400 * np.exp(-0.04 * (i - 630))

    # Wave 5: Omicron (days 700-850) - very high
    for i in range(700, 850):
        if i < 780:
            values[i] += 1500 * np.exp(0.08 * (i - 700)) / np.exp(0.08 * 80)
        else:
            values[i] += 1500 * np.exp(-0.03 * (i - 780))

    # Declining phase (days 850-1000)
    for i in range(850, 1000):
        values[i] += 200 * np.exp(-0.02 * (i - 850))

    # Add weekly pattern (lower on weekends)
    for i in range(n_days):
        day_of_week = i % 7
        if day_of_week in [5, 6]:  # Weekend
            values[i] *= 0.7

    # Add noise
    values += np.random.normal(0, values * 0.1)  # 10% relative noise
    values = np.maximum(values, 0)

    df = pd.DataFrame({"0": values}, index=dates)
    df.index.name = "target"

    return save_dataset(df, "real_covid_sample", "Synthetic COVID-19-like pattern")


def generate_short_series():
    """
    Generate very short time series for boundary testing.
    """
    dates = create_date_index("2020-01-01", 30)

    # Short wave
    values = np.concatenate([
        np.ones(10) * 10,
        np.linspace(10, 100, 10),
        np.linspace(100, 20, 10)
    ])

    df = pd.DataFrame({"0": values}, index=dates)
    df.index.name = "target"

    return save_dataset(df, "short_series", "Very short time series (30 days)")


def generate_known_changepoints():
    """
    Generate data with known, exact changepoints for BCP validation.
    """
    n_days = 300
    dates = create_date_index("2020-01-01", n_days)

    # Define exact changepoints
    changepoints = [50, 100, 180, 250]
    levels = [10, 50, 20, 80, 30]

    values = np.zeros(n_days)
    prev_cp = 0
    for i, cp in enumerate(changepoints + [n_days]):
        values[prev_cp:cp] = levels[i]
        prev_cp = cp

    # Add small noise
    values += np.random.normal(0, 2, n_days)
    values = np.maximum(values, 0)

    df = pd.DataFrame({"0": values}, index=dates)
    df.index.name = "target"

    # Save changepoints metadata
    metadata = pd.DataFrame({
        "changepoint_day": changepoints,
        "changepoint_date": [dates[cp] for cp in changepoints]
    })
    metadata.to_parquet(OUTPUT_DIR / "known_changepoints_metadata.parquet")

    return save_dataset(df, "known_changepoints", f"Data with known changepoints at days {changepoints}")


def generate_exponential_growth_cases():
    """
    Generate various exponential growth scenarios for ExponentialGrowth validation.
    """
    n_days = 100
    dates = create_date_index("2020-01-01", n_days)

    # Location 0: Clear exponential growth
    loc0 = 10 * np.exp(0.05 * np.arange(n_days))

    # Location 1: Exponential then plateau
    loc1 = np.concatenate([
        10 * np.exp(0.05 * np.arange(50)),
        np.ones(50) * 10 * np.exp(0.05 * 50)
    ])

    # Location 2: Linear growth (not exponential)
    loc2 = 10 + 5 * np.arange(n_days)

    # Location 3: Subexponential growth
    loc3 = 10 * np.power(np.arange(1, n_days + 1), 1.5)

    # Location 4: Declining
    loc4 = 500 * np.exp(-0.03 * np.arange(n_days))

    # Add noise
    loc0 = loc0.astype(float) + np.random.normal(0, np.abs(loc0) * 0.05)
    loc1 = loc1.astype(float) + np.random.normal(0, np.abs(loc1) * 0.05)
    loc2 = loc2.astype(float) + np.random.normal(0, np.abs(loc2) * 0.05)
    loc3 = loc3.astype(float) + np.random.normal(0, np.abs(loc3) * 0.05)
    loc4 = loc4.astype(float) + np.random.normal(0, np.abs(loc4) * 0.05)

    df = pd.DataFrame({
        "0": np.maximum(loc0, 0),
        "1": np.maximum(loc1, 0),
        "2": np.maximum(loc2, 0),
        "3": np.maximum(loc3, 0),
        "4": np.maximum(loc4, 0)
    }, index=dates)
    df.index.name = "target"

    return save_dataset(df, "exponential_growth_cases", "Various growth patterns for ExponentialGrowth testing")


def generate_shapelet_test_cases():
    """
    Generate data specifically designed to test Shapelet correlation.
    """
    n_days = 150
    dates = create_date_index("2020-01-01", n_days)

    # Location 0: Perfect exponential surge (should match shapelet perfectly)
    x = np.arange(n_days)
    loc0 = np.zeros(n_days)
    surge_start = 50
    for i in range(surge_start, surge_start + 30):
        loc0[i] = np.exp(0.2 * (i - surge_start))

    # Location 1: Inverted (decline) - should not match
    loc1 = np.zeros(n_days)
    for i in range(surge_start, surge_start + 30):
        loc1[i] = np.exp(-0.2 * (i - surge_start)) * 100

    # Location 2: Multiple small surges
    loc2 = np.zeros(n_days)
    for start in [20, 60, 100]:
        for i in range(start, start + 20):
            loc2[i] += np.exp(0.15 * (i - start))

    # Location 3: Noise only
    loc3 = np.abs(np.random.normal(10, 5, n_days))

    df = pd.DataFrame({
        "0": loc0,
        "1": loc1,
        "2": loc2,
        "3": loc3
    }, index=dates)
    df.index.name = "target"

    return save_dataset(df, "shapelet_test_cases", "Test cases for Shapelet correlation")


def generate_ensemble_test_cases():
    """
    Generate data where different algorithms should agree/disagree.
    """
    n_days = 200
    dates = create_date_index("2020-01-01", n_days)

    # Location 0: Clear outbreak - all algorithms should agree
    loc0 = np.ones(n_days) * 10
    loc0[50:100] = np.concatenate([
        np.linspace(10, 200, 25),
        np.linspace(200, 10, 25)
    ])

    # Location 1: Gradual change - BCP might detect, others might not
    loc1 = np.concatenate([
        np.ones(100) * 20,
        np.ones(100) * 60
    ])

    # Location 2: Noisy peak - algorithms should disagree
    loc2 = np.ones(n_days) * 50
    loc2 += np.random.normal(0, 20, n_days)
    loc2[80:120] += 50

    df = pd.DataFrame({
        "0": np.maximum(loc0, 0),
        "1": np.maximum(loc1, 0),
        "2": np.maximum(loc2, 0)
    }, index=dates)
    df.index.name = "target"

    return save_dataset(df, "ensemble_test_cases", "Cases for ensemble agreement testing")


def main():
    """Generate all test datasets."""
    print("Generating test datasets...")
    print("=" * 50)

    datasets = [
        generate_simple_wave,
        generate_multi_wave,
        generate_noisy_data,
        generate_edge_cases,
        generate_multi_location,
        generate_real_covid_sample,
        generate_short_series,
        generate_known_changepoints,
        generate_exponential_growth_cases,
        generate_shapelet_test_cases,
        generate_ensemble_test_cases,
    ]

    for generate_func in datasets:
        generate_func()

    print("=" * 50)
    print(f"Generated {len(datasets)} test datasets in {OUTPUT_DIR}")

    # Create summary
    summary = []
    for f in OUTPUT_DIR.glob("*.parquet"):
        if "metadata" not in f.name:
            df = pd.read_parquet(f)
            summary.append({
                "name": f.stem,
                "n_days": len(df),
                "n_locations": len(df.columns),
                "date_range": f"{df.index.min()} to {df.index.max()}"
            })

    summary_df = pd.DataFrame(summary)
    print("\nDataset Summary:")
    print(summary_df.to_string(index=False))


if __name__ == "__main__":
    main()
