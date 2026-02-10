#!/usr/bin/env python
"""
Compare Python BCP implementation with R BCP implementation.

This script creates visualizations comparing the two implementations
on various test cases.
"""

import numpy as np
import matplotlib.pyplot as plt
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from epylabel.bcp_python import bcp as bcp_python

# Check if R is available
R_AVAILABLE = False
try:
    import rpy2.robjects as robjects
    import rpy2.robjects.packages as rpackages
    bcp_r_pkg = rpackages.importr("bcp")
    R_AVAILABLE = True
    print("R/rpy2 available - will compare with R implementation")
except Exception as e:
    print(f"R/rpy2 not available ({e})")
    print("Will only show Python implementation results")


def run_r_bcp(data, p0=0.2):
    """Run R BCP and return results."""
    if not R_AVAILABLE:
        return None, None

    x_r = robjects.FloatVector(data)
    result = bcp_r_pkg.bcp(x_r, p0=p0)

    posterior_mean = np.array(result.rx2["posterior.mean"]).flatten()
    posterior_prob = np.array(result.rx2["posterior.prob"]).flatten()

    return posterior_mean, posterior_prob


def compare_step_function():
    """Compare on step function with single changepoint."""
    np.random.seed(42)

    # Clean step function
    data = np.concatenate([
        np.ones(30) * 10,
        np.ones(30) * 50,
    ])

    # Python result
    result_py = bcp_python(data, p0=0.2, burnin=100, mcmc=500)

    # R result (if available)
    mean_r, prob_r = run_r_bcp(data, p0=0.2)

    # Create plot
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Data
    ax = axes[0, 0]
    ax.plot(data, 'b-', linewidth=1.5, label='Data')
    ax.axvline(30, color='r', linestyle='--', label='True changepoint')
    ax.set_xlabel('Position')
    ax.set_ylabel('Value')
    ax.set_title('Step Function Data')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Posterior probability
    ax = axes[0, 1]
    ax.plot(result_py.posterior_prob, 'g-', linewidth=1.5, label='Python')
    if prob_r is not None:
        ax.plot(prob_r, 'b--', linewidth=1.5, label='R')
    ax.axvline(30, color='r', linestyle='--', alpha=0.5, label='True changepoint')
    ax.set_xlabel('Position')
    ax.set_ylabel('Posterior Probability')
    ax.set_title('Changepoint Posterior Probability')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim(-0.05, 1.05)

    # Posterior mean
    ax = axes[1, 0]
    ax.plot(data, 'b-', linewidth=1, alpha=0.5, label='Data')
    ax.plot(result_py.posterior_mean, 'g-', linewidth=2, label='Python posterior mean')
    if mean_r is not None:
        ax.plot(mean_r, 'r--', linewidth=2, label='R posterior mean')
    ax.set_xlabel('Position')
    ax.set_ylabel('Value')
    ax.set_title('Posterior Mean')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Summary statistics
    ax = axes[1, 1]
    ax.axis('off')

    py_high_prob = np.where(result_py.posterior_prob > 0.3)[0]

    summary_text = f"""Summary Statistics:

Python Implementation:
  Mean posterior prob: {np.mean(result_py.posterior_prob):.4f}
  Max posterior prob: {np.max(result_py.posterior_prob):.4f}
  Positions with prob > 0.3: {py_high_prob}
  Expected changepoint: 30
"""

    if prob_r is not None:
        r_high_prob = np.where(prob_r > 0.3)[0]
        summary_text += f"""
R Implementation:
  Mean posterior prob: {np.mean(prob_r):.4f}
  Max posterior prob: {np.max(prob_r):.4f}
  Positions with prob > 0.3: {r_high_prob}
"""

    ax.text(0.1, 0.5, summary_text, fontsize=12, family='monospace',
            verticalalignment='center', transform=ax.transAxes)

    plt.suptitle('BCP Comparison: Step Function', fontsize=14)
    plt.tight_layout()

    return fig


def compare_multiple_changepoints():
    """Compare on data with multiple changepoints."""
    np.random.seed(42)

    # Multiple changepoints
    data = np.concatenate([
        np.ones(25) * 10,
        np.ones(25) * 50,
        np.ones(25) * 25,
        np.ones(25) * 80,
    ])

    # Python result
    result_py = bcp_python(data, p0=0.2, burnin=100, mcmc=500)

    # R result
    mean_r, prob_r = run_r_bcp(data, p0=0.2)

    # Create plot
    fig, axes = plt.subplots(2, 1, figsize=(14, 8))

    # Data and posterior prob
    ax = axes[0]
    ax.plot(data, 'b-', linewidth=1.5, label='Data')
    for cp in [25, 50, 75]:
        ax.axvline(cp, color='r', linestyle='--', alpha=0.5)
    ax.set_xlabel('Position')
    ax.set_ylabel('Value')
    ax.set_title('Data with Multiple Changepoints (at 25, 50, 75)')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Posterior probability
    ax = axes[1]
    ax.plot(result_py.posterior_prob, 'g-', linewidth=2, label='Python')
    if prob_r is not None:
        ax.plot(prob_r, 'b--', linewidth=2, label='R')
    for cp in [25, 50, 75]:
        ax.axvline(cp, color='r', linestyle='--', alpha=0.5, label='True changepoint' if cp == 25 else '')
    ax.set_xlabel('Position')
    ax.set_ylabel('Posterior Probability')
    ax.set_title('Changepoint Posterior Probability')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim(-0.05, 1.05)

    plt.tight_layout()

    return fig


def compare_noisy_data():
    """Compare on noisy data with changepoint."""
    np.random.seed(42)

    # Noisy step function
    noise_level = 5
    data = np.concatenate([
        np.random.normal(10, noise_level, 40),
        np.random.normal(50, noise_level, 40),
    ])

    # Python result
    result_py = bcp_python(data, p0=0.2, burnin=100, mcmc=500)

    # R result
    mean_r, prob_r = run_r_bcp(data, p0=0.2)

    # Create plot
    fig, axes = plt.subplots(2, 1, figsize=(14, 8))

    # Data
    ax = axes[0]
    ax.plot(data, 'b.', alpha=0.5, label='Data')
    ax.plot(result_py.posterior_mean, 'g-', linewidth=2, label='Python posterior mean')
    if mean_r is not None:
        ax.plot(mean_r, 'r--', linewidth=2, label='R posterior mean')
    ax.axvline(40, color='r', linestyle='--', alpha=0.5)
    ax.set_xlabel('Position')
    ax.set_ylabel('Value')
    ax.set_title('Noisy Step Function (noise=5, changepoint at 40)')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Posterior probability
    ax = axes[1]
    ax.plot(result_py.posterior_prob, 'g-', linewidth=2, label='Python')
    if prob_r is not None:
        ax.plot(prob_r, 'b--', linewidth=2, label='R')
    ax.axvline(40, color='r', linestyle='--', alpha=0.5, label='True changepoint')
    ax.set_xlabel('Position')
    ax.set_ylabel('Posterior Probability')
    ax.set_title('Changepoint Posterior Probability')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim(-0.05, 1.05)

    plt.tight_layout()

    return fig


def compare_correlation():
    """Create scatter plots comparing Python and R outputs."""
    if not R_AVAILABLE:
        print("Skipping correlation plot - R not available")
        return None

    np.random.seed(42)

    # Multiple test cases
    test_data = [
        np.concatenate([np.ones(30) * 10, np.ones(30) * 50]),
        np.concatenate([np.ones(20) * 0, np.ones(20) * 100, np.ones(20) * 50]),
        np.random.normal(0, 1, 60),  # No changepoint
        np.concatenate([np.random.normal(10, 2, 30), np.random.normal(50, 2, 30)]),
    ]

    all_py_prob = []
    all_r_prob = []
    all_py_mean = []
    all_r_mean = []

    for data in test_data:
        result_py = bcp_python(data, p0=0.2, burnin=100, mcmc=500)
        mean_r, prob_r = run_r_bcp(data, p0=0.2)

        all_py_prob.extend(result_py.posterior_prob)
        all_r_prob.extend(prob_r)
        all_py_mean.extend(result_py.posterior_mean)
        all_r_mean.extend(mean_r)

    all_py_prob = np.array(all_py_prob)
    all_r_prob = np.array(all_r_prob)
    all_py_mean = np.array(all_py_mean)
    all_r_mean = np.array(all_r_mean)

    # Create correlation plots
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Posterior probability correlation
    ax = axes[0]
    ax.scatter(all_r_prob, all_py_prob, alpha=0.5, s=20)
    ax.plot([0, 1], [0, 1], 'r--', label='y=x')
    ax.set_xlabel('R Posterior Probability')
    ax.set_ylabel('Python Posterior Probability')
    ax.set_title('Posterior Probability: Python vs R')

    # Compute correlation
    corr = np.corrcoef(all_r_prob, all_py_prob)[0, 1]
    ax.text(0.05, 0.95, f'Correlation: {corr:.3f}', transform=ax.transAxes, fontsize=12)
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Posterior mean correlation
    ax = axes[1]
    ax.scatter(all_r_mean, all_py_mean, alpha=0.5, s=20)

    # Identity line
    min_val = min(all_r_mean.min(), all_py_mean.min())
    max_val = max(all_r_mean.max(), all_py_mean.max())
    ax.plot([min_val, max_val], [min_val, max_val], 'r--', label='y=x')

    ax.set_xlabel('R Posterior Mean')
    ax.set_ylabel('Python Posterior Mean')
    ax.set_title('Posterior Mean: Python vs R')

    # Compute correlation
    corr = np.corrcoef(all_r_mean, all_py_mean)[0, 1]
    ax.text(0.05, 0.95, f'Correlation: {corr:.3f}', transform=ax.transAxes, fontsize=12)
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    return fig


def main():
    output_dir = Path(__file__).parent.parent / "output"
    output_dir.mkdir(exist_ok=True)

    print("\n" + "="*60)
    print("BCP Implementation Comparison")
    print("="*60)

    # Step function comparison
    print("\n1. Step function comparison...")
    fig1 = compare_step_function()
    fig1.savefig(output_dir / "bcp_comparison_step.png", dpi=150, bbox_inches='tight')
    print(f"   Saved to {output_dir / 'bcp_comparison_step.png'}")

    # Multiple changepoints
    print("\n2. Multiple changepoints comparison...")
    fig2 = compare_multiple_changepoints()
    fig2.savefig(output_dir / "bcp_comparison_multiple.png", dpi=150, bbox_inches='tight')
    print(f"   Saved to {output_dir / 'bcp_comparison_multiple.png'}")

    # Noisy data
    print("\n3. Noisy data comparison...")
    fig3 = compare_noisy_data()
    fig3.savefig(output_dir / "bcp_comparison_noisy.png", dpi=150, bbox_inches='tight')
    print(f"   Saved to {output_dir / 'bcp_comparison_noisy.png'}")

    # Correlation plot
    print("\n4. Correlation analysis...")
    fig4 = compare_correlation()
    if fig4 is not None:
        fig4.savefig(output_dir / "bcp_comparison_correlation.png", dpi=150, bbox_inches='tight')
        print(f"   Saved to {output_dir / 'bcp_comparison_correlation.png'}")

    print("\n" + "="*60)
    print("Done! All comparison plots saved to output/")
    print("="*60)

    plt.close('all')


if __name__ == "__main__":
    main()
