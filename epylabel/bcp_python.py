"""
Pure Python implementation of Bayesian Change Point detection.

This is a simplified Python implementation inspired by the Barry & Hartigan (1993)
product partition model. It provides a pure Python alternative when R is not
available, but does NOT produce identical results to the R bcp package.

For exact reproducibility of R bcp results, use the R implementation via rpy2.
This Python version is suitable for:
- Environments where R cannot be installed
- Quick prototyping and testing
- Educational purposes

The algorithm uses MCMC (Gibbs sampling) to estimate posterior probabilities
of change points. Results will be qualitatively similar but numerically different
from the R bcp package due to:
- Different prior parameterization
- Different MCMC implementation details
- Different numerical optimizations

References:
- Barry, D. and Hartigan, J. A. (1993). A Bayesian Analysis for Change Point
  Problems. Journal of the American Statistical Association, 88(421), 309-319.
- Erdman, C. and Emerson, J. W. (2007). bcp: An R Package for Performing a
  Bayesian Analysis of Change Point Problems. Journal of Statistical Software,
  23(3), 1-13.

Note: For production use requiring exact R bcp compatibility, use:
    from epylabel.labeler import Bcp
    bcp = Bcp(d=..., p0=..., thresh=..., use_python=False)  # Uses R
"""

import numpy as np
from scipy import special
from typing import Tuple, Optional
from dataclasses import dataclass


@dataclass
class BcpResult:
    """Result of Bayesian Change Point detection."""
    posterior_mean: np.ndarray  # Posterior mean of segment means
    posterior_prob: np.ndarray  # Posterior probability of changepoint at each position
    blocks: np.ndarray  # Block assignments (optional)


class BcpPython:
    """
    Pure Python implementation of Bayesian Change Point detection.

    This class implements the Barry & Hartigan (1993) product partition model
    using MCMC (Gibbs sampling), similar to the R bcp package.

    The algorithm assumes data comes from blocks/segments with different means
    and unknown common variance. Within each block, observations are i.i.d.
    normal. The prior probability of a change point at any position is (1-p0).

    Parameters
    ----------
    p0 : float, default=0.2
        Prior probability of NO change point at each position.
        Higher values = fewer expected change points.

    w0 : float, default=0.2
        Prior parameter for the signal-to-noise ratio.
        Controls how much the block means can vary.

    burnin : int, default=50
        Number of MCMC burn-in iterations.

    mcmc : int, default=500
        Number of MCMC iterations after burn-in.
    """

    def __init__(self, p0: float = 0.2, w0: float = 0.2,
                 burnin: int = 50, mcmc: int = 500):
        self.p0 = p0
        self.w0 = w0
        self.burnin = burnin
        self.mcmc = mcmc

    def fit(self, x: np.ndarray) -> BcpResult:
        """
        Detect change points in the data using MCMC.

        Parameters
        ----------
        x : np.ndarray
            1D array of observations.

        Returns
        -------
        BcpResult
            Object containing posterior_mean and posterior_prob arrays.
        """
        x = np.asarray(x, dtype=np.float64)
        n = len(x)

        if n < 2:
            return BcpResult(
                posterior_mean=x.copy(),
                posterior_prob=np.zeros(n),
                blocks=np.zeros(n, dtype=int)
            )

        # Run MCMC
        posterior_mean, posterior_prob = self._mcmc_gibbs(x)

        return BcpResult(
            posterior_mean=posterior_mean,
            posterior_prob=posterior_prob,
            blocks=np.zeros(n, dtype=int)
        )

    def _mcmc_gibbs(self, x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Run Gibbs sampling MCMC for change point detection.

        This implementation follows the approach of the R bcp package,
        using a global variance estimate and the Barry & Hartigan (1993)
        product partition model.
        """
        n = len(x)

        # Precompute sufficient statistics for O(1) block likelihood
        cumsum = np.zeros(n + 1)
        cumsum_sq = np.zeros(n + 1)
        cumsum[1:] = np.cumsum(x)
        cumsum_sq[1:] = np.cumsum(x ** 2)

        # Estimate global variance from data (pooled estimate)
        # This is key to matching R's behavior
        global_mean = cumsum[n] / n
        global_ss = cumsum_sq[n] - n * global_mean ** 2
        self._sigma2 = global_ss / (n - 1) if n > 1 else 1.0
        if self._sigma2 < 1e-10:
            self._sigma2 = 1.0

        # Initialize: no change points (single block)
        rho = np.zeros(n, dtype=np.int32)

        # Accumulators for posterior estimates
        rho_sum = np.zeros(n, dtype=np.float64)
        mean_sum = np.zeros(n, dtype=np.float64)

        total_iter = self.burnin + self.mcmc

        for iteration in range(total_iter):
            # Gibbs update: sample each position sequentially
            for i in range(1, n):
                rho[i] = self._sample_rho_i(i, x, rho, cumsum, cumsum_sq)

            # After burn-in, accumulate statistics
            if iteration >= self.burnin:
                rho_sum += rho
                mean_sum += self._compute_block_means(n, rho, cumsum)

        # Compute posterior estimates
        n_samples = self.mcmc
        posterior_prob = rho_sum / n_samples
        posterior_mean = mean_sum / n_samples

        return posterior_mean, posterior_prob

    def _sample_rho_i(self, i: int, x: np.ndarray, rho: np.ndarray,
                      cumsum: np.ndarray, cumsum_sq: np.ndarray) -> int:
        """
        Sample rho[i] from its conditional distribution.

        Uses the Bayes factor approach with known variance.
        """
        n = len(x)

        # Find the previous and next change points
        prev_cp = 0
        for j in range(i - 1, -1, -1):
            if rho[j] == 1 or j == 0:
                prev_cp = j
                break

        next_cp = n
        for j in range(i + 1, n):
            if rho[j] == 1:
                next_cp = j
                break

        # Compute log marginal likelihoods for both cases
        # Case 0: No change point at i (one block from prev_cp to next_cp)
        log_ml_0 = self._log_marginal_likelihood(cumsum, cumsum_sq, prev_cp, next_cp)

        # Case 1: Change point at i (two blocks)
        log_ml_1 = (self._log_marginal_likelihood(cumsum, cumsum_sq, prev_cp, i) +
                    self._log_marginal_likelihood(cumsum, cumsum_sq, i, next_cp))

        # Prior log-odds: p0 = prob of NO change, (1-p0) = prob of change
        log_prior_0 = np.log(self.p0)
        log_prior_1 = np.log(1 - self.p0)

        # Conditional probability of rho[i] = 1
        log_odds = (log_ml_1 + log_prior_1) - (log_ml_0 + log_prior_0)

        # Convert to probability with numerical stability
        if log_odds > 20:
            prob_1 = 1.0
        elif log_odds < -20:
            prob_1 = 0.0
        else:
            prob_1 = 1.0 / (1.0 + np.exp(-log_odds))

        # Sample
        return 1 if np.random.random() < prob_1 else 0

    def _log_marginal_likelihood(self, cumsum: np.ndarray, cumsum_sq: np.ndarray,
                                  start: int, end: int) -> float:
        """
        Compute log marginal likelihood for a block [start, end).

        Uses the normal model with known variance (estimated globally)
        and integrates out the unknown block mean with a normal prior.

        For n observations in a block with known variance σ²:
        - Prior on mean: μ ~ N(0, σ²/w0)  [vague prior as w0 -> 0]
        - Likelihood: y_i | μ ~ N(μ, σ²)

        Marginal likelihood integrating out μ:
        log p(y) = -n/2 * log(2πσ²) - SS/(2σ²) - (y_bar)² * n * w0 / (2σ² * (1 + n*w0))
                   - 0.5 * log(1 + n * w0)
        """
        block_n = end - start

        if block_n <= 0:
            return 0.0

        # Sufficient statistics
        block_sum = cumsum[end] - cumsum[start]
        block_sum_sq = cumsum_sq[end] - cumsum_sq[start]

        block_mean = block_sum / block_n
        ss = block_sum_sq - block_n * block_mean ** 2
        ss = max(ss, 0.0)

        sigma2 = self._sigma2
        w0 = self.w0

        # Log marginal likelihood with conjugate normal prior on mean
        # Using the formula for integrating out the mean
        #
        # The key term that distinguishes changepoints is:
        # -SS/(2σ²) - 0.5 * log(1 + n*w0) + correction for mean prior

        # Precision-weighted terms
        prior_precision = w0 / sigma2 if w0 > 0 else 0.0
        data_precision = block_n / sigma2

        log_ml = (
            - 0.5 * block_n * np.log(2 * np.pi * sigma2)  # Normalization
            - ss / (2 * sigma2)  # Data fit (deviations from block mean)
            - 0.5 * np.log(1 + block_n * w0)  # Prior weight adjustment
        )

        return log_ml

    def _compute_block_means(self, n: int, rho: np.ndarray,
                             cumsum: np.ndarray) -> np.ndarray:
        """
        Compute block means given the current partition.

        Each position gets the mean of its block.
        """
        means = np.zeros(n)

        # Find block boundaries
        boundaries = [0]
        for i in range(1, n):
            if rho[i] == 1:
                boundaries.append(i)
        boundaries.append(n)

        # Compute mean for each block
        for b in range(len(boundaries) - 1):
            start = boundaries[b]
            end = boundaries[b + 1]
            block_sum = cumsum[end] - cumsum[start]
            block_n = end - start
            means[start:end] = block_sum / block_n

        return means


def bcp(x: np.ndarray, p0: float = 0.2, w0: float = 0.2,
        burnin: int = 50, mcmc: int = 500) -> BcpResult:
    """
    Bayesian Change Point detection using MCMC.

    This function implements the Barry & Hartigan (1993) product partition
    model using Gibbs sampling, similar to the R bcp package.

    Parameters
    ----------
    x : np.ndarray
        1D array of observations.
    p0 : float, default=0.2
        Prior probability of no change point at each position.
    w0 : float, default=0.2
        Prior parameter for signal-to-noise ratio.
    burnin : int, default=50
        Number of MCMC burn-in iterations.
    mcmc : int, default=500
        Number of MCMC iterations after burn-in.

    Returns
    -------
    BcpResult
        Object with posterior_mean and posterior_prob arrays.

    References
    ----------
    Barry, D. and Hartigan, J. A. (1993). A Bayesian Analysis for Change
    Point Problems. JASA, 88(421), 309-319.

    Erdman, C. and Emerson, J. W. (2007). bcp: An R Package for Performing
    a Bayesian Analysis of Change Point Problems. JSS, 23(3), 1-13.
    """
    model = BcpPython(p0=p0, w0=w0, burnin=burnin, mcmc=mcmc)
    return model.fit(x)
