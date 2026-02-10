"""
Pure Python implementation of Bayesian Change Point detection.

This implements the Barry & Hartigan (1993) product partition model using MCMC,
following the algorithm from the R bcp package (Erdman & Emerson, 2007).

The algorithm uses the B/W statistics (Between/Within block sum of squares)
and integrates out block means using Beta function formulas.

References:
- Barry, D. and Hartigan, J. A. (1993). A Bayesian Analysis for Change Point
  Problems. Journal of the American Statistical Association, 88(421), 309-319.
- Erdman, C. and Emerson, J. W. (2007). bcp: An R Package for Performing a
  Bayesian Analysis of Change Point Problems. Journal of Statistical Software,
  23(3), 1-13.
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
    using MCMC (Gibbs sampling), matching the R bcp package algorithm.

    The algorithm assumes data comes from blocks/segments with different means
    and unknown common variance. The key formula uses B/W statistics:
    - B = between-block sum of squares (sum of squared block means * block sizes)
    - W = within-block sum of squares (total SS - B)

    Parameters
    ----------
    p0 : float, default=0.2
        Prior probability of NO change point at each position.
        Higher values = fewer expected change points.

    w0 : float, default=0.2
        Prior parameter for the signal-to-noise ratio.
        Controls shrinkage of block means toward global mean.

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

        Implements the MCMC algorithm from Erdman & Emerson (2007).
        """
        n = len(x)
        kk = 1  # Univariate case

        # Precompute cumulative sums for O(1) block statistics
        # cumy[i] = sum of x[0:i+1]
        cumy = np.cumsum(x)
        # cumysq[i] = sum of x[0:i+1]^2
        cumysq = np.cumsum(x ** 2)

        # Global statistics
        global_mean = cumy[n - 1] / n
        total_ss = cumysq[n - 1]  # Sum of squares

        # Initialize: no change points (single block)
        rho = np.zeros(n, dtype=np.int32)
        rho[n - 1] = 1  # Last position always marked (convention)

        # Accumulators for posterior estimates
        rho_sum = np.zeros(n, dtype=np.float64)
        mean_sum = np.zeros(n, dtype=np.float64)

        total_iter = self.burnin + self.mcmc

        for iteration in range(total_iter):
            # Forward pass through all positions
            rho = self._pass(x, rho, cumy, cumysq, n, kk)

            # After burn-in, accumulate statistics
            if iteration >= self.burnin:
                rho_sum += rho

                # Compute block means for this iteration
                block_means = self._compute_block_means_with_shrinkage(
                    x, rho, cumy, global_mean, n
                )
                mean_sum += block_means

        # Compute posterior estimates
        n_samples = self.mcmc
        posterior_prob = rho_sum / n_samples
        posterior_mean = mean_sum / n_samples

        # First position can't be a changepoint
        posterior_prob[0] = 0.0

        return posterior_mean, posterior_prob

    def _pass(self, x: np.ndarray, rho: np.ndarray,
              cumy: np.ndarray, cumysq: np.ndarray,
              n: int, kk: int) -> np.ndarray:
        """
        One MCMC pass through all positions.

        For each position i (1 to n-2), we consider whether to place
        a changepoint there or not, based on the likelihood ratio.
        """
        rho_new = rho.copy()

        for i in range(1, n - 1):
            # Find previous and next changepoints
            prev_cp = 0
            for j in range(i - 1, -1, -1):
                if rho_new[j] == 1:
                    prev_cp = j + 1  # Block starts after changepoint
                    break

            next_cp = n
            for j in range(i + 1, n):
                if rho_new[j] == 1:
                    next_cp = j + 1  # Block ends at changepoint
                    break

            # Compute B and W for current configuration
            B_curr, W_curr, b_curr = self._compute_BW(
                rho_new, cumy, cumysq, n
            )

            # Try flipping rho[i]
            rho_test = rho_new.copy()
            rho_test[i] = 1 - rho_test[i]

            B_test, W_test, b_test = self._compute_BW(
                rho_test, cumy, cumysq, n
            )

            # Compute log likelihoods
            lik_curr = self._likelihood(B_curr, W_curr, b_curr, n, kk)
            lik_test = self._likelihood(B_test, W_test, b_test, n, kk)

            # Log odds of flipping
            log_odds = lik_test - lik_curr

            # Convert to probability
            if log_odds > 20:
                prob_flip = 1.0
            elif log_odds < -20:
                prob_flip = 0.0
            else:
                prob_flip = 1.0 / (1.0 + np.exp(-log_odds))

            # Sample
            if np.random.random() < prob_flip:
                rho_new[i] = rho_test[i]

        return rho_new

    def _compute_BW(self, rho: np.ndarray, cumy: np.ndarray,
                    cumysq: np.ndarray, n: int) -> Tuple[float, float, int]:
        """
        Compute B (between) and W (within) sum of squares.

        B = sum over blocks of (block_size * block_mean^2)
        W = total_SS - sum over blocks of (block_size * block_mean^2) + adjustment
            = sum over blocks of within-block SS

        Actually, following the C++ code more carefully:
        - B is the sum of weighted squared deviations of block means from 0
        - W is the residual sum of squares
        """
        # Find block boundaries
        boundaries = [0]
        for i in range(n):
            if rho[i] == 1:
                boundaries.append(i + 1)
        if boundaries[-1] != n:
            boundaries.append(n)

        b = len(boundaries) - 1  # Number of blocks

        # Compute B and W
        B = 0.0
        Z_total = 0.0  # Sum of (block_mean^2 * block_size)

        for k in range(b):
            start = boundaries[k]
            end = boundaries[k + 1]
            block_size = end - start

            if start > 0:
                block_sum = cumy[end - 1] - cumy[start - 1]
            else:
                block_sum = cumy[end - 1]

            block_mean = block_sum / block_size
            Z_total += block_mean ** 2 * block_size

        # Total sum of squares
        total_ss = cumysq[n - 1]

        # B = Z_total (sum of block_size * block_mean^2)
        # W = total_ss - Z_total (within-block SS)
        B = Z_total
        W = total_ss - Z_total

        # Ensure W is positive
        if W < 1e-10:
            W = 1e-10

        return B, W, b

    def _likelihood(self, B: float, W: float, b: int,
                    n: int, kk: int) -> float:
        """
        Compute log likelihood for a partition.

        This implements the key formula from the R bcp package:

        For B > 0:
            lik = priors[b] - (kk*b + 1)/2 * log(B)
                  - ((n - b)*kk - 2)/2 * log(W)
                  + log(I_xmax(alpha, beta))
                  + log(Beta(alpha, beta))

        where:
            xmax = B * w0 / W / (1 + B * w0 / W)
            alpha = (kk * b + 1) / 2
            beta = ((n - b) * kk - 2) / 2
            priors[b] = b * log(p0)

        The Beta function terms come from integrating out the unknown
        common variance under an improper prior.
        """
        w0 = self.w0
        p0 = self.p0

        # Prior on number of blocks (geometric prior on changepoint prob)
        prior = b * np.log(p0)

        # Parameters for the incomplete beta function
        alpha = (kk * b + 1) / 2.0
        beta_param = ((n - b) * kk - 2) / 2.0

        # Handle edge cases
        if b >= n - 4 / kk:
            return -np.inf

        if beta_param <= 0:
            return -np.inf

        if B <= 0 or B < 1e-10:
            # Special case when B = 0 (all block means equal)
            if b == 1:
                # Single block case
                lik = prior + (kk + 1) * np.log(w0) / 2 - (n * kk - 1) * np.log(W) / 2
            else:
                lik = prior + (kk * b + 1) * np.log(w0) / 2 - (n * kk - 1) * np.log(W) / 2 - np.log(alpha)
            return lik

        # Main formula with Beta function
        xmax = B * w0 / W / (1 + B * w0 / W)

        # Clamp xmax to valid range for beta function
        xmax = np.clip(xmax, 1e-15, 1 - 1e-15)

        # Log of regularized incomplete beta function I_x(a,b)
        # scipy.special.betainc(a, b, x) gives I_x(a,b)
        # We need log of this
        log_betainc = np.log(np.maximum(special.betainc(alpha, beta_param, xmax), 1e-300))

        # Log of beta function B(a, b)
        log_beta = special.betaln(alpha, beta_param)

        lik = (prior
               - (kk * b + 1) * np.log(B) / 2
               - ((n - b) * kk - 2) * np.log(W) / 2
               + log_betainc
               + log_beta)

        return lik

    def _compute_block_means_with_shrinkage(
            self, x: np.ndarray, rho: np.ndarray,
            cumy: np.ndarray, global_mean: float, n: int) -> np.ndarray:
        """
        Compute block means with shrinkage toward global mean.

        Following the C++ code, the posterior mean incorporates
        shrinkage based on wstar.
        """
        means = np.zeros(n)

        # Find block boundaries
        boundaries = [0]
        for i in range(n):
            if rho[i] == 1:
                boundaries.append(i + 1)
        if boundaries[-1] != n:
            boundaries.append(n)

        b = len(boundaries) - 1

        # Compute wstar for shrinkage (simplified version)
        # In the full algorithm, wstar depends on B and W
        B, W, _ = self._compute_BW(rho, cumy, np.cumsum(x ** 2), n)

        if b > 1 and B > 0:
            xmax = B * self.w0 / W / (1 + B * self.w0 / W)
            xmax = np.clip(xmax, 1e-15, 1 - 1e-15)

            alpha1 = (b + 3) / 2
            beta1 = (n - b - 4) / 2
            alpha2 = (b + 1) / 2
            beta2 = (n - b - 2) / 2

            if beta1 > 0 and beta2 > 0:
                log_wstar = (np.log(W) - np.log(B)
                             + special.betaln(alpha1, beta1)
                             + np.log(np.maximum(special.betainc(alpha1, beta1, xmax), 1e-300))
                             - special.betaln(alpha2, beta2)
                             - np.log(np.maximum(special.betainc(alpha2, beta2, xmax), 1e-300)))
                wstar = np.exp(np.clip(log_wstar, -50, 50))
                wstar = np.clip(wstar, 0, 1)
            else:
                wstar = self.w0 / 2
        else:
            wstar = self.w0 / 2

        # Compute mean for each block with shrinkage
        for k in range(b):
            start = boundaries[k]
            end = boundaries[k + 1]
            block_size = end - start

            if start > 0:
                block_sum = cumy[end - 1] - cumy[start - 1]
            else:
                block_sum = cumy[end - 1]

            block_mean = block_sum / block_size

            # Shrink toward global mean
            shrunk_mean = block_mean * (1 - wstar) + global_mean * wstar
            means[start:end] = shrunk_mean

        return means


def bcp(x: np.ndarray, p0: float = 0.2, w0: float = 0.2,
        burnin: int = 50, mcmc: int = 500) -> BcpResult:
    """
    Bayesian Change Point detection using MCMC.

    This function implements the Barry & Hartigan (1993) product partition
    model using Gibbs sampling, matching the R bcp package algorithm.

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
