"""
Pure Python implementation of Bayesian Change Point detection.

Based on Barry & Hartigan (1993) "A Bayesian Analysis for Change Point Problems"
Journal of the American Statistical Association, 88(421), 309-319.

This implementation uses the exact recursive algorithm to compute posterior
probabilities of change points and posterior means of segment parameters.
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

    This class implements the Barry & Hartigan (1993) algorithm for
    detecting change points in univariate time series data using the
    exact recursive algorithm.

    The algorithm uses a product partition model where:
    - Data is assumed to come from blocks/segments with different means
    - Within each block, observations are i.i.d. normal with unknown mean and variance
    - The prior probability of a change point at any position is (1 - p0)

    Parameters
    ----------
    p0 : float, default=0.2
        Prior probability of NO change point at each position.
        Higher values = fewer expected change points.

    w0 : float, optional
        Prior weight parameter. If None, computed from data.

    burnin : int, default=50
        Number of MCMC burn-in iterations (for compatibility, used in MCMC mode).

    mcmc : int, default=500
        Number of MCMC iterations (for compatibility, used in MCMC mode).

    use_exact : bool, default=True
        If True, use exact recursive algorithm. If False, use MCMC.
    """

    def __init__(self, p0: float = 0.2, w0: Optional[float] = None,
                 burnin: int = 50, mcmc: int = 500, use_exact: bool = True):
        self.p0 = p0
        self.w0 = w0
        self.burnin = burnin
        self.mcmc = mcmc
        self.use_exact = use_exact

    def fit(self, x: np.ndarray) -> BcpResult:
        """
        Detect change points in the data.

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

        if self.use_exact:
            posterior_mean, posterior_prob = self._exact_algorithm(x)
        else:
            posterior_mean, posterior_prob = self._mcmc_sample(x)

        return BcpResult(
            posterior_mean=posterior_mean,
            posterior_prob=posterior_prob,
            blocks=np.zeros(n, dtype=int)
        )

    def _exact_algorithm(self, x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Exact recursive algorithm for computing posterior probabilities.

        Uses dynamic programming to compute:
        1. Block log-likelihoods B[i,j] for all pairs
        2. Forward recursion for partition function
        3. Backward recursion for posterior probabilities
        """
        n = len(x)

        # Compute global statistics for prior
        x_var = np.var(x, ddof=1) if n > 1 else 1.0
        if x_var < 1e-10:
            x_var = 1.0

        # w0 parameter (prior weight on mean)
        w0 = self.w0 if self.w0 is not None else 0.2

        # Precompute sufficient statistics
        cumsum_x = np.zeros(n + 1)
        cumsum_x2 = np.zeros(n + 1)
        cumsum_x[1:] = np.cumsum(x)
        cumsum_x2[1:] = np.cumsum(x ** 2)

        # Compute block log-likelihoods B[i, j] for block from i to j (inclusive)
        # Use log scale to avoid numerical issues
        log_B = self._compute_block_loglik_matrix(cumsum_x, cumsum_x2, n, w0)

        # Prior log-probability of change (1 - p0) vs no change (p0)
        log_p = np.log(1 - self.p0 + 1e-300)  # log prob of changepoint
        log_q = np.log(self.p0 + 1e-300)      # log prob of no changepoint

        # Forward recursion: compute log P(y_1:i) for each i
        # log_alpha[i] = log P(y_0:i)
        log_alpha = np.full(n + 1, -np.inf)
        log_alpha[0] = 0.0  # Empty sequence has probability 1

        for i in range(1, n + 1):
            # Sum over all possible last changepoints
            terms = []
            for j in range(i):
                # Block from j to i-1, changepoint at j
                # P(y_0:i) = sum_j P(y_0:j) * P(changepoint at j) * P(y_j:i-1 | one block)
                if j == 0:
                    log_term = log_B[0, i - 1]  # First block, no prior changepoint
                else:
                    log_term = log_alpha[j] + log_p + log_B[j, i - 1]
                terms.append(log_term)

            log_alpha[i] = _logsumexp(np.array(terms))

        # Backward recursion: compute log P(y_i:n | changepoint at i)
        # log_beta[i] = log P(y_i:n-1)
        log_beta = np.full(n + 1, -np.inf)
        log_beta[n] = 0.0  # Empty sequence has probability 1

        for i in range(n - 1, -1, -1):
            terms = []
            for j in range(i + 1, n + 1):
                # Block from i to j-1
                if j == n:
                    log_term = log_B[i, n - 1]  # Last block, no following changepoint
                else:
                    log_term = log_B[i, j - 1] + log_p + log_beta[j]
                terms.append(log_term)

            log_beta[i] = _logsumexp(np.array(terms))

        # Compute posterior probability of changepoint at each position
        # P(changepoint at i | y) = P(y_0:i) * P(y_i:n) / P(y)
        log_evidence = log_alpha[n]
        posterior_prob = np.zeros(n)

        for i in range(1, n):
            # Probability of changepoint at position i
            log_prob = log_alpha[i] + log_p + log_beta[i] - log_evidence
            posterior_prob[i] = np.exp(np.clip(log_prob, -700, 0))

        # Compute posterior means using weighted block means
        posterior_mean = self._compute_posterior_means(
            x, cumsum_x, log_alpha, log_beta, log_B, log_p, log_evidence, n
        )

        return posterior_mean, posterior_prob

    def _compute_block_loglik_matrix(self, cumsum_x: np.ndarray,
                                      cumsum_x2: np.ndarray,
                                      n: int, w0: float) -> np.ndarray:
        """
        Compute matrix of block log-likelihoods.

        log_B[i, j] = log marginal likelihood of data from index i to j (inclusive).
        """
        log_B = np.full((n, n), -np.inf)

        for i in range(n):
            for j in range(i, n):
                log_B[i, j] = self._block_marginal_loglik(
                    cumsum_x, cumsum_x2, i, j + 1, w0
                )

        return log_B

    def _block_marginal_loglik(self, cumsum_x: np.ndarray, cumsum_x2: np.ndarray,
                                start: int, end: int, w0: float) -> float:
        """
        Compute log marginal likelihood for a block [start, end).

        Uses the normal model with unknown mean and variance,
        integrating out both parameters with conjugate priors.
        """
        block_n = end - start
        if block_n <= 0:
            return 0.0

        # Sufficient statistics
        block_sum = cumsum_x[end] - cumsum_x[start]
        block_sum2 = cumsum_x2[end] - cumsum_x2[start]

        block_mean = block_sum / block_n
        block_ss = block_sum2 - block_n * block_mean ** 2
        block_ss = max(block_ss, 1e-10)

        if block_n == 1:
            # Single observation: marginal likelihood is just the prior
            return -0.5 * np.log(2 * np.pi) - 0.5 * np.log(1 + w0)

        # Marginal likelihood for normal model with unknown mean and variance
        # Using the standard result for normal-inverse-gamma conjugate prior
        # log p(y) = log Gamma((n-1)/2) - log Gamma(1/2)
        #          - (n-1)/2 * log(pi)
        #          - 0.5 * log(n)
        #          - (n-1)/2 * log(SS)

        nu = block_n - 1  # Degrees of freedom

        loglik = (
            special.gammaln(0.5 * nu)
            - special.gammaln(0.5)
            - 0.5 * nu * np.log(np.pi)
            - 0.5 * np.log(block_n + w0)
            - 0.5 * nu * np.log(block_ss)
        )

        return loglik

    def _compute_posterior_means(self, x: np.ndarray, cumsum_x: np.ndarray,
                                  log_alpha: np.ndarray, log_beta: np.ndarray,
                                  log_B: np.ndarray, log_p: float,
                                  log_evidence: float, n: int) -> np.ndarray:
        """
        Compute posterior means for each position.

        The posterior mean at position i is the expected value of the
        block mean, weighted by the posterior probability of each partition.
        """
        posterior_mean = np.zeros(n)

        # For each position, compute weighted mean over all possible blocks
        for k in range(n):
            weighted_sum = 0.0
            weight_total = 0.0

            # Consider all blocks that contain position k
            for i in range(k + 1):  # Block start
                for j in range(k, n):  # Block end
                    # Log probability of this block configuration
                    if i == 0:
                        log_left = 0.0
                    else:
                        log_left = log_alpha[i] + log_p

                    if j == n - 1:
                        log_right = 0.0
                    else:
                        log_right = log_p + log_beta[j + 1]

                    log_prob = log_left + log_B[i, j] + log_right - log_evidence
                    prob = np.exp(np.clip(log_prob, -700, 0))

                    # Block mean
                    block_sum = cumsum_x[j + 1] - cumsum_x[i]
                    block_n = j - i + 1
                    block_mean = block_sum / block_n

                    weighted_sum += prob * block_mean
                    weight_total += prob

            if weight_total > 1e-10:
                posterior_mean[k] = weighted_sum / weight_total
            else:
                posterior_mean[k] = x[k]

        return posterior_mean

    def _mcmc_sample(self, x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        MCMC sampling algorithm (backup method).
        """
        n = len(x)
        w0 = self.w0 if self.w0 is not None else 0.2

        cumsum_x = np.zeros(n + 1)
        cumsum_x2 = np.zeros(n + 1)
        cumsum_x[1:] = np.cumsum(x)
        cumsum_x2[1:] = np.cumsum(x ** 2)

        # Initialize with no changepoints
        rho = np.zeros(n, dtype=int)
        rho[0] = 1

        posterior_prob_sum = np.zeros(n)
        posterior_mean_sum = np.zeros(n)
        n_samples = 0

        total_iter = self.burnin + self.mcmc

        for iteration in range(total_iter):
            rho = self._gibbs_update(x, rho, cumsum_x, cumsum_x2, w0)

            if iteration >= self.burnin:
                posterior_prob_sum += rho
                posterior_mean_sum += self._get_block_means(x, rho, cumsum_x)
                n_samples += 1

        posterior_prob = posterior_prob_sum / n_samples
        posterior_mean = posterior_mean_sum / n_samples
        posterior_prob[0] = 0.0

        return posterior_mean, posterior_prob

    def _gibbs_update(self, x: np.ndarray, rho: np.ndarray,
                      cumsum_x: np.ndarray, cumsum_x2: np.ndarray,
                      w0: float) -> np.ndarray:
        """Single Gibbs sampling update."""
        n = len(x)
        rho_new = rho.copy()

        for i in range(1, n):
            prev_cp = 0
            for j in range(i - 1, -1, -1):
                if rho_new[j] == 1:
                    prev_cp = j
                    break

            next_cp = n
            for j in range(i + 1, n):
                if rho_new[j] == 1:
                    next_cp = j
                    break

            # No changepoint: one block from prev_cp to next_cp
            log_ml_0 = self._block_marginal_loglik(cumsum_x, cumsum_x2, prev_cp, next_cp, w0)

            # Changepoint at i: two blocks
            log_ml_1 = (
                self._block_marginal_loglik(cumsum_x, cumsum_x2, prev_cp, i, w0) +
                self._block_marginal_loglik(cumsum_x, cumsum_x2, i, next_cp, w0)
            )

            log_prior_0 = np.log(self.p0 + 1e-300)
            log_prior_1 = np.log(1 - self.p0 + 1e-300)

            log_post_0 = log_ml_0 + log_prior_0
            log_post_1 = log_ml_1 + log_prior_1

            prob_1 = 1.0 / (1.0 + np.exp(log_post_0 - log_post_1))
            prob_1 = np.clip(prob_1, 0.0, 1.0)
            if np.isnan(prob_1):
                prob_1 = 0.5

            rho_new[i] = 1 if np.random.random() < prob_1 else 0

        return rho_new

    def _get_block_means(self, x: np.ndarray, rho: np.ndarray,
                         cumsum_x: np.ndarray) -> np.ndarray:
        """Compute block means given partition."""
        n = len(x)
        means = np.zeros(n)
        cps = np.where(rho == 1)[0]
        boundaries = np.concatenate([cps, [n]])

        for i in range(len(boundaries) - 1):
            start = boundaries[i]
            end = boundaries[i + 1]
            if end > start:
                means[start:end] = (cumsum_x[end] - cumsum_x[start]) / (end - start)

        return means


def _logsumexp(log_values: np.ndarray) -> float:
    """Compute log(sum(exp(log_values))) in a numerically stable way."""
    if len(log_values) == 0:
        return -np.inf
    max_val = np.max(log_values)
    if np.isinf(max_val):
        return -np.inf
    return max_val + np.log(np.sum(np.exp(log_values - max_val)))


def bcp(x: np.ndarray, p0: float = 0.2, w0: Optional[float] = None,
        burnin: int = 50, mcmc: int = 500, use_exact: bool = True) -> BcpResult:
    """
    Convenience function for Bayesian Change Point detection.

    Parameters
    ----------
    x : np.ndarray
        1D array of observations.
    p0 : float, default=0.2
        Prior probability of no change point.
    w0 : float, optional
        Prior weight parameter. If None, uses default.
    burnin : int, default=50
        Number of MCMC burn-in iterations (if use_exact=False).
    mcmc : int, default=500
        Number of MCMC iterations (if use_exact=False).
    use_exact : bool, default=True
        If True, use exact recursive algorithm. Otherwise use MCMC.

    Returns
    -------
    BcpResult
        Object with posterior_mean and posterior_prob arrays.
    """
    model = BcpPython(p0=p0, w0=w0, burnin=burnin, mcmc=mcmc, use_exact=use_exact)
    return model.fit(x)
