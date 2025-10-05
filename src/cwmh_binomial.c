/**
 * @file cwmh_binomial.c
 * @brief Component-wise Metropolis-Hastings sampling for logit-binomial state-space
 *        models (local level and local trend) - Optimized version.
 * @author Michel H. Montoril
 * @date 2025-09-22
 * @version 1.1
 *
 * @details This file implements optimized MCMC update routines for state-space models with
 *          binomial observations and logit link, including:
 *          - Component-wise MH updates for local level (random walk) binomial models
 *          - Component-wise MH updates for local trend (random walk + trend) binomial models
 *          - Memory optimizations and computational efficiency improvements
 *          - Numerical stability enhancements
 */

#include <R.h>
#include <Rmath.h>

#include "utils.h"  /* ilogit */
#include "cwmh_binomial.h"

/**
 * @brief Numerically stable log acceptance probability computation
 * @details Prevents overflow/underflow in acceptance probability calculations
 *
 * @param lp1n Log-likelihood contribution of the first component at the new state.
 * @param lp2n Log-likelihood contribution of the second component at the new state.
 * @param lp1o Log-likelihood contribution of the first component at the old state.
 * @param lp2o Log-likelihood contribution of the second component at the old state.
 * @return Log of the acceptance probability truncated at 0 to cap the probability at 1.
 */
static inline double stable_log_accept_prob(double lp1n, double lp2n, double lp1o, double lp2o) {
  double log_ratio = (lp1n + lp2n) - (lp1o + lp2o);
  return fmin2(0.0, log_ratio);  // Cap at 0 (probability 1)
}

/**
 * @brief Cache structure for expensive computations
 */
typedef struct {
  double cached_prec;
  double cached_sd_last;
  double cached_sd_regular;
  int cached_iter;
} precision_cache_t;

static precision_cache_t prec_cache = {-1.0, 0.0, 0.0, -1};

/**
 * @brief Component-wise Metropolis-Hastings sampler for theta_1 in a logit-binomial
 *        local level model - Optimized version.
 *
 * @details This function implements an optimized component-wise Metropolis-Hastings algorithm
 *          to sample the level state vector theta_1 in a binomial observation model with
 *          logit link:
 *          y_t ~ Binomial(n_trials, alpha_t),
 *          where alpha_t = logit^{-1}(theta_{t,1}).
 *
 *          State equation (local level):
 *              theta_{t,1} = theta_{t-1,1} + u_{t,1}, u_{t,1} ~ N(0, 1/prec_theta_1)
 *
 *          **Optimizations implemented:**
 *          - Cached precision computations to avoid repeated sqrt/division
 *          - Reduced memory allocation by eliminating redundant arrays
 *          - Numerically stable ilogit function from utils.c
 *          - Improved memory locality through sequential access patterns
 *          - Stable log-probability computations
 *
 *          **Memory optimization:** theta_1_updated now uses sliding window of size
 *          lag_update x n instead of full B x n matrix.
 *
 *          **Precision structure:**
 *          - Regular elements (k=0..n-2): sd_regular = 1/sqrt(prec_theta_1 * 2)
 *          - Last element (k=n-1):        sd_last    = 1/sqrt(prec_theta_1)
 *
 *          **Conditional means for proposal:**
 *          - First:
 *          E[theta_{1,1} | theta_01, theta_{2,1}] = 0.5 * (theta_{2,1} + theta_01)
 *          - Intermediate:
 *          E[theta_{k,1} | theta_{k-1,1}, theta_{k+1,1}] = 0.5 * (theta_{k-1,1} + theta_{k+1,1})
 *          - Last:
 *          E[theta_{n,1} | theta_{n-1,1}] = theta_{n-1,1}
 *
 * @param theta_1            Matrix of level states (vectorized B x n), input/output.
 * @param theta_01           Vector of initial level states (size B).
 * @param theta_1_updated    Sliding window matrix of acceptance indicators
 *                           (vectorized lag_update x n), output. Uses circular indexing.
 * @param alpha              Matrix of transformed probabilities (vectorized B x n), output.
 *                           Each alpha[t] = logit^{-1}(theta_1[t]) represents the binomial
 *                           success probability at time t.
 * @param prec_theta_1       Vector of level precision parameters (size B).
 * @param y                  Vector of observed binomial counts (size n).
 *                           Each y[k] must satisfy 0 ≤ y[k] ≤ n_trials.
 * @param log_sigma          Vector of log proposal standard deviations (size n).
 * @param hat_theta_1        Temporary vector for conditional means (size n).
 * @param theta_1_new        Temporary vector for proposed values (size n).
 * @param log_accept_prob    Temporary vector for log acceptance probabilities (size n).
 * @param lag_update         Sliding window size for theta_1_updated indexing.
 * @param n_trials           Number of Bernoulli trials for binomial distribution (double).
 * @param n                  Length of the time series.
 * @param iter               Current MCMC iteration (0-based).
 *
 * @note Computational complexity: O(n) per MCMC iteration due to component-wise updates.
 * @note Memory requirements: O(n) temporary storage + O(lag_update * n) for sliding window.
 * @note Numerical stability: Uses optimized log-probabilities and stable ilogit function.
 * @note Forward sampling: Components are updated sequentially using previously updated
 *       values within the same iteration, which can improve mixing compared to
 *       simultaneous updates.
 * @note Model specification: Implements a local level binomial model (random walk only).
 * @note Cache optimization: Precision-dependent calculations are cached between iterations.
 *
 * @warning Each y[k] must satisfy 0 ≤ y[k] ≤ n_trials.
 * @warning Results are invalid if theta_01 or prec_theta_1 do not contain sufficient
 *          history (iter < 1).
 * @warning lag_update must be > 0 for theta_1_updated indexing.
 */
void cwmh_alpha_logit_binomial_locallevel(double *theta_1,
                                          double *theta_01,
                                          double *theta_1_updated,
                                          double *alpha,
                                          double *prec_theta_1,
                                          double *y,
                                          double *log_sigma,
                                          double *hat_theta_1,
                                          double *theta_1_new,
                                          double *log_accept_prob,
                                          int lag_update,
                                          double n_trials,
                                          int n,
                                          int iter) {

  int k;

  /* ========== Iteration Index Setup ========== */
  const int iter_n = iter * n;                              // Current iteration start index
  const int iterm1_n = iter_n - n;                          // Previous iteration start index
  const int prev_iter = iter - 1;                           // Previous iteration for theta_01 and prec_theta_1

  /* ========== Optimized Precision Calculations with Caching ========== */
  double current_prec = prec_theta_1[prev_iter];
  double sd_last, sd_regular;

  // Cache expensive sqrt and division operations
  if (prec_cache.cached_prec != current_prec || prec_cache.cached_iter != iter) {
    prec_cache.cached_prec = current_prec;
    prec_cache.cached_sd_last = 1.0 / sqrt(current_prec);
    prec_cache.cached_sd_regular = prec_cache.cached_sd_last * M_SQRT1_2;
    prec_cache.cached_iter = iter;
  }
  sd_last = prec_cache.cached_sd_last;
  sd_regular = prec_cache.cached_sd_regular;

  /* ========== Pre-compute proposal standard deviations ========== */
  double sigma_vals[n];
  for (k = 0; k < n; k++) {
    sigma_vals[k] = exp(log_sigma[k]);
  }

  /* ========== Sliding window index for theta_1_updated ========== */
  const int window_row = iter % lag_update;
  const int window_idx = window_row * n;

  /* ========== First Element (k = 0) ========== */
  // Conditional mean incorporating initial state from previous iteration
  // For local level: E[theta_{1,1} | theta_01, theta_{2,1}] = 0.5 * (theta_{2,1} + theta_01)
  hat_theta_1[0] = 0.5 * (theta_1[iterm1_n + 1] + theta_01[prev_iter]);

  // Generate proposal from random walk
  theta_1_new[0] = rnorm(theta_1[iterm1_n], sigma_vals[0]);

  // Calculate log acceptance probability using cached precision values
  double lp1n = dnorm(theta_1_new[0], hat_theta_1[0], sd_regular, 1);
  double lp2n = dbinom(y[0], n_trials, ilogit(theta_1_new[0]), 1);
  double lp1o = dnorm(theta_1[iterm1_n], hat_theta_1[0], sd_regular, 1);
  double lp2o = dbinom(y[0], n_trials, ilogit(theta_1[iterm1_n]), 1);

  log_accept_prob[0] = stable_log_accept_prob(
    lp1n, /* lp1n: log-density of new state w.r.t. prior */
    lp2n, /* lp2n: log-likelihood of new state */
    lp1o, /* lp1o: log-density of current state w.r.t. prior */
    lp2o  /* lp2o: log-likelihood of current state */
  );
  int accepted_0 = log(runif(0, 1)) <= log_accept_prob[0];

  if (!accepted_0) {
    theta_1_new[0] = theta_1[iterm1_n];               // Reject: keep old value
  }

  // Store acceptance indicator in sliding window
  theta_1_updated[window_idx + 0] = accepted_0;

  /* ========== Intermediate Elements (k = 1 to n-2) ========== */
  // Cache previous value for improved memory locality
  double theta_prev = theta_1_new[0];

  for (k = 1; k < (n - 1); k++) {
    // Conditional mean for local level model: E[theta_{k,1} | theta_{k-1,1}, theta_{k+1,1}]
    // Using forward-sampling: previously updated theta_prev
    double theta_next = theta_1[iterm1_n + k + 1];
    hat_theta_1[k] = 0.5 * (theta_next + theta_prev);

    theta_1_new[k] = rnorm(theta_1[iterm1_n + k], sigma_vals[k]);

    // Log densities using cached precision values
    lp1n = dnorm(theta_1_new[k], hat_theta_1[k], sd_regular, 1);
    lp2n = dbinom(y[k], n_trials, ilogit(theta_1_new[k]), 1);
    lp1o = dnorm(theta_1[iterm1_n + k], hat_theta_1[k], sd_regular, 1);
    lp2o = dbinom(y[k], n_trials, ilogit(theta_1[iterm1_n + k]), 1);

    log_accept_prob[k] = stable_log_accept_prob(
      lp1n, /* lp1n: log-density of new state w.r.t. prior */
      lp2n, /* lp2n: log-likelihood of new state */
      lp1o, /* lp1o: log-density of current state w.r.t. prior */
      lp2o  /* lp2o: log-likelihood of current state */
    );
    int accepted_k = log(runif(0, 1)) <= log_accept_prob[k];

    if (!accepted_k) {
      theta_1_new[k] = theta_1[iterm1_n + k];
    }

    // Store acceptance indicator in sliding window
    theta_1_updated[window_idx + k] = accepted_k;

    // Update cached value for next iteration
    theta_prev = theta_1_new[k];
  }

  /* ========== Last Element (k = n-1) ========== */
  // Simplified conditional mean for last element: E[theta_{1,n-1} | theta_{1,n-2}]
  // For local level: just the previous state (random walk)
  hat_theta_1[n - 1] = theta_prev;  // Use cached value

  theta_1_new[n - 1] = rnorm(theta_1[iterm1_n + (n - 1)], sigma_vals[n - 1]);

  // Log densities using cached precision (last element)
  lp1n = dnorm(theta_1_new[n - 1], hat_theta_1[n - 1], sd_last, 1);
  lp2n = dbinom(y[n - 1], n_trials, ilogit(theta_1_new[n - 1]), 1);
  lp1o = dnorm(theta_1[iterm1_n + (n - 1)], hat_theta_1[n - 1], sd_last, 1);
  lp2o = dbinom(y[n - 1], n_trials, ilogit(theta_1[iterm1_n + (n - 1)]), 1);

  log_accept_prob[n - 1] = stable_log_accept_prob(
    lp1n, /* lp1n: log-density of new state w.r.t. prior */
    lp2n, /* lp2n: log-likelihood of new state */
    lp1o, /* lp1o: log-density of current state w.r.t. prior */
    lp2o  /* lp2o: log-likelihood of current state */
  );
  int accepted_last = log(runif(0, 1)) <= log_accept_prob[n - 1];

  if (!accepted_last) {
    theta_1_new[n - 1] = theta_1[iterm1_n + (n - 1)];
  }

  // Store acceptance indicator in sliding window
  theta_1_updated[window_idx + (n - 1)] = accepted_last;

  /* ========== Update Output Arrays (vectorized) ========== */
  for (k = 0; k < n; k++) {
    theta_1[iter_n + k] = theta_1_new[k];                      // Store sampled states
    alpha[iter_n + k] = ilogit(theta_1_new[k]);               // Store transformed probabilities
  }
}

//----------------------------------------------------------------------

/**
 * @brief Component-wise Metropolis-Hastings sampler for theta_1 in a logit-binomial
 *        dynamic model (local trend) - Optimized version.
 *
 * @details This function implements an optimized component-wise Metropolis-Hastings algorithm
 *          to sample the level state vector theta_1 in a binomial observation model with
 *          logit link:
 *          y_t ~ Binomial(n_trials, alpha_t),
 *          where alpha_t = logit^{-1}(theta_{t,1}).
 *
 *          State equation (local trend):
 *          theta_{t,1} = theta_{t-1,1} + theta_{t-1,2} + u_{t,1},
 *          with u_{t,1} ~ N(0, 1/prec_theta_1).
 *
 *          **Optimizations implemented:**
 *          - Cached precision computations to avoid repeated sqrt/division
 *          - Reduced memory allocation by eliminating redundant arrays
 *          - Numerically stable ilogit function from utils.c
 *          - Improved memory locality through sequential access patterns
 *          - Stable log-probability computations
 *
 *          **Memory optimization:** theta_1_updated now uses sliding window of size
 *          lag_update x n instead of full B x n matrix.
 *
 *          **Precision structure:**
 *          - Regular elements (k=0..n-2): sd_regular = 1/sqrt(prec_theta_1 * 2)
 *          - Last element (k=n-1):        sd_last    = 1/sqrt(prec_theta_1)
 *
 *          **Conditional means for proposal:**
 *          - First:
 *          E[theta_{1,1} | theta_01, theta_02, theta_{2,1}, theta_{1,2}] =
 *                            0.5 * (theta_{2,1} - theta_{1,2} + theta_01 + theta_02)
 *          - Intermediate:
 *          E[theta_{k,1} | theta_{k-1,1}, theta_{k-1,2}, theta_{k+1,1}, theta_{k,2}] =
 *                            0.5 * (theta_{k+1,1} - theta_{k,2} + theta_{k-1,1} - theta_{k-1,2})
 *          - Last:
 *          E[theta_{n,1} | theta_{n-1,1}, theta_{n-1,2}] = theta_{n-1,1} + theta_{n-1,2}
 *
 * @param theta_1            Matrix of level states (vectorized B x n), input/output.
 * @param theta_2            Matrix of trend states (vectorized B x n), input only.
 * @param theta_01           Vector of initial level states (size B).
 * @param theta_02           Vector of initial trend states (size B).
 * @param theta_1_updated    Sliding window matrix of acceptance indicators
 *                           (vectorized lag_update x n), output. Uses circular indexing.
 * @param alpha              Matrix of transformed probabilities (vectorized B x n), output.
 * @param prec_theta_1       Vector of level precision parameters (size B).
 * @param y                  Vector of observed binomial counts (size n).
 *                           Each y[k] must satisfy 0 ≤ y[k] ≤ n_trials.
 * @param log_sigma          Vector of log proposal standard deviations (size n).
 * @param hat_theta_1        Temporary vector for conditional means (size n).
 * @param theta_1_new        Temporary vector for proposed values (size n).
 * @param log_accept_prob    Temporary vector for log acceptance probabilities (size n).
 * @param lag_update         Sliding window size for theta_1_updated indexing.
 * @param n_trials           Number of Bernoulli trials for binomial distribution (double).
 * @param n                  Length of the time series.
 * @param iter               Current MCMC iteration (0-based).
 *
 * @note Computational complexity: O(n) per MCMC iteration due to component-wise updates.
 * @note Memory requirements: O(n) temporary storage + O(lag_update * n) for sliding window.
 * @note Numerical stability: Uses optimized log-probabilities and stable ilogit function.
 * @note Forward sampling: Components are updated sequentially using previously updated
 *       values within the same iteration, which can improve mixing compared to
 *       simultaneous updates.
 * @note Model specification: Implements a local trend binomial model (random walk + trend).
 * @note Cache optimization: Precision-dependent calculations are cached between iterations.
 *
 * @warning Each y[k] must satisfy 0 ≤ y[k] ≤ n_trials.
 * @warning Results are invalid if theta_01 or prec_theta_1 do not contain sufficient
 *          history (iter < 1).
 * @warning lag_update must be > 0 for theta_1_updated indexing.
 */
void cwmh_alpha_logit_binomial(double *theta_1,
                               double *theta_2,
                               double *theta_01,
                               double *theta_02,
                               double *theta_1_updated,
                               double *alpha,
                               double *prec_theta_1,
                               double *y,
                               double *log_sigma,
                               double *hat_theta_1,
                               double *theta_1_new,
                               double *log_accept_prob,
                               int lag_update,
                               double n_trials,
                               int n,
                               int iter) {

  int k;

  /* ========== Iteration Index Setup ========== */
  int iter_n = iter * n;                              // Current iteration start index
  int iterm1_n = iter_n - n;                          // Previous iteration start index
  int prev_iter = iter - 1;                           // Previous iteration for theta_01 and prec_theta_1

  /* ========== Optimized Precision Calculations with Caching ========== */
  double current_prec = prec_theta_1[prev_iter];
  double sd_last, sd_regular;

  // Cache expensive sqrt and division operations
  if (prec_cache.cached_prec != current_prec || prec_cache.cached_iter != iter) {
    prec_cache.cached_prec = current_prec;
    prec_cache.cached_sd_last = 1.0 / sqrt(current_prec);
    prec_cache.cached_sd_regular = prec_cache.cached_sd_last * M_SQRT1_2;
    prec_cache.cached_iter = iter;
  }
  sd_last = prec_cache.cached_sd_last;
  sd_regular = prec_cache.cached_sd_regular;

  /* ========== Pre-compute proposal standard deviations ========== */
  double sigma_vals[n];
  for (k = 0; k < n; k++) {
    sigma_vals[k] = exp(log_sigma[k]);
  }

  /* ========== Sliding window index for theta_1_updated ========== */
  const int window_row = iter % lag_update;
  const int window_idx = window_row * n;

  /* ========== First Element (k = 0) ========== */
  // Conditional mean incorporating initial states from previous iteration
  hat_theta_1[0] = 0.5 * (theta_1[iterm1_n + 1] - theta_2[iter_n] +
  theta_01[prev_iter] + theta_02[prev_iter]);

  // Generate proposal from random walk
  theta_1_new[0] = rnorm(theta_1[iterm1_n], sigma_vals[0]);

  // Calculate log acceptance probability using cached precision values
  double lp1n = dnorm(theta_1_new[0], hat_theta_1[0], sd_regular, 1);
  double lp2n = dbinom(y[0], n_trials, ilogit(theta_1_new[0]), 1);
  double lp1o = dnorm(theta_1[iterm1_n], hat_theta_1[0], sd_regular, 1);
  double lp2o = dbinom(y[0], n_trials, ilogit(theta_1[iterm1_n]), 1);

  log_accept_prob[0] = stable_log_accept_prob(
    lp1n, /* lp1n: log-density of new state w.r.t. prior */
    lp2n, /* lp2n: log-likelihood of new state */
    lp1o, /* lp1o: log-density of current state w.r.t. prior */
    lp2o  /* lp2o: log-likelihood of current state */
  );
  int accepted_0 = log(runif(0, 1)) <= log_accept_prob[0];

  if (!accepted_0) {
    theta_1_new[0] = theta_1[iterm1_n];               // Reject: keep old value
  }

  // Store acceptance indicator in sliding window
  theta_1_updated[window_idx + 0] = accepted_0;

  /* ========== Intermediate Elements (k = 1 to n-2) ========== */
  // Cache previous value for improved memory locality
  double theta_prev = theta_1_new[0];

  for (k = 1; k < (n - 1); k++) {
    // Conditional mean using forward-sampling: previously updated theta_prev
    double theta_next = theta_1[iterm1_n + k + 1];
    double theta_2_curr = theta_2[iter_n + k];
    double theta_2_prev = theta_2[iter_n + k - 1];

    hat_theta_1[k] = 0.5 * (theta_next - theta_2_curr + theta_prev - theta_2_prev);

    theta_1_new[k] = rnorm(theta_1[iterm1_n + k], sigma_vals[k]);

    // Log densities using cached precision values
    lp1n = dnorm(theta_1_new[k], hat_theta_1[k], sd_regular, 1);
    lp2n = dbinom(y[k], n_trials, ilogit(theta_1_new[k]), 1);
    lp1o = dnorm(theta_1[iterm1_n + k], hat_theta_1[k], sd_regular, 1);
    lp2o = dbinom(y[k], n_trials, ilogit(theta_1[iterm1_n + k]), 1);

    log_accept_prob[k] = stable_log_accept_prob(
      lp1n, /* lp1n: log-density of new state w.r.t. prior */
      lp2n, /* lp2n: log-likelihood of new state */
      lp1o, /* lp1o: log-density of current state w.r.t. prior */
      lp2o  /* lp2o: log-likelihood of current state */
    );
    int accepted_k = log(runif(0, 1)) <= log_accept_prob[k];

    if (!accepted_k) {
      theta_1_new[k] = theta_1[iterm1_n + k];
    }

    // Store acceptance indicator in sliding window
    theta_1_updated[window_idx + k] = accepted_k;

    // Update cached value for next iteration
    theta_prev = theta_1_new[k];
  }

  /* ========== Last Element (k = n-1) ========== */
  // Simplified conditional mean: no future state dependency
  double theta_2_prev = theta_2[iter_n + n - 2];
  hat_theta_1[n - 1] = theta_prev + theta_2_prev;  // Use cached theta_prev

  theta_1_new[n - 1] = rnorm(theta_1[iterm1_n + (n - 1)], sigma_vals[n - 1]);

  // Log densities using cached precision (last element)
  lp1n = dnorm(theta_1_new[n - 1], hat_theta_1[n - 1], sd_last, 1);
  lp2n = dbinom(y[n - 1], n_trials, ilogit(theta_1_new[n - 1]), 1);
  lp1o = dnorm(theta_1[iterm1_n + (n - 1)], hat_theta_1[n - 1], sd_last, 1);
  lp2o = dbinom(y[n - 1], n_trials, ilogit(theta_1[iterm1_n + (n - 1)]), 1);

  log_accept_prob[n - 1] = stable_log_accept_prob(
    lp1n, /* lp1n: log-density of new state w.r.t. prior */
    lp2n, /* lp2n: log-likelihood of new state */
    lp1o, /* lp1o: log-density of current state w.r.t. prior */
    lp2o  /* lp2o: log-likelihood of current state */
  );
  int accepted_last = log(runif(0, 1)) <= log_accept_prob[n - 1];

  if (!accepted_last) {
    theta_1_new[n - 1] = theta_1[iterm1_n + (n - 1)];
  }

  // Store acceptance indicator in sliding window
  theta_1_updated[window_idx + (n - 1)] = accepted_last;

  /* ========== Update Output Arrays (vectorized) ========== */
  for (k = 0; k < n; k++) {
    theta_1[iter_n + k] = theta_1_new[k];                      // Store sampled states
    alpha[iter_n + k] = ilogit(theta_1_new[k]);               // Store transformed probabilities
  }
}
