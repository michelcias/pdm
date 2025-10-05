/**
 * @file cwmh_binomial.h
 * @brief Component-wise Metropolis-Hastings sampling for logit-binomial state-space
 *        models (local level and local trend) - Optimized version.
 * @author Michel H. Montoril
 * @date 2025-09-22
 * @version 1.1
 *
 * @details This header declares functions for MCMC updates in state-space models with binomial
 *          observations and logit link, including:
 *          - Component-wise MH updates for local level (random walk) binomial models
 *          - Component-wise MH updates for local trend (random walk + trend) binomial models
 */

#ifndef CWMH_BINOMIAL_H
#define CWMH_BINOMIAL_H

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
                                          int iter);

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
                               int iter);

#endif /* CWMH_BINOMIAL_H */
