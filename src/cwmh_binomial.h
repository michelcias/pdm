/**
 * @file cwmh_binomial.h
 * @brief Header for component-wise Metropolis-Hastings sampling routines for logit-binomial
 *        state-space models (local level and local trend).
 * @author Michel H. Montoril
 * @date 2025-08-10
 * @version 1.0
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
 *        local level model.
 *
 * @details This function implements a component-wise Metropolis-Hastings algorithm to
 *          sample the level state vector theta_1 in a binomial observation model with
 *          logit link:
 *          y_t ~ Binomial(n_trials, alpha_t),
 *          where alpha_t = logit^{-1}(theta_{1,t}).
 *
 *          State equation (local level):
 *              theta_{1,t} = theta_{1,t-1} + u_{1,t}, u_{1,t} ~ N(0, 1/prec_theta_1)
 *
 *          **Iteration timing:** Uses theta_01[iter-1] and prec_theta_1[iter-1] because
 *          these parameters are sampled later in the Gibbs sequence.
 *
 *          **Precision structure:**
 *          - Interior (k=0..n-2): sd = 1/sqrt(prec_theta_1 * 2)
 *          - Boundary (k=n-1):    sd = 1/sqrt(prec_theta_1)
 *
 *          **Conditional means for proposal:**
 *          - First:
 *          E[theta_{1,1} | theta_01, theta_{1,2}] = 0.5 * (theta_{1,2} + theta_01)
 *          - Intermediate:
 *          E[theta_{1,k} | theta_{1,k-1}, theta_{1,k+1}] = 0.5 * (theta_{1,k-1} + theta_{1,k+1})
 *          - Last:
 *          E[theta_{1,n} | theta_{1,n-1}] = theta_{1,n-1}
 *
 * @param theta_1            Matrix of level states (vectorized B x n), input/output.
 * @param theta_01           Vector of initial level states (size B).
 * @param theta_1_updated    Matrix of acceptance indicators (vectorized B x n), output.
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
 * @param updated            Temporary vector for acceptance indicators (size n).
 * @param n_trials           Number of Bernoulli trials for binomial distribution (double).
 * @param n                  Length of the time series.
 * @param iter               Current MCMC iteration (0-based).
 *
 * @note Computational complexity: O(n) per MCMC iteration due to component-wise updates.
 * @note Memory requirements: O(n) temporary storage for proposal and acceptance vectors.
 * @note Numerical stability: Uses log-probabilities throughout to avoid underflow issues.
 * @note Forward sampling: Components are updated sequentially using previously updated
 *       values within the same iteration, which can improve mixing compared to
 *       simultaneous updates.
 * @note Model specification: Implements a local level binomial model (random walk only).
 *
 * @warning Each y[k] must satisfy 0 ≤ y[k] ≤ n_trials.
 * @warning Results are invalid if theta_01 or prec_theta_1 do not contain sufficient
 *          history (iter < 1).
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
                                          int *updated,
                                          double n_trials,
                                          int n,
                                          int iter);

//----------------------------------------------------------------------

/**
 * @brief Component-wise Metropolis-Hastings sampler for theta_1 in a logit-binomial
 *        dynamic model (local trend).
 *
 * @details This function implements a component-wise Metropolis-Hastings algorithm to
 *          sample the level state vector theta_1 in a binomial observation model with
 *          logit link:
 *          y_t ~ Binomial(n_trials, alpha_t),
 *          where alpha_t = logit^{-1}(theta_{1,t}).
 *
 *          State equation (local trend):
 *          theta_{1,t} = theta_{1,t-1} + theta_{2,t-1} + u_{1,t},
 *          with u_{1,t} ~ N(0, 1/prec_theta_1).
 *
 *          **Iteration timing:** Uses theta_01[iter-1] and prec_theta_1[iter-1] because
 *          these parameters are sampled later in the Gibbs sequence.
 *
 *          **Precision structure:**
 *          - Interior (k=0..n-2): sd = 1/sqrt(prec_theta_1 * 2)
 *          - Boundary (k=n-1):    sd = 1/sqrt(prec_theta_1)
 *
 *          **Conditional means for proposal:**
 *          - First:
 *          E[theta_{1,1} | theta_01, theta_02, theta_{1,2}, theta_{2,1}] =
 *                            0.5 * (theta_{1,2} - theta_{2,1} + theta_01 + theta_02)
 *          - Intermediate:
 *          E[theta_{1,k} | theta_{1,k-1}, theta_{2,k-1}, theta_{1,k+1}, theta_{2,k}] =
 *                            0.5 * (theta_{1,k+1} - theta_{2,k} + theta_{1,k-1} - theta_{2,k-1})
 *          - Last:
 *          E[theta_{1,n} | theta_{1,n-1}, theta_{2,n-1}] = theta_{1,n-1} + theta_{2,n-1}
 *
 * @param theta_1            Matrix of level states (vectorized B x n), input/output.
 * @param theta_2            Matrix of trend states (vectorized B x n), input only.
 * @param theta_01           Vector of initial level states (size B).
 * @param theta_02           Vector of initial trend states (size B).
 * @param theta_1_updated    Matrix of acceptance indicators (vectorized B x n), output.
 * @param alpha              Matrix of transformed probabilities (vectorized B x n), output.
 * @param prec_theta_1       Vector of level precision parameters (size B).
 * @param y                  Vector of observed binomial counts (size n).
 *                           Each y[k] must satisfy 0 ≤ y[k] ≤ n_trials.
 * @param log_sigma          Vector of log proposal standard deviations (size n).
 * @param hat_theta_1        Temporary vector for conditional means (size n).
 * @param theta_1_new        Temporary vector for proposed values (size n).
 * @param log_accept_prob    Temporary vector for log acceptance probabilities (size n).
 * @param updated            Temporary vector for acceptance indicators (size n).
 * @param n_trials           Number of Bernoulli trials for binomial distribution (double).
 * @param n                  Length of the time series.
 * @param iter               Current MCMC iteration (0-based).
 *
 * @note Computational complexity: O(n) per MCMC iteration due to component-wise updates.
 * @note Memory requirements: O(n) temporary storage for proposal and acceptance vectors.
 * @note Numerical stability: Uses log-probabilities throughout to avoid underflow issues.
 * @note Forward sampling: Components are updated sequentially using previously updated
 *       values within the same iteration, which can improve mixing compared to
 *       simultaneous updates.
 * @note Model specification: Implements a local trend binomial model (random walk + trend).
 *
 * @warning Each y[k] must satisfy 0 ≤ y[k] ≤ n_trials.
 * @warning Results are invalid if theta_01 or prec_theta_1 do not contain sufficient
 *          history (iter < 1).
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
                               int *updated,
                               double n_trials,
                               int n,
                               int iter);

#endif /* CWMH_BINOMIAL_H */
