#ifndef CWMH_BINOMIAL_H
#define CWMH_BINOMIAL_H

/**
 * Component-wise Metropolis-Hastings sampler for theta_1 in a logit-binomial local level model.
 *
 * This function implements a component-wise Metropolis-Hastings algorithm to sample
 * the level state vector theta_1 when the observation equation follows a binomial
 * distribution with logit link function: y_t ~ Binomial(n_trials, alpha_t), where
 * alpha_t = logit^{-1}(theta_{1,t}).
 *
 * The state equation is a simple random walk: theta_{1,t} = theta_{1,t-1} + u_{1,t},
 * where u_{1,t} ~ N(0, 1/prec_theta_1). This is simpler than the local trend model
 * as it excludes the trend component entirely.
 *
 * **Important Note on Iteration Timing**: This function uses theta_01[iter-1] and
 * prec_theta_1[iter-1] because these parameters are sampled later in the Gibbs sequence
 * and thus their current iteration values are not yet available.
 *
 * **Precision Matrix Structure**: The function uses two different precision structures:
 * - Intermediate elements (k=0 to n-2): Standard deviation = 1/sqrt(prec_theta_1 * 2)
 *   This reflects the tridiagonal precision matrix structure where interior nodes
 *   have connections to both past and future states.
 * - Boundary element (k=n-1): Standard deviation = 1/sqrt(prec_theta_1)
 *   This reflects the simplified structure at the time series boundary where
 *   there is no future state dependency.
 *
 * The function handles three cases with appropriate boundary conditions:
 * - First element: incorporates initial state theta_01
 * - Intermediate elements: uses forward-sampling with previously updated values
 * - Last element: simplified structure without future state dependency
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
 * @note Model specification: This implements a local level model (random walk) without
 *       trend components, making it computationally simpler than the local trend version.
 */
void CWMH_alpha_logit_binomial_locallevel(double *theta_1,
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
 * Component-wise Metropolis-Hastings sampler for theta_1 in a logit-binomial dynamic model.
 *
 * This function implements a component-wise Metropolis-Hastings algorithm to sample
 * the level state vector theta_1 when the observation equation follows a binomial
 * distribution with logit link function: y_t ~ Binomial(n_trials, alpha_t), where
 * alpha_t = logit^{-1}(theta_{1,t}).
 *
 * The state equation remains linear: theta_{1,t} = theta_{1,t-1} + theta_{2,t-1} + u_{1,t},
 * but the non-linear observation equation requires MCMC sampling instead of closed-form
 * Gibbs updates.
 *
 * **Important Note on Iteration Timing**: This function uses theta_01[iter-1] and
 * prec_theta_1[iter-1] because these parameters are sampled later in the Gibbs sequence
 * and thus their current iteration values are not yet available.
 *
 * **Precision Matrix Structure**: The function uses two different precision structures:
 * - Intermediate elements (k=0 to n-2): Standard deviation = 1/sqrt(prec_theta_1 * 2)
 *   This reflects the tridiagonal precision matrix structure where interior nodes
 *   have connections to both past and future states.
 * - Boundary element (k=n-1): Standard deviation = 1/sqrt(prec_theta_1)
 *   This reflects the simplified structure at the time series boundary where
 *   there is no future state dependency.
 *
 * The function handles three cases with appropriate boundary conditions:
 * - First element: incorporates initial states theta_01 and theta_02
 * - Intermediate elements: uses forward-sampling with previously updated values
 * - Last element: simplified structure without future state dependency
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
 */
void CWMH_alpha_logit_binomial(double *theta_1,
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
