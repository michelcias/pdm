#include <R.h>
#include <Rmath.h>

#include "utils.h"  /* ilogit */
#include "cwmh_binomial.h"


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
                                          int iter) {

  int k;

  /* ========== Iteration Index Setup and Conditional Normal SDs ========== */
  const int iter_n = iter * n;                              // Current iteration start index
  const int iterm1_n = iter_n - n;                          // Previous iteration start index
  const int prev_iter = iter - 1;                           // Previous iteration for theta_01 and prec_theta_1

  /* Conditional Normal SDs from the RW1 prior:
   * Let inv_sqrt_prec = 1 / sqrt(prec_theta_1[prev_iter]).
   * - Interior nodes (k = 0..n-2): sd = inv_sqrt_prec / sqrt(2) = inv_sqrt_prec * M_SQRT1_2
   * - Boundary node (k = n-1):     sd = inv_sqrt_prec
   */
  const double sd_boundary = 1.0 / sqrt(prec_theta_1[prev_iter]);
  const double sd_interior = sd_boundary * M_SQRT1_2;  // 1/sqrt(2)

  /* ========== First Element (k = 0) ========== */
  double sigma_val = exp(log_sigma[0]);                // Proposal standard deviation

  // Conditional mean incorporating initial state from previous iteration
  // For local level: E[theta_{1,1} | theta_01, theta_{1,2}] = 0.5 * (theta_{1,2} + theta_01)
  hat_theta_1[0] = 0.5 * (theta_1[iterm1_n + 1] + theta_01[prev_iter]);

  // Generate proposal from random walk
  theta_1_new[0] = rnorm(theta_1[iterm1_n], sigma_val);

  // Calculate log acceptance probability using previous iteration's precision
  double lp1n = dnorm(theta_1_new[0], hat_theta_1[0], sd_interior, 1);
  double lp2n = dbinom(y[0], n_trials, ilogit(theta_1_new[0]), 1);
  double lp1o = dnorm(theta_1[iterm1_n], hat_theta_1[0], sd_interior, 1);
  double lp2o = dbinom(y[0], n_trials, ilogit(theta_1[iterm1_n]), 1);

  log_accept_prob[0] = lp1n + lp2n - lp1o - lp2o;
  updated[0] = log(runif(0, 1)) <= log_accept_prob[0];

  if (!updated[0]) {
    theta_1_new[0] = theta_1[iterm1_n];               // Reject: keep old value
  }

  /* ========== Intermediate Elements (k = 1 to n-2) ========== */
  for (k = 1; k < (n - 1); k++) {
    sigma_val = exp(log_sigma[k]);

    // Conditional mean for local level model: E[theta_{1,k} | theta_{1,k-1}, theta_{1,k+1}]
    // Using forward-sampling: previously updated theta_1_new[k-1]
    hat_theta_1[k] = 0.5 * (theta_1[iterm1_n + k + 1] + theta_1_new[k - 1]);

    theta_1_new[k] = rnorm(theta_1[iterm1_n + k], sigma_val);

    // Log densities using previous iteration's precision
    lp1n = dnorm(theta_1_new[k], hat_theta_1[k], sd_interior, 1);
    lp2n = dbinom(y[k], n_trials, ilogit(theta_1_new[k]), 1);
    lp1o = dnorm(theta_1[iterm1_n + k], hat_theta_1[k], sd_interior, 1);
    lp2o = dbinom(y[k], n_trials, ilogit(theta_1[iterm1_n + k]), 1);

    log_accept_prob[k] = lp1n + lp2n - lp1o - lp2o;
    updated[k] = log(runif(0, 1)) <= log_accept_prob[k];

    if (!updated[k]) {
      theta_1_new[k] = theta_1[iterm1_n + k];
    }
  }

  /* ========== Last Element (k = n-1) ========== */
  sigma_val = exp(log_sigma[n - 1]);

  // Simplified conditional mean for boundary: E[theta_{1,n-1} | theta_{1,n-2}]
  // For local level: just the previous state (random walk)
  hat_theta_1[n - 1] = theta_1_new[n - 2];

  theta_1_new[n - 1] = rnorm(theta_1[iterm1_n + (n - 1)], sigma_val);

  // Log densities using previous iteration's precision (boundary condition)
  lp1n = dnorm(theta_1_new[n - 1], hat_theta_1[n - 1], sd_boundary, 1);
  lp2n = dbinom(y[n - 1], n_trials, ilogit(theta_1_new[n - 1]), 1);
  lp1o = dnorm(theta_1[iterm1_n + (n - 1)], hat_theta_1[n - 1], sd_boundary, 1);
  lp2o = dbinom(y[n - 1], n_trials, ilogit(theta_1[iterm1_n + (n - 1)]), 1);

  log_accept_prob[n - 1] = lp1n + lp2n - lp1o - lp2o;
  updated[n - 1] = log(runif(0, 1)) <= log_accept_prob[n - 1];

  if (!updated[n - 1]) {
    theta_1_new[n - 1] = theta_1[iterm1_n + (n - 1)];
  }

  /* ========== Update Output Arrays ========== */
  for (k = 0; k < n; k++) {
    theta_1[iter_n + k] = theta_1_new[k];                      // Store sampled states
    theta_1_updated[iter_n + k] = updated[k];                  // Store acceptance indicators
    alpha[iter_n + k] = ilogit(theta_1_new[k]);                // Store transformed probabilities
  }
}

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
                               int iter) {

  int k;

  /* ========== Iteration Index Setup and Conditional Normal SDs ========== */
  int iter_n = iter * n;                              // Current iteration start index
  int iterm1_n = iter_n - n;                          // Previous iteration start index
  int prev_iter = iter - 1;                           // Previous iteration for theta_01 and prec_theta_1

  /* Conditional Normal SDs from the RW1 prior:
   * Let inv_sqrt_prec = 1 / sqrt(prec_theta_1[prev_iter]).
   * - Interior nodes (k = 0..n-2): sd = inv_sqrt_prec / sqrt(2) = inv_sqrt_prec * M_SQRT1_2
   * - Boundary node (k = n-1):     sd = inv_sqrt_prec
   */
  const double sd_boundary = 1.0 / sqrt(prec_theta_1[prev_iter]);
  const double sd_interior = sd_boundary * M_SQRT1_2;  // 1/sqrt(2)

  /* ========== First Element (k = 0) ========== */
  double sigma_val = exp(log_sigma[0]);                // Proposal standard deviation

  // Conditional mean incorporating initial states from previous iteration
  hat_theta_1[0] = 0.5 * (theta_1[iterm1_n + 1] - theta_2[iter_n] +
    theta_01[prev_iter] + theta_02[prev_iter]);

  // Generate proposal from random walk
  theta_1_new[0] = rnorm(theta_1[iterm1_n], sigma_val);

  // Calculate log acceptance probability using previous iteration's precision
  double lp1n = dnorm(theta_1_new[0], hat_theta_1[0], sd_interior, 1);
  double lp2n = dbinom(y[0], n_trials, ilogit(theta_1_new[0]), 1);
  double lp1o = dnorm(theta_1[iterm1_n], hat_theta_1[0], sd_interior, 1);
  double lp2o = dbinom(y[0], n_trials, ilogit(theta_1[iterm1_n]), 1);

  log_accept_prob[0] = lp1n + lp2n - lp1o - lp2o;
  updated[0] = log(runif(0, 1)) <= log_accept_prob[0];

  if (!updated[0]) {
    theta_1_new[0] = theta_1[iterm1_n];               // Reject: keep old value
  }

  /* ========== Intermediate Elements (k = 1 to n-2) ========== */
  for (k = 1; k < (n - 1); k++) {
    sigma_val = exp(log_sigma[k]);

    // Conditional mean using forward-sampling: previously updated theta_1_new[k-1]
    hat_theta_1[k] = 0.5 * (theta_1[iterm1_n + k + 1] - theta_2[iter_n + k] +
      theta_1_new[k - 1] - theta_2[iter_n + k - 1]);

    theta_1_new[k] = rnorm(theta_1[iterm1_n + k], sigma_val);

    // Log densities using previous iteration's precision
    lp1n = dnorm(theta_1_new[k], hat_theta_1[k], sd_interior, 1);
    lp2n = dbinom(y[k], n_trials, ilogit(theta_1_new[k]), 1);
    lp1o = dnorm(theta_1[iterm1_n + k], hat_theta_1[k], sd_interior, 1);
    lp2o = dbinom(y[k], n_trials, ilogit(theta_1[iterm1_n + k]), 1);

    log_accept_prob[k] = lp1n + lp2n - lp1o - lp2o;
    updated[k] = log(runif(0, 1)) <= log_accept_prob[k];

    if (!updated[k]) {
      theta_1_new[k] = theta_1[iterm1_n + k];
    }
  }

  /* ========== Last Element (k = n-1) ========== */
  sigma_val = exp(log_sigma[n - 1]);

  // Simplified conditional mean: no future state dependency
  hat_theta_1[n - 1] = theta_1_new[n - 2] + theta_2[iter_n + n - 2];

  theta_1_new[n - 1] = rnorm(theta_1[iterm1_n + (n - 1)], sigma_val);

  // Log densities using previous iteration's precision (different structure for boundary)
  lp1n = dnorm(theta_1_new[n - 1], hat_theta_1[n - 1], sd_boundary, 1);
  lp2n = dbinom(y[n - 1], n_trials, ilogit(theta_1_new[n - 1]), 1);
  lp1o = dnorm(theta_1[iterm1_n + (n - 1)], hat_theta_1[n - 1], sd_boundary, 1);
  lp2o = dbinom(y[n - 1], n_trials, ilogit(theta_1[iterm1_n + (n - 1)]), 1);

  log_accept_prob[n - 1] = lp1n + lp2n - lp1o - lp2o;
  updated[n - 1] = log(runif(0, 1)) <= log_accept_prob[n - 1];

  if (!updated[n - 1]) {
    theta_1_new[n - 1] = theta_1[iterm1_n + (n - 1)];
  }

  /* ========== Update Output Arrays ========== */
  for (k = 0; k < n; k++) {
    theta_1[iter_n + k] = theta_1_new[k];                      // Store sampled states
    theta_1_updated[iter_n + k] = updated[k];                  // Store acceptance indicators
    alpha[iter_n + k] = ilogit(theta_1_new[k]);                // Store transformed probabilities
  }
}
