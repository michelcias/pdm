/**
 * @file cwmh_adaptive.c
 * @brief Adaptive parameter tuning for component-wise Metropolis-Hastings (CWMH) sampling in the
 *        pdm package.
 * @author Michel H. Montoril
 * @date 2025-08-09
 * @version 1.0
 *
 * @details This file contains functions for adaptive MCMC algorithms, including:
 *          - Online adaptation of proposal variance (log_sigma) in component-wise Metropolis-
 *            Hastings chains
 *          - Monitoring acceptance rates via sliding windows
 *          - Diminishing adaptation for robust convergence
 */

#include <R.h>
#include <Rmath.h>
#include "cwmh_adaptive.h"

// Symbolic constants for numerical stability and readability
#define MIN_DEVIATION_THRESHOLD 1e-12
#define POSITIVE_STEP_DIRECTION 1.0
#define NEGATIVE_STEP_DIRECTION -1.0

/**
 * @brief Adapts proposal variance for each component in a Component-Wise Metropolis-Hastings
 *        (CWMH) MCMC sampler.
 *
 * @details This function updates the proposal log standard deviations (log_sigma) for each
 *          dimension to achieve a target acceptance rate in adaptive MCMC. The adaptation uses a
 *          sliding window of recent updates to compute acceptance rates for each component, then
 *          adjusts log_sigma towards the target acceptance rate using a diminishing adaptation
 *          step.
 *
 *          The adaptation step size is computed as:
 *              step_size = min(max_step_size, base_adaptation_rate / iter^decay_exponent)
 *          The log_sigma update rule is:
 *              log_sigma[k] += sign(accept_prop[k] - target_acceptance) * step_size
 *          where accept_prop[k] is the observed acceptance rate for component k in the last
 *          lag_update iterations.
 *
 * @param theta_updated   Vectorized (B x n) matrix of acceptance indicators (1 if accepted,
 *                        0 if not).
 * @param acceptance_probs Output vector (size n) of acceptance proportions for each parameter component.
 * @param log_sigma       Input/output vector (size n) of log proposal standard deviations
 *                        (updated in-place).
 * @param lag_update      Number of recent iterations to use for acceptance rate calculation
 *                        (sliding window).
 * @param n               Number of components (dimensions) in the parameter vector.
 * @param iter            Current MCMC iteration (0-based). Must be >= lag_update.
 * @param max_step_size   Maximum adaptation step size.
 * @param base_adaptation_rate  Base adaptation rate (initial step size).
 * @param decay_exponent  Exponent controlling decay speed of adaptation step size
 *                        (e.g. 0.5 = sqrt, 1.0 = linear).
 * @param target_acceptance     Target acceptance rate for parameter optimization (e.g. 0.44
 *                        univariate, 0.234 multivariate).
 * @return None (results are written to accept_prop and log_sigma).
 *
 * @note Complexity: O(n * lag_update) per call.
 * @note Requires iter >= lag_update for valid adaptation history.
 * @note Common choices: max_step_size = 0.01, base_adaptation_rate = 1.0-10.0,
 *       decay_exponent = 0.3-0.8.
 * @note Numerical stability depends on adaptation parameters.
 *
 * @warning Results are invalid if theta_updated does not contain sufficient history
 *          (iter < lag_update).
 * @warning Inappropriate adaptation rates and step sizes may lead to poor MCMC mixing.
 *
 */
void adapt_cwmh_parameters(double *theta_updated,
                           double *accept_prop,
                           double *log_sigma,
                           int lag_update,
                           int n,
                           int iter,
                           double max_step_size,
                           double base_adaptation_rate,
                           double decay_exponent,
                           double target_acceptance) {

  // Robust input validation
  if (theta_updated == NULL) {
    error("theta_updated cannot be NULL");
  }
  if (accept_prop == NULL) {
    error("accept_prop cannot be NULL");
  }
  if (log_sigma == NULL) {
    error("log_sigma cannot be NULL");
  }
  if (iter < lag_update) {
    error("iter (%d) must be >= lag_update (%d)", iter, lag_update);
  }
  if (n <= 0) {
    error("n must be positive, got %d", n);
  }
  if (lag_update <= 0) {
    error("lag_update must be positive, got %d", lag_update);
  }
  if (max_step_size <= 0.0) {
    error("max_step_size must be positive, got %f", max_step_size);
  }
  if (base_adaptation_rate <= 0.0) {
    error("base_adaptation_rate must be positive, got %f", base_adaptation_rate);
  }
  if (decay_exponent <= 0.0) {
    error("decay_exponent must be positive, got %f", decay_exponent);
  }
  if (target_acceptance <= 0.0 || target_acceptance >= 1.0) {
    error("target_acceptance must be in (0,1), got %f", target_acceptance);
  }

  int start_idx = (iter - lag_update) * n;
  int end_idx = iter * n;
  int k, row;

  // Compute step_size only once per call
  double step_size = fmin2(max_step_size,
                           base_adaptation_rate / R_pow((double)iter, decay_exponent));

  for (k = 0; k < n; k++) {
    accept_prop[k] = 0.0;

    // Sum acceptances over the sliding window
    for (row = start_idx; row < end_idx; row += n) {
      accept_prop[k] += theta_updated[row + k];
    }

    // Average acceptance rate for component k
    accept_prop[k] /= (double)lag_update;

    // Update log_sigma towards target acceptance
    double deviation = accept_prop[k] - target_acceptance;
    
    // Only update log_sigma when |deviation| > 1e-12 to avoid numerically insignificant updates
    if (fabs(deviation) > MIN_DEVIATION_THRESHOLD) {
      log_sigma[k] += (deviation > 0 ? POSITIVE_STEP_DIRECTION : NEGATIVE_STEP_DIRECTION) * step_size;
    }
  }
}
