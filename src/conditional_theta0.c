#include <R.h>
#include <Rmath.h>
#include "conditional_theta0.h"

/**
 * Generates a sample from the full conditional posterior distribution of theta_01
 * within a Gibbs sampler iteration.
 *
 * This function updates the value of theta_01 for the current MCMC iteration (`iter`)
 * based on a Normal prior and likelihood information derived from other parameters.
 * The conditional posterior distribution for theta_01 is assumed to be Normal.
 *
 * Calculation steps:
 * 1. Computes the variance of the conditional posterior distribution by combining
 *    the prior precision and the precision associated with theta_1.
 * 2. Computes the mean of the conditional posterior distribution as a precision-weighted
 *    average of the prior mean and information from theta_1 and theta_02.
 * 3. Samples a new value for theta_01 from the resulting Normal distribution using rnorm().
 *
 * @param theta_01_post       Output array storing posterior samples of theta_01. The value
 *                            at index `iter` will be updated.
 * @param theta_02_post       Array storing posterior samples of theta_02. Uses value at `iter`.
 * @param theta_1_post        Array storing posterior samples of theta_1 (vectorized B x n matrix).
 *                            Uses the first element (index 0) of the vector from the *previous*
 *                            iteration (`iter - 1`).
 * @param prec_theta_1_post   Array storing posterior samples of the precision related to theta_1.
 *                            Uses value at `iter`.
 * @param mean_theta_01       Prior mean for theta_01.
 * @param prec_theta_01       Prior precision (inverse variance) for theta_01.
 * @param n                   Sample size, representing the number of data points.
 * @param iter                Current MCMC iteration index (0-based). Assumes iter > 0.
 */
void generate_theta_01(
    double *theta_01_post,
    double *theta_02_post,
    double *theta_1_post,
    double *prec_theta_1_post,
    double mean_theta_01,
    double prec_theta_01,
    int n,
    int iter
) {
  // Compute the posterior variance of theta_01
  double var_theta_01_post = 1.0 / (prec_theta_01 + prec_theta_1_post[iter]);

  // Compute the posterior mean of theta_01
  // Note: Accessing theta_1 from the previous iteration (iter - 1)
  double mean_theta_01_post = (mean_theta_01 * prec_theta_01 +
                               (theta_1_post[(iter - 1) * n] - theta_02_post[iter]) *
                               prec_theta_1_post[iter]) * var_theta_01_post;

  // Sample theta_01 from its conditional posterior Normal distribution
  theta_01_post[iter] = rnorm(mean_theta_01_post, sqrt(var_theta_01_post));
}

//----------------------------------------------------------------------

/**
 * Generates a sample from the full conditional posterior distribution of theta_0k
 * (for indices k from 2 to p-1) within a Gibbs sampler iteration.
 *
 * This function updates the value of theta_0k for the current MCMC iteration (`iter`)
 * based on a Normal prior and likelihood information derived from other parameters.
 * It handles the general case for the levels between the first (k=1) and the last (k=p).
 * The conditional posterior distribution for theta_0k is assumed to be Normal.
 *
 * Calculation steps:
 * 1. Computes the variance of the conditional posterior distribution by combining the
 *    prior precision and the precisions associated with theta_(k-1) and theta_k.
 * 2. Computes the mean of the conditional posterior distribution as a precision-weighted
 *    average of the prior mean and information derived from related parameters.
 * 3. Samples a new value for theta_0k from the resulting Normal distribution using rnorm().
 *
 * @param theta_0km1_post     Array storing posterior samples of theta_0(k-1). Uses value at `iter - 1`.
 * @param theta_0k_post       Output array storing posterior samples of theta_0k. The value
 *                            at index `iter` will be updated.
 * @param theta_0kp1_post     Array storing posterior samples of theta_0(k+1). Uses value at `iter`.
 * @param theta_km1_post      Array storing posterior samples of theta_(k-1) (vectorized B x n matrix).
 *                            Uses the first element (index 0) of the vector from `iter - 1`.
 * @param theta_k_post        Array storing posterior samples of theta_k (vectorized B x n matrix).
 *                            Uses the first element (index 0) of the vector from `iter - 1`.
 * @param prec_theta_km1_post Array storing posterior samples of the precision related to theta_(k-1).
 *                            Uses value at `iter - 1`.
 * @param prec_theta_k_post   Array storing posterior samples of the precision related to theta_k.
 *                            Uses value at `iter`.
 * @param mean_theta_0k       Prior mean for theta_0k.
 * @param prec_theta_0k       Prior precision (inverse variance) for theta_0k.
 * @param n                   Sample size, used for indexing the vectorized state parameter arrays
 *                            theta_km1_post and theta_k_post.
 * @param iter                Current MCMC iteration index (0-based). Assumes iter > 0.
 */
void generate_theta_0k(
    double *theta_0km1_post,
    double *theta_0k_post,
    double *theta_0kp1_post,
    double *theta_km1_post,
    double *theta_k_post,
    double *prec_theta_km1_post,
    double *prec_theta_k_post,
    double mean_theta_0k,
    double prec_theta_0k,
    int n,
    int iter
) {
  // Compute the posterior variance of theta_0k
  // Combines prior precision and precisions from related states (k-1 and k)
  double var_theta_0k_post = 1.0 / (prec_theta_0k + prec_theta_km1_post[iter - 1]
                                      + prec_theta_k_post[iter]);

  // Compute the posterior mean of theta_0k
  // Weighted average of prior mean and terms derived from state differences
  // Note the mix of iterations (iter and iter - 1) being accessed
  double mean_theta_0k_post = (mean_theta_0k * prec_theta_0k +
  (theta_km1_post[(iter - 1) * n] - theta_0km1_post[iter - 1]) * // Info from k-1 state (iter - 1)
  prec_theta_km1_post[iter - 1] +
  (theta_k_post[(iter - 1) * n] - theta_0kp1_post[iter]) *       // Info from k state (iter - 1 for theta_k, iter for theta_0(k+1))
  prec_theta_k_post[iter]) * var_theta_0k_post;

  // Sample theta_0k from its conditional posterior Normal distribution
  theta_0k_post[iter] = rnorm(mean_theta_0k_post, sqrt(var_theta_0k_post));
}

//----------------------------------------------------------------------

/**
 * Generates a sample from the full conditional posterior distribution of theta_0p
 * (the last level, k=p) within a Gibbs sampler iteration.
 *
 * This function updates the value of theta_0p for the current MCMC iteration (`iter`)
 * based on a Normal prior and likelihood information derived from other parameters.
 * It specifically handles the boundary case for the last level p.
 * The conditional posterior distribution for theta_0p is assumed to be Normal.
 *
 * Calculation steps:
 * 1. Computes the variance of the conditional posterior distribution by combining the
 *    prior precision and the precisions associated with theta_(p-1) and theta_p.
 * 2. Computes the mean of the conditional posterior distribution as a precision-weighted
 *    average of the prior mean and information derived from related parameters. Note
 *    the term related to theta_p does not involve subtraction of a higher-level baseline.
 * 3. Samples a new value for theta_0p from the resulting Normal distribution using rnorm().
 *
 * @param theta_0pm1_post     Array storing posterior samples of theta_0(p-1). Uses value at `iter - 1`.
 * @param theta_0p_post       Output array storing posterior samples of theta_0p. The value
 *                            at index `iter` will be updated.
 * @param theta_pm1_post      Array storing posterior samples of theta_(p-1) (vectorized B x n matrix).
 *                            Uses the first element (index 0) of the vector from `iter - 1`.
 * @param theta_p_post        Array storing posterior samples of theta_p (vectorized B x n matrix).
 *                            Uses the first element (index 0) of the vector from `iter - 1`.
 * @param prec_theta_pm1_post Array storing posterior samples of the precision related to theta_(p-1).
 *                            Uses value at `iter - 1`.
 * @param prec_theta_p_post   Array storing posterior samples of the precision related to theta_p.
 *                            Uses value at `iter`.
 * @param mean_theta_0p       Prior mean for theta_0p.
 * @param prec_theta_0p       Prior precision (inverse variance) for theta_0p.
 * @param n                   Sample size, used for indexing the vectorized state parameter arrays
 *                            theta_pm1_post and theta_p_post.
 * @param iter                Current MCMC iteration index (0-based). Assumes iter > 0.
 */
void generate_theta_0p(
    double *theta_0pm1_post,
    double *theta_0p_post,
    double *theta_pm1_post,
    double *theta_p_post,
    double *prec_theta_pm1_post,
    double *prec_theta_p_post,
    double mean_theta_0p,
    double prec_theta_0p,
    int n,
    int iter
) {
  // Compute the posterior variance of theta_0p
  // Combines prior precision and precisions from related states (p-1 and p)
  double var_theta_0p_post = 1.0 / (prec_theta_0p + prec_theta_pm1_post[iter - 1]
                                      + prec_theta_p_post[iter]);

  // Compute the posterior mean of theta_0p
  // Weighted average of prior mean and terms derived from state differences/values
  // Note the term for state p uses only theta_p_post (no theta_0(p+1))
  double mean_theta_0p_post = (mean_theta_0p * prec_theta_0p +
  (theta_pm1_post[(iter - 1) * n] - theta_0pm1_post[iter - 1]) * // Info from p-1 state (iter - 1)
  prec_theta_pm1_post[iter - 1] +
  (theta_p_post[(iter - 1) * n]) *                               // Info from p state (iter - 1)
  prec_theta_p_post[iter]) *
  var_theta_0p_post;

  // Sample theta_0p from its conditional posterior Normal distribution
  theta_0p_post[iter] = rnorm(mean_theta_0p_post, sqrt(var_theta_0p_post));
}
