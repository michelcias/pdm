#ifndef CONDITIONAL_THETA0_H
#define CONDITIONAL_THETA0_H


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
 *    average of the prior mean and information from theta_1.
 * 3. Samples a new value for theta_01 from the resulting Normal distribution using rnorm().
 *
 * @param theta_01_post       Output array storing posterior samples of theta_01. The value
 *                            at index `iter` will be updated.
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
void generate_theta_01_localtrend(double *theta_01_post,
                                  double *theta_1_post,
                                  double *prec_theta_1_post,
                                  double mean_theta_01,
                                  double prec_theta_01,
                                  int n,
                                  int iter);


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
void generate_theta_01(double *theta_01_post,
                       double *theta_02_post,
                       double *theta_1_post,
                       double *prec_theta_1_post,
                       double mean_theta_01,
                       double prec_theta_01,
                       int n,
                       int iter);


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
void generate_theta_0k(double *theta_0km1_post,
                       double *theta_0k_post,
                       double *theta_0kp1_post,
                       double *theta_km1_post,
                       double *theta_k_post,
                       double *prec_theta_km1_post,
                       double *prec_theta_k_post,
                       double mean_theta_0k,
                       double prec_theta_0k,
                       int n,
                       int iter);


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
void generate_theta_0p(double *theta_0pm1_post,
                       double *theta_0p_post,
                       double *theta_pm1_post,
                       double *theta_p_post,
                       double *prec_theta_pm1_post,
                       double *prec_theta_p_post,
                       double mean_theta_0p,
                       double prec_theta_0p,
                       int n,
                       int iter);

#endif
