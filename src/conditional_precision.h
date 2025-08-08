#ifndef CONDITIONAL_PRECISION_H
#define CONDITIONAL_PRECISION_H

/**
 * Generates a sample from the full conditional posterior distribution of the data precision 1/V
 * within a Gibbs sampler iteration.
 *
 * This function updates the precision associated with the observation noise (data precision)
 * for the current MCMC iteration (`iter`). It assumes a Gamma posterior distribution,
 * derived from a Gamma prior and a Normal likelihood for the data.
 *
 * Calculation steps:
 * 1. Computes the sum of squared residuals (`ss_y`): \sum_{i=1}^n (y_i - theta_1_post[i])^2
 * 2. Updates the shape parameter (`nu_y_post`) of the Gamma posterior: prior shape + n/2.
 * 3. Updates the rate parameter (`eta_y_post`) of the Gamma posterior: prior rate + ss_y/2.
 * 4. Samples a new value for the data precision `prec_y_post[iter]` from the resulting
 *    Gamma(shape, scale=1/rate) distribution using rgamma().
 *
 * @param y              Array of observed data (length n).
 * @param theta_1_post   Array storing posterior samples of theta_1 (vectorized B x n matrix).
 *                       Uses the vector corresponding to `iter`. Elements indexed 0 to n-1.
 * @param prec_y_post    Output array storing posterior samples of the data precision 1/V.
 *                       The value at index `iter` will be updated.
 * @param nu_y           Prior shape parameter for the Gamma distribution of 1/V.
 * @param eta_y          Prior rate parameter for the Gamma distribution of 1/V.
 * @param n              Sample size, representing the number of elements (0 to n-1).
 * @param iter           Current MCMC iteration index (0-based).
 */
void generate_precision_data(double *y,
                             double *theta_1_post,
                             double *prec_y_post,
                             double nu_y,
                             double eta_y,
                             int n,
                             int iter);
/**
 * Generates a sample from the full conditional posterior distribution of the precision 1/W_k
 * (for indices k from 1 to p-1) within a Gibbs sampler iteration.
 *
 * @param theta_0k_post       Array storing posterior samples of theta_0k. Uses value at `iter`.
 * @param theta_0kp1_post     Array storing posterior samples of theta_0(k+1). Uses value at `iter`.
 * @param theta_k_post        Array storing posterior samples of theta_k (vectorized B x n matrix).
 *                            Uses the vector corresponding to `iter`. Elements indexed 0 to n-1.
 * @param theta_kp1_post      Array storing posterior samples of theta_(k+1) (vectorized B x n matrix).
 *                            Uses the vector corresponding to `iter`. Elements indexed 0 to n-1.
 * @param prec_theta_k_post   Output array storing posterior samples of the precision 1/W_k.
 *                            The value at index `iter` will be updated.
 * @param nu_0k               Prior shape parameter for the Gamma distribution of 1/W_k.
 * @param eta_0k              Prior rate parameter for the Gamma distribution of 1/W_k.
 * @param n                   Sample size, representing the number of elements (0 to n-1) in the state vectors.
 * @param iter                Current MCMC iteration index (0-based).
 */
void generate_precision_theta_k(double *theta_0k_post,
                                double *theta_0kp1_post,
                                double *theta_k_post,
                                double *theta_kp1_post,
                                double *prec_theta_k_post,
                                double nu_0k,
                                double eta_0k,
                                int n,
                                int iter);

/**
 * Generates a sample from the full conditional posterior distribution of the precision 1/W_p
 * (for the last level, k=p) within a Gibbs sampler iteration.
 *
 * @param theta_0p_post       Array storing posterior samples of theta_0p. Uses value at `iter`.
 * @param theta_p_post        Array storing posterior samples of theta_p (vectorized B x n matrix).
 *                            Uses the vector corresponding to `iter`. Elements indexed 0 to n-1.
 * @param prec_theta_p_post   Output array storing posterior samples of the precision 1/W_p.
 *                            The value at index `iter` will be updated.
 * @param nu_0p               Prior shape parameter for the Gamma distribution of 1/W_p.
 * @param eta_0p              Prior rate parameter for the Gamma distribution of 1/W_p.
 * @param n                   Sample size, representing the number of elements (0 to n-1) in the state vector.
 * @param iter                Current MCMC iteration index (0-based).
 */
void generate_precision_theta_p(double *theta_0p_post,
                                double *theta_p_post,
                                double *prec_theta_p_post,
                                double nu_0p,
                                double eta_0p,
                                int n,
                                int iter);

#endif /* CONDITIONAL_PRECISION_H */
