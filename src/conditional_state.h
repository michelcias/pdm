#ifndef CONDITIONAL_STATE_H
#define CONDITIONAL_STATE_H

/**
 * Generates a sample for the state vector theta_1 (local trend component)
 * assuming a simple local trend model (theta_1[j] = theta_1[j-1] + error).
 * This is a step within a Gibbs sampler iteration.
 *
 * @param data                Array (size n) of observed data points, indexed 0 to n-1.
 * @param theta_1_post        Output array (vectorized B x n matrix) storing posterior samples of theta_1.
 * @param prec_data_post      Array storing posterior samples of data precision (1/V). Uses value from `iter - 1`.
 * @param prec_theta_1_post   Array storing posterior samples of theta_1 innovation precision (1/W_1). Uses value from `iter - 1`.
 * @param theta_01_post       Array storing posterior samples of the initial level theta_01. Uses value from `iter`.
 * @param n                   Sample size (number of data points/state elements, 0 to n-1). Must be > 2.
 * @param iter                Current MCMC iteration index (0-based). Assumes iter > 0.
 */
void generate_theta_1_localtrend(double *data,
                                 double *theta_1_post,
                                 double *prec_data_post,
                                 double *prec_theta_1_post,
                                 double *theta_01_post,
                                 int n,
                                 int iter);

/**
 * Generates a sample for the state vector theta_1 (local level component)
 * assuming a local linear trend model (theta_1 depends on theta_2).
 * This is a step within a Gibbs sampler iteration.
 *
 * @param data                Array (size n) of observed data points, indexed 0 to n-1.
 * @param theta_1_post        Output array (vectorized B x n matrix) storing posterior samples of theta_1.
 * @param theta_2_post        Array storing posterior samples of theta_2 (vectorized B x n matrix).
 * @param prec_data_post      Array storing posterior samples of data precision (1/V). Uses value from `iter - 1`.
 * @param prec_theta_1_post   Array storing posterior samples of theta_1 innovation precision (1/W_1). Uses value from `iter - 1`.
 * @param theta_01_post       Array storing posterior samples of the initial level theta_01. Uses value from `iter`.
 * @param theta_02_post       Array storing posterior samples of the initial trend theta_02. Uses value from `iter`.
 * @param n                   Sample size (number of data points/state elements, 0 to n-1). Must be > 2.
 * @param iter                Current MCMC iteration index (0-based). Assumes iter > 0.
 */
void generate_theta_1(double *data,
                      double *theta_1_post,
                      double *theta_2_post,
                      double *prec_data_post,
                      double *prec_theta_1_post,
                      double *theta_01_post,
                      double *theta_02_post,
                      int n,
                      int iter);

/**
 * Generates a sample for the state vector theta_k (k from 2 to p-1)
 * within a Gibbs sampler iteration.
 *
 * @param theta_km1_post      Array storing posterior samples of theta_(k-1) (vectorized B x n matrix).
 * @param theta_k_post        Output array (vectorized B x n matrix) storing posterior samples of theta_k.
 * @param theta_kp1_post      Array storing posterior samples of theta_(k+1) (vectorized B x n matrix).
 * @param prec_theta_km1_post Array storing posterior samples of precision 1/W_{k-1}. Uses value from `iter - 1`.
 * @param prec_theta_k_post   Array storing posterior samples of precision 1/W_k. Uses value from `iter - 1`.
 * @param theta_0k_post       Array storing posterior samples of the baseline theta_0k. Uses value from `iter`.
 * @param theta_0kp1_post     Array storing posterior samples of the baseline theta_0(k+1). Uses value from `iter`.
 * @param n                   Sample size (number of elements in state vectors, 0 to n-1). Must be > 2.
 * @param iter                Current MCMC iteration index (0-based). Assumes iter > 0.
 */
void generate_theta_k(double *theta_km1_post,
                      double *theta_k_post,
                      double *theta_kp1_post,
                      double *prec_theta_km1_post,
                      double *prec_theta_k_post,
                      double *theta_0k_post,
                      double *theta_0kp1_post,
                      int n,
                      int iter);

/**
 * Generates a sample for the state vector theta_p (last level, k=p)
 * within a Gibbs sampler iteration.
 *
 * @param theta_pm1_post      Array storing posterior samples of theta_(p-1) (vectorized B x n matrix).
 * @param theta_p_post        Output array (vectorized B x n matrix) storing posterior samples of theta_p.
 * @param prec_theta_pm1_post Array storing posterior samples of precision 1/W_{p-1}. Uses value from `iter - 1`.
 * @param prec_theta_p_post   Array storing posterior samples of precision 1/W_p. Uses value from `iter - 1`.
 * @param theta_0p_post       Array storing posterior samples of the baseline theta_0p. Uses value from `iter`.
 * @param n                   Sample size (number of elements in state vectors, 0 to n-1). Must be > 2.
 * @param iter                Current MCMC iteration index (0-based). Assumes iter > 0.
 */
void generate_theta_p(double *theta_pm1_post,
                      double *theta_p_post,
                      double *prec_theta_pm1_post,
                      double *prec_theta_p_post,
                      double *theta_0p_post,
                      int n,
                      int iter);

#endif // CONDITIONAL_STATE_H
