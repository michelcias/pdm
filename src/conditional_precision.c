#include <R.h>
#include <Rmath.h>
#include "conditional_precision.h"

/**
 * Generates a sample from the full conditional posterior distribution of the precision 1/W_k
 * (for indices k from 1 to p-1) within a Gibbs sampler iteration.
 *
 * This function updates the precision associated with the innovations of state theta_k
 * for the current MCMC iteration (`iter`). It assumes a Gamma posterior distribution,
 * derived from a Gamma prior and a Normal likelihood for the state innovations.
 *
 * Calculation steps:
 * 1. Computes the sum of squared deviations (`ss_theta`). This sum represents the squared
 *    innovations based on the assumed model structure. Here, `j` is a 0-based index for elements
 *    within the state vectors. The innovation for element `j > 0` is calculated as
 *    `theta_k[j] - theta_k[j-1] - theta_(k+1)[j-1]`, while for `j = 0` (the first element)
 *    it is `theta_k[0] - theta_0k - theta_0(k+1)`.
 * 2. Updates the shape parameter (`nu_0k_post`) of the Gamma posterior: prior shape + n/2.
 * 3. Updates the rate parameter (`eta_0k_post`) of the Gamma posterior: prior rate + ss_theta/2.
 * 4. Samples a new value for the precision `prec_theta_k_post[iter]` from the resulting
 *    Gamma(shape, scale=1/rate) distribution using rgamma().
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
                                int iter) {

  int j;
  int iter_n = iter * n;     // Pre-calculate offset for current iteration

  // Compute sum of squared innovations based on the assumed state evolution model for theta_k
  // Innovation for element j=0 (first element): theta_k[0] - theta_0k - theta_0(k+1)
  double theta_centered = theta_k_post[iter_n] - theta_0k_post[iter - 1] - theta_0kp1_post[iter];
  double ss_theta = theta_centered * theta_centered;

  for (j = 1; j < n; j++) {
    // Innovation for element j>0 (0-based index): theta_k[j] - theta_k[j-1] - theta_(k+1)[j-1]
    theta_centered = theta_k_post[iter_n + j] - theta_k_post[iter_n + j - 1]
    - theta_kp1_post[iter_n + j - 1];
    ss_theta += theta_centered * theta_centered;
  }

  // Update Gamma posterior parameters
  double nu_0k_post = nu_0k + n / 2.0;            // Posterior shape = prior shape + n/2
  double eta_0k_post = eta_0k + ss_theta / 2.0;   // Posterior rate = prior rate + ss/2

  // Sample precision 1/W_k from Gamma(shape, scale) distribution
  // R's rgamma uses scale = 1 / rate
  prec_theta_k_post[iter] = rgamma(nu_0k_post, 1.0 / eta_0k_post);
}

//----------------------------------------------------------------------

/**
 * Generates a sample from the full conditional posterior distribution of the precision 1/W_p
 * (for the last level, k=p) within a Gibbs sampler iteration.
 *
 * This function updates the precision associated with the innovations of the last state theta_p
 * for the current MCMC iteration (`iter`). It assumes a Gamma posterior distribution,
 * derived from a Gamma prior and a Normal likelihood for the state innovations (likely a random walk).
 *
 * Calculation steps:
 * 1. Computes the sum of squared deviations (`ss_theta`). This sum represents the squared
 *    innovations based on the assumed model structure for `theta_p` (e.g., random walk).
 *    Here, `j` is a 0-based index for elements within the state vector. The innovation
 *    for element `j > 0` is calculated as `theta_p[j] - theta_p[j-1]`, while for `j = 0`
 *    (the first element) it is `theta_p[0] - theta_0p`.
 * 2. Updates the shape parameter (`nu_0p_post`) of the Gamma posterior: prior shape + n/2.
 * 3. Updates the rate parameter (`eta_0p_post`) of the Gamma posterior: prior rate + ss_theta/2.
 * 4. Samples a new value for the precision `prec_theta_p_post[iter]` from the resulting
 *    Gamma(shape, scale=1/rate) distribution using rgamma().
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
                                int iter) {

  int j;
  int iter_n = iter * n; // Pre-calculate offset for current iteration's data

  // Compute sum of squared innovations based on the assumed state evolution model for theta_p (e.g., random walk)
  // Innovation for element j=0 (first element): theta_p[0] - theta_0p
  double theta_centered = theta_p_post[iter_n] - theta_0p_post[iter - 1];
  double ss_theta = theta_centered * theta_centered;

  for (j = 1; j < n; j++) {
    // Innovation for element j>0 (0-based index): theta_p[j] - theta_p[j-1]
    theta_centered = theta_p_post[iter_n + j] - theta_p_post[iter_n + j - 1];
    ss_theta += theta_centered * theta_centered;
  }

  // Update Gamma posterior parameters
  double nu_0p_post = nu_0p + n / 2.0;            // Posterior shape = prior shape + n/2
  double eta_0p_post = eta_0p + ss_theta / 2.0;   // Posterior rate = prior rate + ss/2

  // Sample precision 1/W_p from Gamma(shape, scale) distribution
  // R's rgamma uses scale = 1 / rate
  prec_theta_p_post[iter] = rgamma(nu_0p_post, 1.0 / eta_0p_post);
}
