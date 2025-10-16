/**
 * @file generate_normal_mixture_parameters.c
 * @brief Gibbs sampling for Gaussian mixture model component parameters
 * @author Michel H. Montoril
 * @date 2025-10-14
 * @version 1.0
 *
 * @details This file implements Gibbs sampling for component parameters (means and
 *          precisions) of Gaussian mixture models with dynamic mixture weights.
 *
 *          Current implementation:
 *          - Two-component Gaussian mixture (k = 2)
 *
 *          Planned extensions:
 *          - K-component Gaussian mixture (k >= 2)
 *          - Mixture of other distributions (Poisson, Binomial, etc.)
 *
 *          Model specification:
 *          y_t | z_t, mu, phi ~ N(z_t * mu_2 + (1 - z_t) * mu_1,
 *                                  (z_t * phi_2 + (1 - z_t) * phi_1)^{-1})
 *          z_t | alpha_t ~ Bernoulli(alpha_t)
 *
 *          where mu = (mu_1, mu_2)' are component means, phi = (phi_1, phi_2)' are
 *          component precisions, and alpha_t are dynamic mixture weights estimated
 *          separately.
 *
 *          Prior distributions:
 *          mu_k ~ N(mu_0k, sigma^2_0k)  <=>  mu_k ~ N(mu_0k, (prec_0k)^{-1})
 *          phi_k ~ Gamma(nu_0k, eta_0k)
 *
 *          Full conditional posteriors:
 *          mu_k | y, [...] ~ N(mu_bar_k, sigma_bar^2_k)
 *          phi_k | y, [...] ~ Gamma(nu_bar_k, eta_bar_k)
 *
 *          where:
 *          sigma_bar^2_k = (T_k * phi_k + prec_0k)^{-1}
 *          mu_bar_k = sigma_bar^2_k * (s_k * phi_k + mu_0k * prec_0k)
 *          nu_bar_k = nu_0k + T_k / 2
 *          eta_bar_k = eta_0k + v_k / 2
 *
 *          with T_k = #{z_t = k-1}, s_k = Sum_{t: z_t = k-1} y_t,
 *          v_k = Sum_{t: z_t = k-1} (y_t - mu_k)^2
 *
 *          Label switching constraint:
 *          To ensure identifiability, the constraint mu_1 < mu_2 is enforced by
 *          swapping components when necessary (mu_1 is the "lower" component,
 *          mu_2 is the "upper").
 *
 *          Integration with dynamic weights:
 *          This function samples component parameters conditional on latent indicators
 *          z_t, which are themselves sampled conditional on dynamic mixture weights
 *          alpha_t. The alpha_t are estimated using component-wise Metropolis-Hastings
 *          or probit link methods for nonlinear dynamic models.
 *
 *          Parameter vector structure:
 *          The parameter vector params has length 2*k where k is the number of
 *          components. For k=2:
 *          params[0] = mu_1    (mean of lower component)
 *          params[1] = prec_1  (precision of lower component)
 *          params[2] = mu_2    (mean of upper component)
 *          params[3] = prec_2  (precision of upper component)
 *
 *          This structure facilitates:
 *          - Natural extension to k>2 components
 *          - Direct interface with R matrices
 *          - Memory-efficient MCMC iterations
 *          - Cache-friendly contiguous memory layout
 */

#include <R.h>
#include <Rmath.h>
#include "generate_normal_mixture_parameters.h"

/**
 * @brief Generate component parameters for a two-component Gaussian mixture model
 *
 * @details Implements one Gibbs sampling iteration for the component parameters
 *          (mu_1, phi_1, mu_2, phi_2) in a two-component Gaussian mixture model.
 *          The function:
 *
 *          1. Computes sufficient statistics (counts, sums) for each component based on z_t
 *          2. Samples means from their full conditional Normal posteriors
 *          3. Computes sum of squared deviations using newly sampled means
 *          4. Samples precisions from their full conditional Gamma posteriors
 *          5. Enforces identifiability constraint mu_1 < mu_2 by swapping if necessary
 *          6. Stores results in params_current vector
 *
 *          Algorithm:
 *          The sufficient statistics are computed in a single pass through the data:
 *          - T_0 = count of observations assigned to component 1 (z_t = 0)
 *          - T_1 = count of observations assigned to component 2 (z_t = 1)
 *          - s_0 = sum of y_t for z_t = 0
 *          - s_1 = sum of y_t for z_t = 1
 *
 *          After sampling means, another pass computes squared deviations v_0 and v_1
 *          using the newly sampled means. This ordering ensures consistency with the
 *          full conditional posteriors.
 *
 *          Full conditional posterior for means:
 *          Given the conjugate Normal-Normal model, the posterior for mu_k is:
 *
 *          mu_k | y, z, phi_k, [...] ~ N(mu_bar_k, sigma_bar^2_k)
 *
 *          where the posterior precision is:
 *          1 / sigma_bar^2_k = prec_0k + T_k * phi_k
 *
 *          and the posterior mean is:
 *          mu_bar_k = sigma_bar^2_k * (mu_0k * prec_0k + s_k * phi_k)
 *
 *          This is the standard conjugate update for the normal mean with known precision.
 *
 *          Full conditional posterior for precisions:
 *          Given the conjugate Gamma-Normal model, the posterior for phi_k is:
 *
 *          phi_k | y, z, mu_k, [...] ~ Gamma(nu_bar_k, eta_bar_k)
 *
 *          where:
 *          nu_bar_k = nu_0k + T_k / 2
 *          eta_bar_k = eta_0k + v_k / 2
 *
 *          with v_k = Sum_{t: z_t = k-1} (y_t - mu_k)^2 being the sum of squared
 *          deviations for component k.
 *
 *          Label switching:
 *          To ensure identifiability of the mixture components, we enforce the
 *          ordering constraint mu_1 < mu_2. If this constraint is violated after
 *          sampling, the components are swapped. This is a common approach to
 *          address the label switching problem in mixture models.
 *
 *          Numerical stability:
 *          - Uses R's rnorm and rgamma for consistent random number generation
 *          - Computes posterior variance directly from posterior precision
 *          - Handles edge cases where T_k = 0 gracefully (prior dominates)
 *          - All intermediate calculations use double precision
 *
 *          Memory efficiency:
 *          - Single-pass computation of sufficient statistics
 *          - Contiguous memory layout for cache efficiency
 *          - In-place updates of output parameters
 *          - No dynamic memory allocation
 *
 *          Parameter vector indexing:
 *          params_previous and params_current are vectors of length 4:
 *          [0] = mu_1    (mean of component with smaller mean)
 *          [1] = prec_1  (precision of component with smaller mean)
 *          [2] = mu_2    (mean of component with larger mean)
 *          [3] = prec_2  (precision of component with larger mean)
 *
 * @param y                  Observed data vector [n] (const, read-only).
 *                           Contains the response values y_1, ..., y_n.
 * @param z                  Latent indicator vector [n] (const, read-only).
 *                           Each z[t] is 0.0 (component 1) or 1.0 (component 2).
 *                           Generated from full conditional:
 *                           P(z_t = 1 | y_t, alpha_t, mu, phi) proportional to
 *                           alpha_t * N(y_t | mu_2, phi_2^{-1}).
 * @param params_previous    Parameter vector from previous iteration [4] (const, read-only).
 *                           Structure: [mu_1, prec_1, mu_2, prec_2].
 *                           Used for computing posterior distributions.
 * @param params_current     Output parameter vector for current iteration [4].
 *                           Will be filled with [mu_1, prec_1, mu_2, prec_2].
 *                           Must be pre-allocated with size 4.
 * @param mu_01              Prior mean for mu_1 (lower component).
 *                           Typical values: sample mean or domain knowledge.
 * @param prec_01            Prior precision for mu_1 (inverse of prior variance sigma^2_01).
 *                           Typical values: 0.01 for vague prior, higher for informative.
 * @param nu_01              Prior shape parameter for phi_1 (Gamma distribution).
 *                           Typical values: 0.01 for vague prior, higher for informative.
 * @param eta_01             Prior rate parameter for phi_1 (Gamma distribution).
 *                           Typical values: 0.01 for vague prior, higher for informative.
 * @param mu_02              Prior mean for mu_2 (upper component).
 *                           Typical values: sample mean or domain knowledge.
 * @param prec_02            Prior precision for mu_2 (inverse of prior variance sigma^2_02).
 *                           Typical values: 0.01 for vague prior, higher for informative.
 * @param nu_02              Prior shape parameter for phi_2 (Gamma distribution).
 *                           Typical values: 0.01 for vague prior, higher for informative.
 * @param eta_02             Prior rate parameter for phi_2 (Gamma distribution).
 *                           Typical values: 0.01 for vague prior, higher for informative.
 * @param n                  Sample size (length of y and z vectors).
 *
 * @return None (results are written to params_current).
 *
 * @note Computational complexity: O(n) per iteration (two passes through data).
 * @note Memory requirements: O(1) additional space beyond input/output arrays.
 * @note Conjugate posteriors: Uses exact Gibbs sampling (no Metropolis-Hastings needed).
 * @note Label switching: Enforces mu_1 < mu_2 constraint for identifiability.
 * @note Integration: Designed to work with dynamic mixture weight estimation methods.
 * @note Prior specification: Gamma(nu, eta) parameterization uses rate (not scale).
 * @note Parameter structure: Uses contiguous vector for scalability to k>2 components.
 * @note R interface: params_current can be directly passed from R matrix columns.
 * @note Sufficient statistics: Computed using double precision to avoid numerical issues.
 *
 * @warning All prior hyperparameters must be positive and finite.
 * @warning params_previous[1] and params_previous[3] (precisions) must be positive.
 * @warning Each z[t] must be exactly 0.0 or 1.0.
 * @warning n must be > 0.
 * @warning Handle case where all z[t] = 0 or all z[t] = 1 carefully (prior dominates).
 * @warning params_previous must have length 4 and params_current must be pre-allocated
 *          with length 4.
 * @warning For numerical stability, ensure prec_01, prec_02 > 0 and nu_01, nu_02,
 *          eta_01, eta_02 > 0.
 *
 * @see generate_mixture_normal_k (future K-component generalization)
 * @see Roberts, G. O., & Rosenthal, J. S. (2009). Examples of adaptive MCMC.
 *      Journal of Computational and Graphical Statistics, 18(2), 349-367.
 *
 * @example
 * @code
 * // Setup for MCMC iteration
 * double params_prev[4] = {0.0, 4.0, 2.0, 4.0};  // initial values
 * double params_curr[4];
 *
 * // Generate component parameters for current iteration
 * generate_mixture_normal_2(
 *     y,                      // observed data
 *     z,                      // latent indicators
 *     params_prev,            // previous [mu_1, prec_1, mu_2, prec_2]
 *     params_curr,            // output: current [mu_1, prec_1, mu_2, prec_2]
 *     0.0,                    // mu_01: prior mean for component 1
 *     0.01,                   // prec_01: prior precision for mu_1
 *     0.01,                   // nu_01: prior shape for phi_1
 *     0.01,                   // eta_01: prior rate for phi_1
 *     2.0,                    // mu_02: prior mean for component 2
 *     0.01,                   // prec_02: prior precision for mu_2
 *     0.01,                   // nu_02: prior shape for phi_2
 *     0.01,                   // eta_02: prior rate for phi_2
 *     n                       // sample size
 * );
 *
 * // Access results
 * double mu_1 = params_curr[0];
 * double prec_1 = params_curr[1];
 * double mu_2 = params_curr[2];
 * double prec_2 = params_curr[3];
 *
 * // Copy for next iteration
 * memcpy(params_prev, params_curr, 4 * sizeof(double));
 * @endcode
 *
 * @version 1.0
 */
void generate_mixture_normal_2(const double *y,
                               const double *z,
                               const double *params_previous,
                               double       *params_current,
                               double        mu_01,
                               double        prec_01,
                               double        nu_01,
                               double        eta_01,
                               double        mu_02,
                               double        prec_02,
                               double        nu_02,
                               double        eta_02,
                               int           n) {

  int t;
  double T_0 = 0.0, T_1 = 0.0;
  double s_0 = 0.0, s_1 = 0.0;
  double v_0, v_1;
  double diff;

  /* Extract previous iteration parameters for readability */
  double mu_1_prev = params_previous[0];
  double prec_1_prev = params_previous[1];
  double mu_2_prev = params_previous[2];
  double prec_2_prev = params_previous[3];

  /* ========== Input Validation ========== */
  if (y == NULL) {
    error("y cannot be NULL");
  }
  if (z == NULL) {
    error("z cannot be NULL");
  }
  if (params_previous == NULL) {
    error("params_previous cannot be NULL");
  }
  if (params_current == NULL) {
    error("params_current cannot be NULL");
  }
  if (n <= 0) {
    error("n must be positive, got %d", n);
  }
  if (prec_1_prev <= 0.0 || !R_FINITE(prec_1_prev)) {
    error("params_previous[1] (prec_1) must be positive and finite, got %f", prec_1_prev);
  }
  if (prec_2_prev <= 0.0 || !R_FINITE(prec_2_prev)) {
    error("params_previous[3] (prec_2) must be positive and finite, got %f", prec_2_prev);
  }
  if (prec_01 <= 0.0 || !R_FINITE(prec_01)) {
    error("prec_01 must be positive and finite, got %f", prec_01);
  }
  if (prec_02 <= 0.0 || !R_FINITE(prec_02)) {
    error("prec_02 must be positive and finite, got %f", prec_02);
  }
  if (nu_01 <= 0.0 || !R_FINITE(nu_01)) {
    error("nu_01 must be positive and finite, got %f", nu_01);
  }
  if (eta_01 <= 0.0 || !R_FINITE(eta_01)) {
    error("eta_01 must be positive and finite, got %f", eta_01);
  }
  if (nu_02 <= 0.0 || !R_FINITE(nu_02)) {
    error("nu_02 must be positive and finite, got %f", nu_02);
  }
  if (eta_02 <= 0.0 || !R_FINITE(eta_02)) {
    error("eta_02 must be positive and finite, got %f", eta_02);
  }

  /* ========== Compute Sufficient Statistics ========== */
  /* First pass: compute counts (T_k) and sums (s_k) for each component.
   * T_0 = number of observations assigned to component 1 (z_t = 0)
   * T_1 = number of observations assigned to component 2 (z_t = 1)
   * s_0 = sum of y_t for observations in component 1
   * s_1 = sum of y_t for observations in component 2
   *
   * Using double precision for counts ensures numerical consistency with
   * subsequent variance calculations, especially when T_k is large. */
  for (t = 0; t < n; t++) {
    if (z[t] == 1.0) {
      T_1 += 1.0;
      s_1 += y[t];
    } else if (z[t] == 0.0) {
      T_0 += 1.0;
      s_0 += y[t];
    } else {
      error("z[%d] must be 0.0 or 1.0, got %f", t, z[t]);
    }
  }

  /* ========== Sample Mean of Component 1 (Lower) ========== */
  /* Full conditional posterior:
   * mu_1 | y, [...] ~ N(mu_bar_1, sigma_bar^2_1)
   * where:
   * sigma_bar^2_1 = (T_0 * phi_1 + prec_01)^{-1}
   * mu_bar_1 = sigma_bar^2_1 * (s_0 * phi_1 + mu_01 * prec_01)
   *
   * This is the standard conjugate Normal-Normal update where the posterior
   * precision is the sum of prior precision and data precision (T_0 * phi_1). */
  double var_mu_1 = 1.0 / (prec_01 + T_0 * prec_1_prev);
  double mean_mu_1 = (mu_01 * prec_01 + s_0 * prec_1_prev) * var_mu_1;
  double mu_1_curr = rnorm(mean_mu_1, sqrt(var_mu_1));

  /* ========== Sample Mean of Component 2 (Upper) ========== */
  /* Full conditional posterior:
   * mu_2 | y, [...] ~ N(mu_bar_2, sigma_bar^2_2)
   * where:
   * sigma_bar^2_2 = (T_1 * phi_2 + prec_02)^{-1}
   * mu_bar_2 = sigma_bar^2_2 * (s_1 * phi_2 + mu_02 * prec_02) */
  double var_mu_2 = 1.0 / (prec_02 + T_1 * prec_2_prev);
  double mean_mu_2 = (mu_02 * prec_02 + s_1 * prec_2_prev) * var_mu_2;
  double mu_2_curr = rnorm(mean_mu_2, sqrt(var_mu_2));

  /* ========== Compute Sum of Squared Deviations ========== */
  /* Second pass: compute v_k = Sum_{t: z_t = k-1} (y_t - mu_k)^2 using newly
   * sampled means. This ensures consistency with the full conditional posteriors
   * for the precision parameters.
   *
   * Note: We use the newly sampled means (mu_1_curr, mu_2_curr) rather than
   * the previous iteration values, as required by the Gibbs sampling algorithm. */
  v_0 = 0.0;
  v_1 = 0.0;

  for (t = 0; t < n; t++) {
    if (z[t] == 1.0) {
      diff = y[t] - mu_2_curr;
      v_1 += diff * diff;
    } else {
      diff = y[t] - mu_1_curr;
      v_0 += diff * diff;
    }
  }

  /* ========== Sample Precision of Component 1 (Lower) ========== */
  /* Full conditional posterior:
   * phi_1 | y, [...] ~ Gamma(nu_bar_1, eta_bar_1)
   * where:
   * nu_bar_1 = nu_01 + T_0 / 2
   * eta_bar_1 = eta_01 + v_0 / 2
   *
   * This is the standard conjugate Gamma-Normal update where the posterior
   * shape increases by the number of observations T_0/2 and the rate increases
   * by half the sum of squared deviations v_0/2. */
  double nu_bar_1 = nu_01 + T_0 / 2.0;
  double eta_bar_1 = eta_01 + v_0 / 2.0;
  double prec_1_curr = rgamma(nu_bar_1, 1.0 / eta_bar_1);

  /* ========== Sample Precision of Component 2 (Upper) ========== */
  /* Full conditional posterior:
   * phi_2 | y, [...] ~ Gamma(nu_bar_2, eta_bar_2)
   * where:
   * nu_bar_2 = nu_02 + T_1 / 2
   * eta_bar_2 = eta_02 + v_1 / 2 */
  double nu_bar_2 = nu_02 + T_1 / 2.0;
  double eta_bar_2 = eta_02 + v_1 / 2.0;
  double prec_2_curr = rgamma(nu_bar_2, 1.0 / eta_bar_2);

  /* ========== Enforce Label Switching Constraint ========== */
  /* To ensure identifiability, enforce mu_1 < mu_2 (ordering constraint).
   * If the constraint is violated, swap the components.
   * This maintains consistency with the assumption that component 1 has the
   * smaller mean (lower component) and component 2 has the larger mean
   * (upper component).
   *
   * Label switching is a well-known issue in Bayesian mixture models where
   * the posterior is invariant to permutations of component labels. Enforcing
   * an ordering constraint on the means is a simple and effective solution. */
  if (mu_1_curr > mu_2_curr) {
    /* Swap means */
    double temp_mu = mu_1_curr;
    mu_1_curr = mu_2_curr;
    mu_2_curr = temp_mu;

    /* Swap precisions */
    double temp_prec = prec_1_curr;
    prec_1_curr = prec_2_curr;
    prec_2_curr = temp_prec;
  }

  /* ========== Store Results in Output Vector ========== */
  /* Parameter vector structure for k=2 components:
   * params_current[0] = mu_1    (mean of lower component)
   * params_current[1] = prec_1  (precision of lower component)
   * params_current[2] = mu_2    (mean of upper component)
   * params_current[3] = prec_2  (precision of upper component) */
  params_current[0] = mu_1_curr;
  params_current[1] = prec_1_curr;
  params_current[2] = mu_2_curr;
  params_current[3] = prec_2_curr;
}
