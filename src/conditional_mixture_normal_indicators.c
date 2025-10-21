/**
 * @file conditional_mixture_normal_indicators.c
 * @brief Gibbs sampling for latent indicators in Gaussian mixture models
 * @author Michel H. Montoril
 * @date 2025-10-14
 * @version 1.0
 *
 * @details This file implements Gibbs sampling for latent indicator variables in
 *          Gaussian mixture models with dynamic mixture weights.
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
 *          where z_t in {0, 1} are latent indicators (0 = component 1, 1 = component 2),
 *          mu = (mu_1, mu_2)' are component means, phi = (phi_1, phi_2)' are component
 *          precisions, and alpha_t are dynamic mixture weights.
 *
 *          Full conditional posterior for indicators:
 *          The conditional posterior distribution for each z_t is:
 *
 *          P(z_t = 1 | y_t, alpha_t, mu, phi) = alpha_star_t
 *
 *          where:
 *          alpha_star_t = (alpha_t * f_2(y_t)) / ((1 - alpha_t) * f_1(y_t) + alpha_t * f_2(y_t))
 *
 *          with f_k(y_t) = N(y_t | mu_k, phi_k^{-1}) being the density of component k.
 *
 *          This is derived using Bayes' theorem:
 *          P(z_t = 1 | y_t, ...) proportional to P(y_t | z_t = 1, ...) * P(z_t = 1 | alpha_t)
 *                                            = N(y_t | mu_2, phi_2^{-1}) * alpha_t
 *
 *          Algorithm:
 *          For each observation t:
 *          1. Compute weighted densities for both components
 *          2. Calculate probability of belonging to component 2
 *          3. Sample z_t from Bernoulli(alpha_star_t)
 *
 *          Integration with dynamic model:
 *          This function is part of the Gibbs sampling cycle for mixture models with
 *          dynamic weights. The sampling order is typically:
 *          1. Sample component parameters (mu, phi) given z and alpha
 *          2. Sample indicators z given y, (mu, phi), and alpha
 *          3. Sample dynamic weights alpha given z using CWMH or probit methods
 *
 *          Numerical stability:
 *          - Computes probabilities in original scale (not log-scale) for simplicity
 *          - Division by sum automatically normalizes weighted densities
 *          - Handles edge cases where alpha_t is near 0 or 1 gracefully
 */

#include <R.h>
#include <Rmath.h>
#include "conditional_mixture_normal_indicators.h"

/**
 * @brief Generate latent indicators for a two-component Gaussian mixture model
 *
 * @details Implements one Gibbs sampling iteration for the latent indicator variables
 *          z_t in a two-component Gaussian mixture model. For each observation, the
 *          function:
 *
 *          1. Computes the weighted density for component 1:
 *             w_1(y_t) = (1 - alpha_t) * N(y_t | mu_1, phi_1^{-1})
 *
 *          2. Computes the weighted density for component 2:
 *             w_2(y_t) = alpha_t * N(y_t | mu_2, phi_2^{-1})
 *
 *          3. Calculates the conditional probability:
 *             alpha_star_t = w_2(y_t) / (w_1(y_t) + w_2(y_t))
 *
 *          4. Samples from Bernoulli(alpha_star_t):
 *             z_t = 1 with probability alpha_star_t
 *             z_t = 0 with probability 1 - alpha_star_t
 *
 *          Mathematical derivation:
 *          Using Bayes' theorem, the conditional posterior is:
 *
 *          P(z_t = k | y_t, alpha_t, mu, phi) proportional to
 *              P(y_t | z_t = k, mu, phi) * P(z_t = k | alpha_t)
 *
 *          For k = 1: P(y_t | z_t = 0, ...) * P(z_t = 0 | alpha_t)
 *                   = N(y_t | mu_1, phi_1^{-1}) * (1 - alpha_t)
 *
 *          For k = 2: P(y_t | z_t = 1, ...) * P(z_t = 1 | alpha_t)
 *                   = N(y_t | mu_2, phi_2^{-1}) * alpha_t
 *
 *          Normalizing gives the probability formula above.
 *
 *          Numerical considerations:
 *          - Uses dnorm with log=0 for density evaluation
 *          - Standard deviation is sqrt(1/phi_k) = 1/sqrt(phi_k)
 *          - Division normalizes automatically, avoiding log-sum-exp
 *          - Works well when densities are well-separated or alpha is informative
 *
 *          Independence assumption:
 *          The z_t are conditionally independent given (y, alpha, mu, phi), so each
 *          indicator can be sampled independently in any order.
 *
 *          Memory efficiency:
 *          - In-place updates of z vector
 *          - No dynamic memory allocation
 *          - Single pass through data
 *
 * @param y                  Observed data vector [n] (const, read-only).
 *                           Contains the response values y_1, ..., y_n.
 * @param z                  Output latent indicator vector [n].
 *                           Each z[t] will be set to 0.0 (component 1) or 1.0 (component 2).
 *                           Must be pre-allocated with size n.
 * @param params             Current parameter vector [4] (const, read-only).
 *                           Structure: [mu_1, prec_1, mu_2, prec_2].
 *                           Contains component means and precisions from current iteration.
 * @param alpha              Current dynamic weight vector [n] (const, read-only).
 *                           Contains alpha_1, ..., alpha_n where alpha_t = P(z_t = 1).
 *                           Each alpha[t] must be in (0, 1).
 * @param n                  Sample size (length of y, z, and alpha vectors).
 *
 * @return None (results are written to z).
 *
 * @note Computational complexity: O(n) per iteration (single pass through data).
 * @note Memory requirements: O(1) additional space beyond input/output arrays.
 * @note Independence: z_t are conditionally independent, sampled in sequential order.
 * @note Numerical stability: Uses original scale densities with normalization.
 * @note Integration: Part of Gibbs cycle with mixture_normal and dynamic weight samplers.
 * @note Random number generation: Uses R's unif_rand() for Bernoulli sampling.
 *
 * @warning Each alpha[t] must be strictly in (0, 1) to avoid division by zero or degeneracy.
 * @warning params[1] and params[3] (precisions) must be positive for valid standard deviations.
 * @warning n must be > 0.
 * @warning z must be pre-allocated with size n.
 * @warning For numerical stability, ensure alpha values are not too close to 0 or 1
 *          (e.g., alpha in [0.01, 0.99]).
 *
 * @see conditional_mixture_normal_parameters_k2 (for sampling component parameters)
 * @see conditional_mixture_normal_indicators_k (future K-component generalization)
 *
 * @example
 * @code
 * // Setup for MCMC iteration
 * double y[n];           // observed data
 * double z[n];           // latent indicators (output)
 * double params[4] = {0.0, 4.0, 2.0, 4.0};  // [mu_1, prec_1, mu_2, prec_2]
 * double alpha[n];       // dynamic weights from previous step
 *
 * // Generate latent indicators for current iteration
 * conditional_mixture_normal_indicators_k2(
 *     y,                 // observed data
 *     z,                 // output: latent indicators
 *     params,            // current component parameters
 *     alpha,             // current dynamic weights
 *     n                  // sample size
 * );
 *
 * // z now contains 0.0 or 1.0 for each observation
 * // Use z to sample component parameters in next step
 * @endcode
 *
 * @version 1.0
 */
void conditional_mixture_normal_indicators_k2(const double *y,
                                              double       *z,
                                              const double *params,
                                              const double *alpha,
                                              int           n) {

  int t;

  /* Extract parameters for readability */
  double mu_1 = params[0];
  double prec_1 = params[1];
  double mu_2 = params[2];
  double prec_2 = params[3];

  /* Compute standard deviations from precisions */
  double sd_1 = 1.0 / sqrt(prec_1);
  double sd_2 = 1.0 / sqrt(prec_2);

  /* ========== Input Validation ========== */
  if (y == NULL) {
    error("y cannot be NULL");
  }
  if (z == NULL) {
    error("z cannot be NULL");
  }
  if (params == NULL) {
    error("params cannot be NULL");
  }
  if (alpha == NULL) {
    error("alpha cannot be NULL");
  }
  if (n <= 0) {
    error("n must be positive, got %d", n);
  }
  if (prec_1 <= 0.0 || !R_FINITE(prec_1)) {
    error("params[1] (prec_1) must be positive and finite, got %f", prec_1);
  }
  if (prec_2 <= 0.0 || !R_FINITE(prec_2)) {
    error("params[3] (prec_2) must be positive and finite, got %f", prec_2);
  }

  /* ========== Sample Latent Indicators ========== */
  /* For each observation, compute the conditional probability of belonging to
   * component 2 and sample from the resulting Bernoulli distribution.
   *
   * The algorithm computes:
   * P(z_t = 1 | y_t, alpha_t, mu, phi) =
   *     (alpha_t * f_2(y_t)) / ((1 - alpha_t) * f_1(y_t) + alpha_t * f_2(y_t))
   *
   * where f_k(y_t) = N(y_t | mu_k, phi_k^{-1}) is the density of component k
   * evaluated at y_t.
   *
   * Special cases for computational efficiency:
   * - If alpha_t = 0, then z_t = 0 deterministically (no sampling needed)
   * - If alpha_t = 1, then z_t = 1 deterministically (no sampling needed)
   * - If alpha_t in (0, 1), compute full conditional and sample */
  for (t = 0; t < n; t++) {
    /* Validate alpha[t] is in valid probability range */
    if (alpha[t] < 0.0 || alpha[t] > 1.0 || !R_FINITE(alpha[t])) {
      error("alpha[%d] must be in [0, 1], got %f", t, alpha[t]);
    }

    /* ========== Fast Path: Deterministic Cases ========== */
    /* When alpha_t is at boundary, assignment is deterministic.
     * This optimization avoids unnecessary density calculations. */
    if (alpha[t] == 0.0) {
      /* Certainty of component 1: P(z_t = 1 | alpha_t = 0) = 0 */
      z[t] = 0.0;
      continue;
    }

    if (alpha[t] == 1.0) {
      /* Certainty of component 2: P(z_t = 1 | alpha_t = 1) = 1 */
      z[t] = 1.0;
      continue;
    }

    /* ========== General Case: Stochastic Assignment ========== */
    /* For alpha_t in (0, 1), compute weighted densities and sample. */

    /* Compute weighted density for component 1 (z_t = 0):
     * w_1(y_t) = (1 - alpha_t) * N(y_t | mu_1, sd_1^2) */
    double dens_1 = dnorm(y[t], mu_1, sd_1, 0);
    double weighted_dens_1 = (1.0 - alpha[t]) * dens_1;

    /* Compute weighted density for component 2 (z_t = 1):
     * w_2(y_t) = alpha_t * N(y_t | mu_2, sd_2^2) */
    double dens_2 = dnorm(y[t], mu_2, sd_2, 0);
    double weighted_dens_2 = alpha[t] * dens_2;

    /* Compute conditional probability of component 2:
     * P(z_t = 1 | ...) = w_2(y_t) / (w_1(y_t) + w_2(y_t)) */
    double prob_component_2 = weighted_dens_2 / (weighted_dens_1 + weighted_dens_2);

    /* Sample from Bernoulli(prob_component_2):
     * z_t = 1 with probability prob_component_2
     * z_t = 0 with probability 1 - prob_component_2 */
    z[t] = (unif_rand() < prob_component_2) ? 1.0 : 0.0;
  }
}
