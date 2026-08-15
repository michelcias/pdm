/**
 * @file conditional_mixture_poisson_indicators.c
 * @brief Gibbs sampling for latent indicators in Poisson mixture models
 * @author Michel H. Montoril
 * @date 2026-08-15
 * @version 1.0
 *
 * @details Implements the full-conditional sampler for the latent component
 *          indicators of a two-component Poisson mixture with dynamic mixture
 *          weights.
 *
 *          **Model specification:**
 *          y_t | z_t, lambda ~ Poisson(z_t * lambda_2 + (1 - z_t) * lambda_1)
 *          z_t | alpha_t     ~ Bernoulli(alpha_t)
 *
 *          **Full conditional posterior:**
 *          By Bayes' theorem, P(z_t = k | y_t, ...) is proportional to
 *          P(y_t | z_t = k, lambda) P(z_t = k | alpha_t), so
 *
 *          P(z_t = 1 | y_t, alpha_t, lambda) =
 *              alpha_t f_2(y_t) /
 *              [(1 - alpha_t) f_1(y_t) + alpha_t f_2(y_t)]
 *
 *          with f_k(y_t) = Poisson(y_t | lambda_k).
 *
 *          **Numerical stability:** the ratio is evaluated through a
 *          log-sum-exp rather than by dividing two densities, because Poisson
 *          densities underflow at ordinary count magnitudes -- see the header
 *          for the worked figure. Subtracting the larger of the two log weights
 *          before exponentiating leaves one term at exactly 1 and the other in
 *          (0, 1], so the denominator can never be zero.
 *
 *          **Integration with the dynamic model:** this is step 2 of the Gibbs
 *          cycle in the mcmc_poisson_mixture_* drivers, sitting between the
 *          component-rate update and the dynamic-weight update.
 */

#include <R.h>
#include <Rmath.h>
#include "conditional_mixture_poisson_indicators.h"

/**
 * @brief Draw the latent indicators of a two-component Poisson mixture
 *
 * @details For each observation t:
 *
 *          1. Form log w_1 = log(1 - alpha_t) + log f_1(y_t) and
 *             log w_2 = log(alpha_t) + log f_2(y_t).
 *          2. Normalise with a log-sum-exp to get P(z_t = 1 | ...).
 *          3. Draw z_t from the resulting Bernoulli.
 *
 *          The two boundary values of alpha_t short-circuit to a deterministic
 *          assignment, which also keeps log(0) out of step 1.
 *
 * @param y        Observed counts [n] (const, read-only).
 * @param z        Output latent indicators [n], set to 0.0 or 1.0.
 * @param params   Current [2]: [lambda_1, lambda_2] (const, read-only).
 * @param alpha    Current dynamic weights [n] (const, read-only).
 * @param n        Sample size.
 * @return None (results are written to z).
 *
 * @note Requires GetRNGstate()/PutRNGstate() bracket in the calling function.
 * @note Complexity: O(n) time, O(1) additional space.
 * @note The indicators are conditionally independent, so the sequential order
 *       of the sweep is immaterial.
 *
 * @warning params[0] and params[1] must be positive and finite.
 * @warning Each alpha[t] must lie in [0, 1].
 * @warning n must be > 0.
 *
 * @see conditional_mixture_poisson_parameters_k2
 */
void conditional_mixture_poisson_indicators_k2(const double *y,
                                               double       *z,
                                               const double *params,
                                               const double *alpha,
                                               int           n) {

  int t;

  /* Extract parameters for readability */
  double lambda_1 = params[0];
  double lambda_2 = params[1];

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
  if (lambda_1 <= 0.0 || !R_FINITE(lambda_1)) {
    error("params[0] (lambda_1) must be positive and finite, got %f", lambda_1);
  }
  if (lambda_2 <= 0.0 || !R_FINITE(lambda_2)) {
    error("params[1] (lambda_2) must be positive and finite, got %f", lambda_2);
  }

  /* ========== Sample Latent Indicators ========== */
  for (t = 0; t < n; t++) {
    if (alpha[t] < 0.0 || alpha[t] > 1.0 || !R_FINITE(alpha[t])) {
      error("alpha[%d] must be in [0, 1], got %f", t, alpha[t]);
    }

    /* ---------- Fast Path: Deterministic Cases ---------- */
    /* At either boundary the assignment carries no randomness, and taking it
     * here also keeps log(0) out of the general branch below. */
    if (alpha[t] == 0.0) {
      z[t] = 0.0;
      continue;
    }

    if (alpha[t] == 1.0) {
      z[t] = 1.0;
      continue;
    }

    /* ---------- General Case: Log-Sum-Exp Normalisation ---------- */
    double log_w_1 = log1p(-alpha[t]) + dpois(y[t], lambda_1, 1);
    double log_w_2 = log(alpha[t])    + dpois(y[t], lambda_2, 1);

    /* Shift by the larger term: one exponential comes back as exactly 1 and the
     * other lands in (0, 1], so the sum is bounded away from zero however small
     * the two densities are on the original scale. */
    double log_max = (log_w_1 > log_w_2) ? log_w_1 : log_w_2;
    double w_1     = exp(log_w_1 - log_max);
    double w_2     = exp(log_w_2 - log_max);

    double prob_component_2 = w_2 / (w_1 + w_2);

    z[t] = (unif_rand() < prob_component_2) ? 1.0 : 0.0;
  }
}
