/**
 * @file conditional_mixture_poisson_parameters.c
 * @brief Gibbs sampling for Poisson mixture model component rates
 * @author Michel H. Montoril
 * @date 2026-08-15
 * @version 1.0
 *
 * @details Implements the full-conditional sampler for the component rates of a
 *          two-component Poisson mixture with dynamic mixture weights.
 *
 *          Current implementation:
 *          - Two-component Poisson mixture (k = 2)
 *
 *          **Model specification:**
 *          y_t | z_t, lambda ~ Poisson(z_t * lambda_2 + (1 - z_t) * lambda_1)
 *          z_t | alpha_t     ~ Bernoulli(alpha_t)
 *
 *          **Prior distributions:**
 *          lambda_k ~ Gamma(a_0k, b_0k), shape-rate.
 *
 *          **Full conditional posteriors:**
 *          lambda_k | y, z ~ Gamma(a_0k + s_k, b_0k + T_k)
 *
 *          with T_k = #{t : z_t = k - 1} and s_k = Sum_{t: z_t = k-1} y_t. The
 *          Gamma-Poisson conjugacy makes both draws exact, so this step carries
 *          no tuning of any kind.
 *
 *          **Empty components:** when T_k = 0 the posterior collapses to the
 *          prior, Gamma(a_0k, b_0k), which is finite for any proper prior. This
 *          is the reason the Poisson mixture needs no analogue of the Gaussian
 *          degeneracy guard: a Poisson component covering no observation is
 *          harmless, whereas a Gaussian one collapsing onto a single point
 *          drives its precision up without bound.
 *
 *          **Label switching:**
 *          The constraint lambda_1 < lambda_2 is enforced by swapping the two
 *          draws whenever it is violated, so component 1 is the low-rate
 *          component by construction.
 */

#include <R.h>
#include <Rmath.h>
#include "conditional_mixture_poisson_parameters.h"
#include "utils.h"          /* rgamma_positive */

/**
 * @brief Draw the two component rates of a Poisson mixture from their full
 *        conditionals
 *
 * @details One Gibbs sweep over (lambda_1, lambda_2):
 *
 *          1. Accumulate T_k and s_k in a single pass over the data.
 *          2. Draw lambda_1 ~ Gamma(a_01 + s_0, b_01 + T_0).
 *          3. Draw lambda_2 ~ Gamma(a_02 + s_1, b_02 + T_1).
 *          4. Swap the pair if lambda_1 > lambda_2.
 *
 *          `rgamma_positive()` is used rather than a bare `rgamma()` for the
 *          same reason as everywhere else in the package: a rate that underflows
 *          to zero would make the next indicator step evaluate a degenerate
 *          Poisson density.
 *
 *          The draws use the shape-scale form R's `rgamma()` takes, so each rate
 *          is passed as its reciprocal.
 *
 * @param y                Observed counts [n] (const, read-only).
 * @param z                Latent indicators [n] (const, read-only), 0.0 or 1.0.
 * @param params_current   Output [2]: [lambda_1, lambda_2].
 * @param shape_01         Prior shape a_01 > 0 for lambda_1.
 * @param rate_01          Prior rate b_01 > 0 for lambda_1.
 * @param shape_02         Prior shape a_02 > 0 for lambda_2.
 * @param rate_02          Prior rate b_02 > 0 for lambda_2.
 * @param n                Sample size.
 * @return None (results are written to params_current).
 *
 * @note Requires GetRNGstate()/PutRNGstate() bracket in the calling function.
 * @note Complexity: O(n) time, O(1) additional space.
 *
 * @warning Each z[t] must be exactly 0.0 or 1.0.
 * @warning Counts must be non-negative; negative y would give a negative
 *          posterior shape.
 *
 * @see conditional_mixture_poisson_indicators_k2
 */
void conditional_mixture_poisson_parameters_k2(const double *y,
                                               const double *z,
                                               double       *params_current,
                                               double        shape_01,
                                               double        rate_01,
                                               double        shape_02,
                                               double        rate_02,
                                               int           n) {

  int    t;
  double T_0 = 0.0, T_1 = 0.0;
  double s_0 = 0.0, s_1 = 0.0;

  /* ========== Input Validation ========== */
  if (y == NULL) {
    error("y cannot be NULL");
  }
  if (z == NULL) {
    error("z cannot be NULL");
  }
  if (params_current == NULL) {
    error("params_current cannot be NULL");
  }
  if (n <= 0) {
    error("n must be positive, got %d", n);
  }
  if (shape_01 <= 0.0 || !R_FINITE(shape_01)) {
    error("shape_01 must be positive and finite, got %f", shape_01);
  }
  if (rate_01 <= 0.0 || !R_FINITE(rate_01)) {
    error("rate_01 must be positive and finite, got %f", rate_01);
  }
  if (shape_02 <= 0.0 || !R_FINITE(shape_02)) {
    error("shape_02 must be positive and finite, got %f", shape_02);
  }
  if (rate_02 <= 0.0 || !R_FINITE(rate_02)) {
    error("rate_02 must be positive and finite, got %f", rate_02);
  }

  /* ========== Compute Sufficient Statistics ========== */
  /* Single pass: T_k counts the observations assigned to component k and s_k
   * sums them. The pair (T_k, s_k) is sufficient for lambda_k, which is the
   * whole of the Gamma-Poisson update. */
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

  /* ========== Sample the Two Component Rates ========== */
  /* lambda_k | y, z ~ Gamma(a_0k + s_k, rate = b_0k + T_k), drawn in the
   * shape-scale form R's rgamma() takes. */
  double lambda_1_curr = rgamma_positive(shape_01 + s_0, 1.0 / (rate_01 + T_0));
  double lambda_2_curr = rgamma_positive(shape_02 + s_1, 1.0 / (rate_02 + T_1));

  /* ========== Enforce Label Switching Constraint ========== */
  /* The likelihood is invariant to permuting the component labels, so the
   * ordering lambda_1 < lambda_2 is what identifies them. Component 1 is the
   * low-rate component after this swap, by construction. */
  if (lambda_1_curr > lambda_2_curr) {
    double temp   = lambda_1_curr;
    lambda_1_curr = lambda_2_curr;
    lambda_2_curr = temp;
  }

  /* ========== Store Results in Output Vector ========== */
  params_current[0] = lambda_1_curr;
  params_current[1] = lambda_2_curr;
}
