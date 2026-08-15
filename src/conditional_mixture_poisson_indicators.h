/**
 * @file conditional_mixture_poisson_indicators.h
 * @brief Gibbs sampling for latent indicators in Poisson mixture models
 * @author Michel H. Montoril
 * @date 2026-08-15
 * @version 1.0
 *
 * @details Declares the full-conditional sampler for the latent component
 *          indicators of a two-component Poisson mixture with dynamic mixture
 *          weights.
 *
 *          **Model specification:**
 *          y_t | z_t, lambda ~ Poisson(z_t * lambda_2 + (1 - z_t) * lambda_1)
 *          z_t | alpha_t     ~ Bernoulli(alpha_t)
 *
 *          **Full conditional posterior:**
 *          P(z_t = 1 | y_t, alpha_t, lambda) = alpha*_t, where
 *
 *          alpha*_t = alpha_t f_2(y_t) /
 *                     [(1 - alpha_t) f_1(y_t) + alpha_t f_2(y_t)]
 *
 *          with f_k(y_t) = Poisson(y_t | lambda_k).
 *
 *          **Why this one works on the log scale:** the Gaussian sibling
 *          normalises two densities in the original scale, which is safe because
 *          a Normal density underflows only in the far tail. A Poisson density
 *          does not have that luxury -- @f$P(Y = 100 \mid \lambda = 1)@f$ is
 *          about @f$10^{-158}@f$, and a count series with a wide rate
 *          separation lands both weighted densities under `DBL_MIN` at once,
 *          turning the normalising division into 0/0. The log-sum-exp form here
 *          returns the same probability with no representable-range hazard.
 */

#ifndef CONDITIONAL_MIXTURE_POISSON_INDICATORS_H
#define CONDITIONAL_MIXTURE_POISSON_INDICATORS_H

/**
 * @brief Draw the latent indicators of a two-component Poisson mixture
 *
 * @details For each observation, forms the log weighted density of both
 *          components, normalises them with a log-sum-exp, and draws
 *          z_t ~ Bernoulli of the resulting probability. The indicators are
 *          conditionally independent given (y, alpha, lambda), so a single
 *          sequential pass is a valid Gibbs sweep.
 *
 * @param y        Observed counts [n] (const, read-only).
 * @param z        Output latent indicators [n]; each entry set to 0.0
 *                 (component 1) or 1.0 (component 2). Pre-allocated.
 * @param params   Current parameter vector [2] (const, read-only):
 *                 [lambda_1, lambda_2].
 * @param alpha    Current dynamic weights [n] (const, read-only), each in
 *                 [0, 1].
 * @param n        Sample size (length of y, z and alpha).
 * @return None (results are written to z).
 *
 * @note Requires GetRNGstate()/PutRNGstate() bracket in the calling function.
 * @note Complexity: O(n) time, O(1) additional space.
 *
 * @warning Both rates must be positive and finite.
 * @warning Each alpha[t] must lie in [0, 1]; the two boundary values are
 *          handled as deterministic assignments.
 * @warning n must be > 0.
 *
 * @see conditional_mixture_poisson_parameters_k2
 * @see conditional_mixture_normal_indicators_k2
 */
void conditional_mixture_poisson_indicators_k2(const double *y,
                                               double       *z,
                                               const double *params,
                                               const double *alpha,
                                               int           n);

#endif /* CONDITIONAL_MIXTURE_POISSON_INDICATORS_H */
