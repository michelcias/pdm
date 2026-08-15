/**
 * @file conditional_mixture_poisson_parameters.h
 * @brief Gibbs sampling for Poisson mixture model component rates
 * @author Michel H. Montoril
 * @date 2026-08-15
 * @version 1.0
 *
 * @details Declares the full-conditional sampler for the component rates of a
 *          two-component Poisson mixture with dynamic mixture weights.
 *
 *          **Model specification:**
 *          y_t | z_t, lambda ~ Poisson(z_t * lambda_2 + (1 - z_t) * lambda_1)
 *          z_t | alpha_t     ~ Bernoulli(alpha_t)
 *
 *          where lambda = (lambda_1, lambda_2)' are the component rates and
 *          alpha_t are dynamic mixture weights estimated separately.
 *
 *          **Prior distributions:**
 *          lambda_k ~ Gamma(a_0k, b_0k), shape-rate parameterization.
 *
 *          The Gamma is conjugate for the Poisson rate, so the component step
 *          is a pure Gibbs update with no Metropolis-Hastings and no auxiliary
 *          variable. This is the point where the Poisson mixture is *simpler*
 *          than its Gaussian sibling: there is no mean/precision pair to draw
 *          in sequence, and therefore no Half-t option to dispatch on.
 *
 *          **Full conditional posterior:**
 *          lambda_k | y, z ~ Gamma(a_0k + s_k, b_0k + T_k)
 *
 *          with T_k = #{t : z_t = k - 1} and s_k = Sum_{t: z_t = k-1} y_t.
 *
 *          **Label switching:**
 *          The components are identified only up to their labelling, so the
 *          constraint lambda_1 < lambda_2 is enforced by swapping after each
 *          draw -- the rate analogue of the mu_1 < mu_2 constraint used by
 *          conditional_mixture_normal_parameters_k2().
 */

#ifndef CONDITIONAL_MIXTURE_POISSON_PARAMETERS_H
#define CONDITIONAL_MIXTURE_POISSON_PARAMETERS_H

/**
 * @brief Draw the two component rates of a Poisson mixture from their full
 *        conditionals
 *
 * @details Computes the per-component sufficient statistics in a single pass
 *          over the data, draws each rate from its conjugate Gamma posterior,
 *          and enforces the ordering constraint lambda_1 < lambda_2.
 *
 * @param y                Observed counts [n] (const, read-only). Non-negative
 *                         integers stored as doubles.
 * @param z                Latent indicators [n] (const, read-only). Each entry
 *                         is exactly 0.0 (component 1) or 1.0 (component 2).
 * @param params_current   Output parameter vector [2]: [lambda_1, lambda_2].
 *                         Must be pre-allocated.
 * @param shape_01         Prior shape a_01 > 0 for lambda_1.
 * @param rate_01          Prior rate b_01 > 0 for lambda_1.
 * @param shape_02         Prior shape a_02 > 0 for lambda_2.
 * @param rate_02          Prior rate b_02 > 0 for lambda_2.
 * @param n                Sample size (length of y and z).
 * @return None (results are written to params_current).
 *
 * @note Requires GetRNGstate()/PutRNGstate() bracket in the calling function.
 * @note Complexity: O(n) time, O(1) additional space.
 * @note Unlike the Gaussian sampler, this one never reads the previous draw:
 *       the conjugate posterior depends on the data and z alone.
 *
 * @warning Each z[t] must be exactly 0.0 or 1.0; anything else is an error.
 * @warning Every prior hyperparameter must be positive and finite; they are
 *          validated in R and re-checked here.
 * @warning n must be > 0.
 *
 * @see conditional_mixture_poisson_indicators_k2
 * @see conditional_mixture_normal_parameters_k2
 */
void conditional_mixture_poisson_parameters_k2(const double *y,
                                               const double *z,
                                               double       *params_current,
                                               double        shape_01,
                                               double        rate_01,
                                               double        shape_02,
                                               double        rate_02,
                                               int           n);

#endif /* CONDITIONAL_MIXTURE_POISSON_PARAMETERS_H */
