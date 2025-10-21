/**
 * @file conditional_mixture_normal_indicators.h
 * @brief Gibbs sampling for latent indicators in Gaussian mixture models
 * @author Michel H. Montoril
 * @date 2025-10-14
 * @version 1.0
 *
 * @details This file declares the Gibbs sampling routine for latent indicator variables in
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
 *          Conditional posterior for indicators:
 *          The conditional posterior distribution for each z_t is obtained from Bayes' theorem,
 *          combining component densities with dynamic weights:
 *
 *          P(z_t = 1 | y_t, alpha_t, mu, phi) =
 *              (alpha_t * f_2(y_t)) / ((1 - alpha_t) * f_1(y_t) + alpha_t * f_2(y_t)),
 *
 *          with f_k(y_t) = N(y_t | mu_k, phi_k^{-1}). This probability is used to draw
 *          Bernoulli samples for each latent indicator.
 *
 *          Integration with dynamic model:
 *          This routine is part of the Gibbs sampling cycle for mixture models with
 *          time-varying weights. Typical iteration order:
 *          1. Sample component parameters (mu, phi) given z and alpha
 *          2. Sample indicators z given y, (mu, phi), and alpha
 *          3. Sample dynamic weights alpha given z using CWMH or probit methods
 *
 *          Numerical stability:
 *          - Probabilities computed on original scale with automatic normalization
 *          - Handles edge cases where alpha_t approaches 0 or 1 gracefully
 *          - Uses double precision arithmetic throughout
 */

#ifndef CONDITIONAL_MIXTURE_NORMAL_INDICATORS_H
#define CONDITIONAL_MIXTURE_NORMAL_INDICATORS_H

/**
 * @brief Generate latent indicators for a two-component Gaussian mixture model
 *
 * @details Executes one Gibbs sampling iteration for latent indicator variables
 *          z_t in a two-component Gaussian mixture. For each observation, the function:
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
                                              int           n);

#endif /* CONDITIONAL_MIXTURE_NORMAL_INDICATORS_H */
