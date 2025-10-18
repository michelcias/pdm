/**
 * @file mcmc_normal_locallevel.h
 * @brief Header for MCMC sampling in Gaussian local-level dynamic models
 * @author Michel H. Montoril
 * @date 2025-10-11
 * @version 1.1
 *
 * @details This header declares the Gibbs sampler for Bayesian estimation of local-level
 *          polynomial dynamic models with Gaussian observation equations. The implementation
 *          provides efficient MCMC inference for time series models with level-only dynamics.
 *
 *          **Key features:**
 *          - Complete Gibbs sampler for local-level models
 *          - Memory-efficient O(n) temporary storage using current/previous buffers
 *          - Tridiagonal precision matrix methods for O(n) state sampling
 *          - Conjugate posterior updates for all parameters
 *          - Flexible burn-in and thinning controls
 *          - R interface with proper RNG state management
 *
 *          **Model specification:**
 *          Observation equation: y_t = theta_{t,1} + e_t,  e_t ~ N(0, V)
 *          State equation: theta_{t,1} = theta_{t-1,1} + u_{t,1},  u_{t,1} ~ N(0, W_1)
 *
 *          **Prior distributions:**
 *          - theta_{0,1} ~ N(mu_0, sigma_0^2)
 *          - 1/W_1 ~ Gamma(nu_1, eta_1)
 *          - 1/V   ~ Gamma(nu_y, eta_y)
 *
 *          **Gibbs sampling sequence:**
 *          1. theta_1 | y, theta_{0,1}, W_1, V → Multivariate Normal
 *          2. 1/W_1 | theta_1, theta_{0,1} → Gamma
 *          3. theta_{0,1} | theta_1, W_1 → Normal
 *          4. 1/V | y, theta_1 → Gamma
 *
 *          **Output:**
 *          Returns n_chain posterior samples for all parameters after burn-in and thinning.
 */

#ifndef MCMC_NORMAL_LOCALLEVEL_H
#define MCMC_NORMAL_LOCALLEVEL_H

#include <R.h>
#include <Rinternals.h>

/**
 * @brief Gibbs sampler for local-level dynamic model with Gaussian observations
 *
 * @details Implements a complete Gibbs MCMC algorithm for the local-level dynamic model:
 *
 *          **Observation equation:**
 *          y_t = theta_{t,1} + e_t,     e_t ~ N(0, V)
 *
 *          **State equation:**
 *          theta_{t,1} = theta_{t-1,1} + u_{t,1},  u_{t,1} ~ N(0, W_1)
 *
 *          The algorithm employs the Markovian structure of polynomial dynamic models
 *          to enable efficient block sampling of the state vector through forward-backward
 *          recursions implemented via tridiagonal precision matrix methods.
 *
 *          **Sampling sequence per iteration:**
 *          1. theta_1 | y, theta_{0,1}, W_1, V → Multivariate Normal (tridiagonal precision)
 *          2. 1/W_1 | theta_1, theta_{0,1} → Gamma posterior
 *          3. theta_{0,1} | theta_1, W_1 → Normal posterior
 *          4. 1/V | y, theta_1 → Gamma posterior
 *
 *          **Memory efficiency:**
 *          Uses current/previous iteration buffers instead of full trajectory storage,
 *          requiring only O(n) temporary memory regardless of chain length.
 *
 *          **Iteration count:**
 *          Total iterations = burnin + (n_chain - 1) × thinning + 1
 *
 * @param y_                    SEXP Numeric vector of observed time series data [length n].
 *                              Contains observations y_1, ..., y_n.
 * @param burnin_               SEXP Integer scalar, number of burn-in iterations to discard
 *                              for chain convergence. Typical values: 1000-10000.
 * @param thinning_             SEXP Integer scalar, thinning interval to reduce autocorrelation
 *                              in retained samples. Typical values: 1-10.
 * @param n_chain_              SEXP Integer scalar, target number of retained posterior samples.
 *                              Final output will contain exactly n_chain samples.
 * @param prior_theta01_mean_   SEXP Double scalar, prior mean mu_0 for initial state theta_{0,1}.
 *                              Typical value: 0 (vague prior).
 * @param prior_theta01_prec_   SEXP Double scalar, prior precision tau_0 = 1/sigma_0^2 for
 *                              initial state. Typical value: 0.001 (vague prior).
 * @param prior_prec1_shape_    SEXP Double scalar, shape parameter nu_1 for Gamma(nu_1, eta_1)
 *                              prior on 1/W_1. Typical value: 0.001 (vague prior).
 * @param prior_prec1_rate_     SEXP Double scalar, rate parameter eta_1 for Gamma(nu_1, eta_1)
 *                              prior on 1/W_1. Typical value: 0.001 (vague prior).
 * @param prior_prec_y_shape_   SEXP Double scalar, shape parameter nu_y for Gamma(nu_y, eta_y)
 *                              prior on 1/V. Typical value: 0.001 (vague prior).
 * @param prior_prec_y_rate_    SEXP Double scalar, rate parameter eta_y for Gamma(nu_y, eta_y)
 *                              prior on 1/V. Typical value: 0.001 (vague prior).
 * @param verbose_              Logical flag enabling progress bar display (0 = off, non-zero = on)
 * @param bar_width_            Integer controlling progress bar width (clamped to 10-120 characters)
 *
 * @return SEXP R list containing posterior samples with named components:
 *         - theta_1: Numeric matrix [n_chain × n] of complete state trajectory samples
 *         - theta_01: Numeric vector [n_chain] of initial state theta_{0,1} samples
 *         - prec_1: Numeric vector [n_chain] of innovation precision 1/W_1 samples
 *         - prec_y: Numeric vector [n_chain] of observation precision 1/V samples
 *
 * @note Computational complexity: O(n_iter × n) for n_iter total iterations.
 * @note Memory requirements: O(n) temporary storage for efficient buffer management.
 * @note RNG management: Proper GetRNGstate()/PutRNGstate() bracket for R integration.
 * @note Initialization: Uses prior-based random initialization for all parameters.
 *
 * @warning Minimum sample size n >= 3 enforced for numerical stability of recursions.
 * @warning Integer overflow protection: n <= INT_MAX due to R's integer limitations.
 * @warning Memory allocation failures will terminate R session via R_Calloc errors.
 * @warning No input validation for prior hyperparameters; negative values may cause crashes.
 *
 * @see generate_theta_1_locallevel
 * @see generate_precision_theta_p
 * @see generate_theta_01_locallevel
 * @see generate_precision_data
 */
SEXP C_MCMC_normal_locallevel(SEXP y_,
                              SEXP burnin_,
                              SEXP thinning_,
                              SEXP n_chain_,
                              SEXP prior_theta01_mean_,
                              SEXP prior_theta01_prec_,
                              SEXP prior_prec1_shape_,
                              SEXP prior_prec1_rate_,
                              SEXP prior_prec_y_shape_,
                              SEXP prior_prec_y_rate_,
                              SEXP verbose_,
                              SEXP bar_width_);

#endif /* MCMC_NORMAL_LOCALLEVEL_H */
