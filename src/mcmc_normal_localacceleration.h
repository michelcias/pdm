/**
 * @file mcmc_normal_localacceleration.h
 * @brief Header for MCMC sampling in Gaussian local-acceleration dynamic models
 * @author Michel H. Montoril
 * @date 2025-10-11
 * @version 1.1
 *
 * @details This header declares the Gibbs sampler for Bayesian estimation of local-acceleration
 *          polynomial dynamic models with Gaussian observation equations. The implementation
 *          provides efficient MCMC inference for time series models with level, trend, and
 *          acceleration dynamics.
 *
 *          **Key features:**
 *          - Complete Gibbs sampler for local-acceleration models
 *          - Memory-efficient O(n) temporary storage using current/previous buffers
 *          - Tridiagonal precision matrix methods for O(n) state sampling
 *          - Conjugate posterior updates for all parameters
 *          - Flexible burn-in and thinning controls
 *          - R interface with proper RNG state management
 *
 *          **Model specification:**
 *          Observation equation: y_t = theta_{t,1} + e_t,  e_t ~ N(0, V)
 *          State equations:
 *            theta_{t,1} = theta_{t-1,1} + theta_{t-1,2} + u_{t,1},  u_{t,1} ~ N(0, W_1)
 *            theta_{t,2} = theta_{t-1,2} + theta_{t-1,3} + u_{t,2},  u_{t,2} ~ N(0, W_2)
 *            theta_{t,3} = theta_{t-1,3} + u_{t,3},                  u_{t,3} ~ N(0, W_3)
 *
 *          **Prior distributions:**
 *          - theta_{0,1} ~ N(mu_{0,1}, sigma_{0,1}^2)
 *          - theta_{0,2} ~ N(mu_{0,2}, sigma_{0,2}^2)
 *          - theta_{0,3} ~ N(mu_{0,3}, sigma_{0,3}^2)
 *          - each precision 1/W_1, 1/W_2, 1/W_3, 1/V carries either a Gamma prior
 *            on the precision (default) or a Half-t prior on the corresponding
 *            standard deviation (Gelman, 2006), selected independently via the
 *            prior_prec*_type_ codes (see prec_prior_dispatch.h).
 *
 *          **Gibbs sampling sequence:**
 *          1. theta_3 | theta_2, theta_{0,2}, theta_{0,3}, W_2, W_3 → Multivariate Normal
 *          2. 1/W_3 | theta_3, theta_{0,3} → Gamma
 *          3. theta_{0,3} | theta_2, theta_3, theta_{0,2}, W_2, W_3 → Normal
 *          4. theta_2 | theta_1, theta_3, theta_{0,1}, theta_{0,2}, theta_{0,3}, W_1, W_2 → MVN
 *          5. 1/W_2 | theta_2, theta_3, theta_{0,2}, theta_{0,3} → Gamma
 *          6. theta_{0,2} | theta_1, theta_2, theta_{0,1}, theta_{0,3}, W_1, W_2 → Normal
 *          7. theta_1 | y, theta_2, theta_{0,1}, theta_{0,2}, W_1, V → Multivariate Normal
 *          8. 1/W_1 | theta_1, theta_2, theta_{0,1}, theta_{0,2} → Gamma
 *          9. theta_{0,1} | theta_1, theta_{0,2}, W_1 → Normal
 *          10. 1/V | y, theta_1 → Gamma
 *
 *          **Output:**
 *          Returns n_chain posterior samples for all parameters after burn-in and thinning.
 */

#ifndef MCMC_NORMAL_LOCALACCELERATION_H
#define MCMC_NORMAL_LOCALACCELERATION_H

#include <R.h>
#include <Rinternals.h>

/**
 * @brief Gibbs sampler for local-acceleration dynamic model with Gaussian observations
 *
 * @details Implements a complete Gibbs MCMC algorithm for the local-acceleration dynamic model:
 *
 *          **Observation equation:**
 *          y_t = theta_{t,1} + e_t,     e_t ~ N(0, V)
 *
 *          **State equations:**
 *          theta_{t,1} = theta_{t-1,1} + theta_{t-1,2} + u_{t,1},  u_{t,1} ~ N(0, W_1)
 *          theta_{t,2} = theta_{t-1,2} + theta_{t-1,3} + u_{t,2},  u_{t,2} ~ N(0, W_2)
 *          theta_{t,3} = theta_{t-1,3} + u_{t,3},                  u_{t,3} ~ N(0, W_3)
 *
 *          The algorithm employs the Markovian structure of polynomial dynamic models
 *          to enable efficient block sampling of state vectors through forward-backward
 *          recursions implemented via tridiagonal precision matrix methods.
 *
 *          **Sampling sequence per iteration:**
 *          1. theta_3 | theta_2, theta_{0,2}, theta_{0,3}, W_2, W_3 → MVN (tridiagonal)
 *          2. 1/W_3 | theta_3, theta_{0,3} → Gamma posterior
 *          3. theta_{0,3} | theta_2, theta_3, theta_{0,2}, W_2, W_3 → Normal posterior
 *          4. theta_2 | theta_1, theta_3, theta_{0,1}, theta_{0,2}, theta_{0,3}, W_1, W_2 → MVN
 *          5. 1/W_2 | theta_2, theta_3, theta_{0,2}, theta_{0,3} → Gamma posterior
 *          6. theta_{0,2} | theta_1, theta_2, theta_{0,1}, theta_{0,3}, W_1, W_2 → Normal
 *          7. theta_1 | y, theta_2, theta_{0,1}, theta_{0,2}, W_1, V → MVN (tridiagonal)
 *          8. 1/W_1 | theta_1, theta_2, theta_{0,1}, theta_{0,2} → Gamma posterior
 *          9. theta_{0,1} | theta_1, theta_{0,2}, W_1 → Normal posterior
 *          10. 1/V | y, theta_1 → Gamma posterior
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
 * @param prior_theta01_mean_   SEXP Double scalar, prior mean mu_{0,1} for initial level theta_{0,1}.
 *                              Typical value: 0 (vague prior).
 * @param prior_theta01_prec_   SEXP Double scalar, prior precision tau_{0,1} = 1/sigma_{0,1}^2
 *                              for initial level. Typical value: 0.001 (vague prior).
 * @param prior_theta02_mean_   SEXP Double scalar, prior mean mu_{0,2} for initial trend theta_{0,2}.
 *                              Typical value: 0 (vague prior).
 * @param prior_theta02_prec_   SEXP Double scalar, prior precision tau_{0,2} = 1/sigma_{0,2}^2
 *                              for initial trend. Typical value: 0.001 (vague prior).
 * @param prior_theta03_mean_   SEXP Double scalar, prior mean mu_{0,3} for initial acceleration theta_{0,3}.
 *                              Typical value: 0 (vague prior).
 * @param prior_theta03_prec_   SEXP Double scalar, prior precision tau_{0,3} = 1/sigma_{0,3}^2
 *                              for initial acceleration. Typical value: 0.001 (vague prior).
 * @param prior_prec1_shape_    SEXP Double scalar, shape parameter nu_1 for Gamma(nu_1, eta_1)
 *                              prior on 1/W_1. Typical value: 0.001 (vague prior).
 * @param prior_prec1_rate_     SEXP Double scalar, rate parameter eta_1 for Gamma(nu_1, eta_1)
 *                              prior on 1/W_1. Typical value: 0.001 (vague prior).
 * @param prior_prec2_shape_    SEXP Double scalar, shape parameter nu_2 for Gamma(nu_2, eta_2)
 *                              prior on 1/W_2. Typical value: 0.001 (vague prior).
 * @param prior_prec2_rate_     SEXP Double scalar, rate parameter eta_2 for Gamma(nu_2, eta_2)
 *                              prior on 1/W_2. Typical value: 0.001 (vague prior).
 * @param prior_prec3_shape_    SEXP Double scalar, shape parameter nu_3 for Gamma(nu_3, eta_3)
 *                              prior on 1/W_3. Typical value: 0.001 (vague prior).
 * @param prior_prec3_rate_     SEXP Double scalar, rate parameter eta_3 for Gamma(nu_3, eta_3)
 *                              prior on 1/W_3. Typical value: 0.001 (vague prior).
 * @param prior_prec_y_shape_   SEXP Double scalar, shape parameter nu_y for Gamma(nu_y, eta_y)
 *                              prior on 1/V. Typical value: 0.001 (vague prior).
 * @param prior_prec_y_rate_    SEXP Double scalar, rate parameter eta_y for Gamma(nu_y, eta_y)
 *                              prior on 1/V. Typical value: 0.001 (vague prior).
 * @param verbose_              Logical flag enabling progress bar display (0 = off, non-zero = on)
 * @param bar_width_            Integer controlling progress bar width (clamped to 10-120 characters)
 *
 * @return SEXP R list containing posterior samples with named components:
 *         - theta_1: Numeric matrix [n_chain × n] of level state trajectory samples
 *         - theta_2: Numeric matrix [n_chain × n] of trend state trajectory samples
 *         - theta_3: Numeric matrix [n_chain × n] of acceleration state trajectory samples
 *         - theta_01: Numeric vector [n_chain] of initial level state theta_{0,1} samples
 *         - theta_02: Numeric vector [n_chain] of initial trend state theta_{0,2} samples
 *         - theta_03: Numeric vector [n_chain] of initial acceleration state theta_{0,3} samples
 *         - prec_theta1: Numeric vector [n_chain] of level innovation precision 1/W_1 samples
 *         - prec_theta2: Numeric vector [n_chain] of trend innovation precision 1/W_2 samples
 *         - prec_theta3: Numeric vector [n_chain] of acceleration innovation precision 1/W_3 samples
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
 * @see generate_theta_p
 * @see generate_precision_theta_p
 * @see generate_theta_0p
 * @see generate_theta_k
 * @see generate_precision_theta_k
 * @see generate_theta_0k
 * @see generate_theta_1
 * @see generate_theta_01
 * @see generate_precision_data
 */
SEXP C_MCMC_normal_localacceleration(SEXP y_,
                                     SEXP burnin_,
                                     SEXP thinning_,
                                     SEXP n_chain_,
                                     SEXP prior_theta01_mean_,
                                     SEXP prior_theta01_prec_,
                                     SEXP prior_theta02_mean_,
                                     SEXP prior_theta02_prec_,
                                     SEXP prior_theta03_mean_,
                                     SEXP prior_theta03_prec_,
                                     SEXP prior_prec1_type_,
                                     SEXP prior_prec1_shape_,
                                     SEXP prior_prec1_rate_,
                                     SEXP prior_prec1_scale_,
                                     SEXP prior_prec1_df_,
                                     SEXP prior_prec2_type_,
                                     SEXP prior_prec2_shape_,
                                     SEXP prior_prec2_rate_,
                                     SEXP prior_prec2_scale_,
                                     SEXP prior_prec2_df_,
                                     SEXP prior_prec3_type_,
                                     SEXP prior_prec3_shape_,
                                     SEXP prior_prec3_rate_,
                                     SEXP prior_prec3_scale_,
                                     SEXP prior_prec3_df_,
                                     SEXP prior_prec_y_type_,
                                     SEXP prior_prec_y_shape_,
                                     SEXP prior_prec_y_rate_,
                                     SEXP prior_prec_y_scale_,
                                     SEXP prior_prec_y_df_,
                                     SEXP verbose_,
                                     SEXP bar_width_);

#endif /* MCMC_NORMAL_LOCALACCELERATION_H */
