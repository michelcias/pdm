/**
 * @file mcmc_normal_localtrend.h
 * @brief Header for MCMC sampling in Gaussian local-trend dynamic models
 * @author Michel H. Montoril
 * @date 2026-07-25
 * @version 1.3
 *
 * @details This header declares the Gibbs sampler for Bayesian estimation of local-trend
 *          polynomial dynamic models with Gaussian observation equations. The implementation
 *          provides efficient MCMC inference for time series models with level and trend
 *          dynamics.
 *
 *          **Key features:**
 *          - Complete Gibbs sampler for local-trend models
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
 *            theta_{t,2} = theta_{t-1,2} + u_{t,2},                  u_{t,2} ~ N(0, W_2)
 *
 *          **Prior distributions:**
 *          - theta_{0,1} ~ N(mu_{0,1}, sigma_{0,1}^2)
 *          - theta_{0,2} ~ N(mu_{0,2}, sigma_{0,2}^2)
 *          - each precision 1/W_1, 1/W_2, 1/V carries either a Gamma prior on
 *            the precision or a Half-t prior on the corresponding standard
 *            deviation, selected independently via the prior_prec*_type_ codes
 *            (see prec_prior_dispatch.h).
 *
 *          **Gibbs sampling sequence:**
 *          1. theta_2 | theta_1, theta_{0,1}, theta_{0,2}, W_1, W_2 → Multivariate Normal
 *          2. 1/W_2 | theta_2, theta_{0,2} → Gamma
 *          3. theta_{0,2} | theta_1, theta_2, theta_{0,1}, W_1, W_2 → Normal
 *          4. theta_1 | y, theta_2, theta_{0,1}, theta_{0,2}, W_1, V → Multivariate Normal
 *          5. 1/W_1 | theta_1, theta_2, theta_{0,1}, theta_{0,2} → Gamma
 *          6. theta_{0,1} | theta_1, theta_{0,2}, W_1 → Normal
 *          7. 1/V | y, theta_1 → Gamma
 *
 *          **Output:**
 *          Returns n_draws posterior samples for all parameters after burn-in and thinning.
 */

#ifndef MCMC_NORMAL_LOCALTREND_H
#define MCMC_NORMAL_LOCALTREND_H

#include <R.h>
#include <Rinternals.h>

/**
 * @brief Gibbs sampler for local-trend dynamic model with Gaussian observations
 *
 * @details Implements a complete Gibbs MCMC algorithm for the local-trend dynamic model:
 *
 *          **Observation equation:**
 *          y_t = theta_{t,1} + e_t,     e_t ~ N(0, V)
 *
 *          **State equations:**
 *          theta_{t,1} = theta_{t-1,1} + theta_{t-1,2} + u_{t,1},  u_{t,1} ~ N(0, W_1)
 *          theta_{t,2} = theta_{t-1,2} + u_{t,2},                  u_{t,2} ~ N(0, W_2)
 *
 *          The algorithm employs the Markovian structure of polynomial dynamic models
 *          to enable efficient block sampling of state vectors through forward-backward
 *          recursions implemented via tridiagonal precision matrix methods.
 *
 *          **Sampling sequence per iteration:**
 *          1. theta_2 | theta_1, theta_{0,1}, theta_{0,2}, W_1, W_2 → MVN (tridiagonal)
 *          2. 1/W_2 | theta_2, theta_{0,2} → Gamma posterior
 *          3. theta_{0,2} | theta_1, theta_2, theta_{0,1}, W_1, W_2 → Normal posterior
 *          4. theta_1 | y, theta_2, theta_{0,1}, theta_{0,2}, W_1, V → MVN (tridiagonal)
 *          5. 1/W_1 | theta_1, theta_2, theta_{0,1}, theta_{0,2} → Gamma posterior
 *          6. theta_{0,1} | theta_1, theta_{0,2}, W_1 → Normal posterior
 *          7. 1/V | y, theta_1 → Gamma posterior
 *
 *          **Memory efficiency:**
 *          Uses current/previous iteration buffers instead of full trajectory storage,
 *          requiring only O(n) temporary memory regardless of chain length.
 *
 *          **Iteration count:**
 *          Total iterations = burnin + (n_draws - 1) × thinning + 1
 *
 * @param y_                    SEXP Numeric vector of observed time series data [length n].
 *                              Contains observations y_1, ..., y_n.
 * @param burnin_               SEXP Integer scalar, number of burn-in iterations to discard
 *                              for chain convergence. Typical values: 1000-10000.
 * @param thinning_             SEXP Integer scalar, thinning interval to reduce autocorrelation
 *                              in retained samples. Typical values: 1-10.
 * @param n_draws_              SEXP Integer scalar, target number of retained posterior samples.
 *                              Final output will contain exactly n_draws samples.
 * @param prior_theta01_mean_   SEXP Double scalar, prior mean mu_{0,1} for initial level theta_{0,1}.
 *                              Typical value: 0 (vague prior).
 * @param prior_theta01_prec_   SEXP Double scalar, prior precision tau_{0,1} = 1/sigma_{0,1}^2
 *                              for initial level. Typical value: 0.001 (vague prior).
 * @param prior_theta02_mean_   SEXP Double scalar, prior mean mu_{0,2} for initial trend theta_{0,2}.
 *                              Typical value: 0 (vague prior).
 * @param prior_theta02_prec_   SEXP Double scalar, prior precision tau_{0,2} = 1/sigma_{0,2}^2
 *                              for initial trend. Typical value: 0.001 (vague prior).
 * @param prior_prec1_type_     SEXP Integer scalar, prior kind on 1/W_1
 *                              (0 = Gamma on the precision, 1 = Half-t on sqrt(W_1)).
 * @param prior_prec1_shape_    SEXP Double scalar, Gamma shape nu_1 (Gamma kind).
 * @param prior_prec1_rate_     SEXP Double scalar, Gamma rate eta_1 (Gamma kind).
 * @param prior_prec1_scale_    SEXP Double scalar, Half-t scale A_1 > 0 (Half-t kind).
 * @param prior_prec1_df_       SEXP Double scalar, Half-t df nu_1 > 0 (Half-t kind; 1 = Half-Cauchy).
 * @param prior_prec2_type_     SEXP Integer scalar, prior kind on 1/W_2 (0 = Gamma, 1 = Half-t).
 * @param prior_prec2_shape_    SEXP Double scalar, Gamma shape nu_2 (Gamma kind).
 * @param prior_prec2_rate_     SEXP Double scalar, Gamma rate eta_2 (Gamma kind).
 * @param prior_prec2_scale_    SEXP Double scalar, Half-t scale A_2 > 0 (Half-t kind).
 * @param prior_prec2_df_       SEXP Double scalar, Half-t df nu_2 > 0 (Half-t kind; 1 = Half-Cauchy).
 * @param prior_prec_y_type_    SEXP Integer scalar, prior kind on 1/V (0 = Gamma, 1 = Half-t).
 * @param prior_prec_y_shape_   SEXP Double scalar, Gamma shape nu_y (Gamma kind).
 * @param prior_prec_y_rate_    SEXP Double scalar, Gamma rate eta_y (Gamma kind).
 * @param prior_prec_y_scale_   SEXP Double scalar, Half-t scale A_V > 0 (Half-t kind).
 * @param prior_prec_y_df_      SEXP Double scalar, Half-t df nu_y > 0 (Half-t kind; 1 = Half-Cauchy).
 * @param init_                 SEXP Double vector [8] of starting values, resolved in R by
 *                              resolve_init(): theta_{0,1} and theta_{0,2}, then 1/W_1,
 *                              1/W_2 and 1/V, then the Half-t auxiliary of each precision
 *                              in the same order (0 under a Gamma prior, never read).
 * @param verbose_              Logical flag enabling progress bar display (0 = off, non-zero = on)
 * @param bar_width_            Integer controlling progress bar width (clamped to 10-120 characters)
 *
 * @return SEXP R list containing posterior samples with named components:
 *         - theta_1: Numeric matrix [n_draws × n] of level state trajectory samples
 *         - theta_2: Numeric matrix [n_draws × n] of trend state trajectory samples
 *         - theta_01: Numeric vector [n_draws] of initial level state theta_{0,1} samples
 *         - theta_02: Numeric vector [n_draws] of initial trend state theta_{0,2} samples
 *         - prec_theta1: Numeric vector [n_draws] of level innovation precision 1/W_1 samples
 *         - prec_theta2: Numeric vector [n_draws] of trend innovation precision 1/W_2 samples
 *         - prec_y: Numeric vector [n_draws] of observation precision 1/V samples
 *
 * @note Computational complexity: O(n_iter × n) for n_iter total iterations.
 * @note Memory requirements: O(n) temporary storage for efficient buffer management.
 * @note RNG management: Proper GetRNGstate()/PutRNGstate() bracket for R integration.
 * @note Initialization: Starting values are decided in R and read from init_; this
 *       function draws none of them itself.
 *
 * @warning Minimum sample size n >= 3 enforced for numerical stability of recursions.
 * @warning Integer overflow protection: n <= INT_MAX due to R's integer limitations.
 * @warning Memory allocation failures will terminate R session via R_Calloc errors.
 * @warning No input validation for prior hyperparameters; negative values may cause crashes.
 * @warning No input validation for init_; it is assumed to have the documented
 *          length and to hold finite values, both guaranteed by resolve_init().
 *
 * @see generate_theta_p
 * @see generate_precision_theta_p
 * @see generate_theta_0p
 * @see generate_theta_1
 * @see generate_precision_theta_k
 * @see generate_theta_01
 * @see generate_precision_data
 */
SEXP C_MCMC_normal_localtrend(SEXP y_,
                              SEXP burnin_,
                              SEXP thinning_,
                              SEXP n_draws_,
                              SEXP prior_theta01_mean_,
                              SEXP prior_theta01_prec_,
                              SEXP prior_theta02_mean_,
                              SEXP prior_theta02_prec_,
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
                              SEXP prior_prec_y_type_,
                              SEXP prior_prec_y_shape_,
                              SEXP prior_prec_y_rate_,
                              SEXP prior_prec_y_scale_,
                              SEXP prior_prec_y_df_,
                              SEXP init_,
                              SEXP verbose_,
                              SEXP bar_width_);

#endif /* MCMC_NORMAL_LOCALTREND_H */
