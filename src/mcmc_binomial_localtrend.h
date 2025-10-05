/**
 * @file mcmc_binomial_localtrend.h
 * @brief Header file for MCMC sampling of local-trend binomial dynamic models
 * @details Function declarations for Gibbs sampling of binomial state-space models
 *          with logit link and local-trend structure using component-wise
 *          Metropolis-Hastings algorithms with configurable adaptation threshold.
 * @author Michel H. Montoril
 * @date 2025-09-27
 * @version 1.2
 *
 * @changelog
 * - v1.2 (2025-09-27): Updated function signature to include min_deviation_threshold
 *   parameter for enhanced adaptation control.
 */

#ifndef MCMC_BINOMIAL_LOCALTREND_H
#define MCMC_BINOMIAL_LOCALTREND_H

#include <R.h>
#include <Rinternals.h>

/**
 * @brief Gibbs sampler for local-trend binomial dynamic model with logit link - Optimized version
 *
 * @details Implements a complete optimized Gibbs MCMC algorithm for the local-trend binomial model:
 *
 *          Observation equation:
 *          y_t ~ Binomial(n_trials, alpha_t)
 *          where alpha_t = logit^(-1)(theta_{t,1})
 *
 *          State equations:
 *          theta_{t,1} = theta_{t-1,1} + theta_{t-1,2} + u_{t,1},  u_{t,1} ~ N(0, W_1)
 *          theta_{t,2} = theta_{t-1,2} + u_{t,2},                  u_{t,2} ~ N(0, W_2)
 *
 *          The algorithm employs optimized component-wise Metropolis-Hastings for the non-linear
 *          observation model, with adaptive proposal tuning based on acceptance proportions.
 *          Innovation precisions are sampled from conjugate Gamma posteriors.
 *
 *          **Optimizations implemented:**
 *          - Cached precision computations to avoid repeated sqrt/division
 *          - Reduced memory allocation by eliminating redundant arrays
 *          - Sliding window memory optimization for theta_1_updated
 *          - Stable log-probability computations
 *          - Configurable adaptation threshold with practical default (1.0/lag_update)
 *
 *          **Version 1.2 enhancements:**
 *          Enhanced flexibility by computing and passing practical adaptation threshold
 *          to component-wise sampling functions. This ensures optimal adaptation behavior
 *          while maintaining interface compatibility.
 *
 *          Sampling sequence per iteration:
 *          1. theta_2 | theta_1, theta_0, W_2 -> Gaussian posterior (conditional state)
 *          2. 1/W_2 | theta_2, theta_02 -> Gamma posterior
 *          3. theta_{0,2} | theta_2, theta_01, W_2 -> Gaussian posterior
 *          4. theta_1 | y, theta_2, theta_0, W_1 -> Component-wise Metropolis-Hastings with adaptive threshold
 *          5. 1/W_1 | theta_1, theta_01, theta_02 -> Gamma posterior
 *          6. theta_{0,1} | theta_1, theta_02, W_1 -> Gaussian posterior
 *
 *          Total iterations computed as: burnin + (n_chain - 1) x thinning + 1
 *
 *          Priors:
 *          - theta_{0,1} ~ N(mu_01, tau_01^{-1})
 *          - theta_{0,2} ~ N(mu_02, tau_02^{-1})
 *          - 1/W_1 ~ Gamma(nu_1, eta_1)
 *          - 1/W_2 ~ Gamma(nu_2, eta_2)
 *
 * @param y_                       SEXP Numeric vector of observed binomial counts [length n]
 * @param n_trials_                SEXP Double scalar, number of trials per observation
 * @param burnin_                  SEXP Integer scalar, number of burn-in iterations
 * @param thinning_                SEXP Integer scalar, thinning interval
 * @param n_chain_                 SEXP Integer scalar, target number of retained samples
 * @param prior_theta01_mean_      SEXP Double scalar, prior mean mu_01 for initial state theta_{0,1}
 * @param prior_theta01_prec_      SEXP Double scalar, prior precision tau_01 for initial state theta_{0,1}
 * @param prior_theta02_mean_      SEXP Double scalar, prior mean mu_02 for initial state theta_{0,2}
 * @param prior_theta02_prec_      SEXP Double scalar, prior precision tau_02 for initial state theta_{0,2}
 * @param prior_prec1_shape_       SEXP Double scalar, shape parameter nu_1 for Gamma prior on 1/W_1
 * @param prior_prec1_rate_        SEXP Double scalar, rate parameter eta_1 for Gamma prior on 1/W_1
 * @param prior_prec2_shape_       SEXP Double scalar, shape parameter nu_2 for Gamma prior on 1/W_2
 * @param prior_prec2_rate_        SEXP Double scalar, rate parameter eta_2 for Gamma prior on 1/W_2
 * @param lag_update_              SEXP Integer scalar, adaptation frequency (iterations)
 * @param max_step_size_           SEXP Double scalar, maximum proposal step size
 * @param base_adaptation_rate_    SEXP Double scalar, base adaptation rate
 * @param decay_exponent_          SEXP Double scalar, adaptation decay exponent
 * @param target_acceptance_       SEXP Double scalar, target acceptance proportion
 * @param min_deviation_threshold_ SEXP Double scalar, minimum absolute deviation from
 *                                      target_acceptance required to trigger log_sigma updates.
 *                                      Values >= 0.
 * @param return_log_sigma_        SEXP Logical scalar, whether to return log_sigma diagnostics
 * @param return_accept_prop_      SEXP Logical scalar, whether to return accept_prop diagnostics
 *
 * @return SEXP R list containing posterior samples with named components:
 *         - theta_1:     Numeric matrix [n_chain x n] of level state trajectory samples
 *         - theta_2:     Numeric matrix [n_chain x n] of trend state trajectory samples
 *         - theta_01:    Numeric vector [n_chain] of initial level state samples
 *         - theta_02:    Numeric vector [n_chain] of initial trend state samples
 *         - prec_1:      Numeric vector [n_chain] of level innovation precision samples
 *         - prec_2:      Numeric vector [n_chain] of trend innovation precision samples
 *         - alpha:       Numeric matrix [n_chain x n] of success probability samples
 *         - log_sigma:   Numeric matrix [n_chain x n] of proposal scales (if requested)
 *         - accept_prop: Numeric matrix [n_chain x n] of acceptance proportions (if requested)
 *
 * @note Computational complexity: O(n_iter x n) where n_iter = burnin + (n_chain-1)*thinning + 1
 * @note Memory requirements: O(lag_update x n) for optimized sliding window + O(n_iter x n) for trajectory storage
 * @note RNG management: Proper GetRNGstate()/PutRNGstate() bracket for R integration
 * @note Adaptation: Uses practical threshold for optimal sensitivity control
 * @note Memory optimization: theta_1_updated uses sliding window instead of full matrix
 *
 * @warning Minimum sample size n >= 3 required for numerical stability
 * @warning Each y[i] must satisfy 0 <= y[i] <= n_trials
 * @warning Large sample sizes (n > INT_MAX) not supported due to R integer limitations
 * @warning Prior parameters must be positive for proper Gamma distributions
 * @warning Memory allocation failures will terminate R session via R_Calloc errors
 *
 * @see generate_theta_p
 * @see generate_precision_theta_p
 * @see generate_theta_0p
 * @see generate_alpha_logit_binomial
 * @see generate_precision_theta_k
 * @see generate_theta_01
 */
SEXP C_MCMC_logit_binomial_localtrend(SEXP y_, SEXP n_trials_,
                                      SEXP burnin_, SEXP thinning_, SEXP n_chain_,
                                      SEXP prior_theta01_mean_, SEXP prior_theta01_prec_,
                                      SEXP prior_theta02_mean_, SEXP prior_theta02_prec_,
                                      SEXP prior_prec1_shape_, SEXP prior_prec1_rate_,
                                      SEXP prior_prec2_shape_, SEXP prior_prec2_rate_,
                                      SEXP lag_update_, SEXP max_step_size_,
                                      SEXP base_adaptation_rate_, SEXP decay_exponent_,
                                      SEXP target_acceptance_, SEXP min_deviation_threshold_,
                                      SEXP return_log_sigma_, SEXP return_accept_prop_);

/**
 * @brief Gibbs sampler for local-trend Bernoulli dynamic model with probit link
 *
 * @details Implements the Albert-Chib (1993) latent-variable augmentation for
 *          Bernoulli observations governed by a local-trend Gaussian state
 *          evolution. Returns retained posterior draws for level and trend
 *          states, innovation precisions, and Bernoulli probabilities.
 *
 * @param y_                   SEXP Numeric vector of Bernoulli observations [length n]
 * @param burnin_              SEXP Integer scalar, number of burn-in iterations
 * @param thinning_            SEXP Integer scalar, thinning interval
 * @param n_chain_             SEXP Integer scalar, number of retained samples
 * @param prior_theta01_mean_  SEXP Double scalar, prior mean for theta_{0,1}
 * @param prior_theta01_prec_  SEXP Double scalar, prior precision for theta_{0,1}
 * @param prior_theta02_mean_  SEXP Double scalar, prior mean for theta_{0,2}
 * @param prior_theta02_prec_  SEXP Double scalar, prior precision for theta_{0,2}
 * @param prior_prec1_shape_   SEXP Double scalar, prior shape for 1/W_1
 * @param prior_prec1_rate_    SEXP Double scalar, prior rate for 1/W_1
 * @param prior_prec2_shape_   SEXP Double scalar, prior shape for 1/W_2
 * @param prior_prec2_rate_    SEXP Double scalar, prior rate for 1/W_2
 *
 * @return SEXP R list with components theta_1, theta_2, theta_01, theta_02,
 *         prec_1, prec_2, and alpha.
 */
SEXP C_MCMC_probit_bernoulli_localtrend(SEXP y_,
                                        SEXP burnin_,
                                        SEXP thinning_,
                                        SEXP n_chain_,
                                        SEXP prior_theta01_mean_,
                                        SEXP prior_theta01_prec_,
                                        SEXP prior_theta02_mean_,
                                        SEXP prior_theta02_prec_,
                                        SEXP prior_prec1_shape_,
                                        SEXP prior_prec1_rate_,
                                        SEXP prior_prec2_shape_,
                                        SEXP prior_prec2_rate_);

#endif /* MCMC_BINOMIAL_LOCALTREND_H */
