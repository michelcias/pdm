/**
 * @file mcmc_binomial_locallevel.h
 * @brief Header file for MCMC sampling of local-level binomial dynamic models
 * @details Function declarations for Gibbs sampling of binomial state-space models
 *          with logit link and local-level structure using component-wise
 *          Metropolis-Hastings algorithms.
 * @author Michel H. Montoril
 * @date 2025-09-23
 * @version 1.2
 */

#ifndef MCMC_BINOMIAL_LOCALLEVEL_H
#define MCMC_BINOMIAL_LOCALLEVEL_H

#include <R.h>
#include <Rinternals.h>

/**
 * @brief Gibbs sampler for local-level binomial dynamic model with logit link - Optimized version
 *
 * @details Implements a complete optimized Gibbs MCMC algorithm for the local-level binomial model:
 *
 *          Observation equation:
 *          y_t ~ Binomial(n_trials, alpha_t)
 *          where alpha_t = logit^(-1)(theta_{1,t})
 *
 *          State equation:
 *          theta_{1,t} = theta_{1,t-1} + u_{1,t},  u_{1,t} ~ N(0, W_1)
 *
 *          The algorithm employs optimized component-wise Metropolis-Hastings for the non-linear
 *          observation model, with adaptive proposal tuning based on acceptance proportions.
 *          Innovation precision is sampled from conjugate Gamma posterior.
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
 *          1. theta_1 | y, theta_0, W_1 -> Component-wise Metropolis-Hastings with adaptive threshold
 *          2. 1/W_1 | theta_1, theta_0 -> Gamma posterior
 *          3. theta_{0,1} | theta_1, W_1 -> Normal posterior
 *
 *          Total iterations computed as: burnin + (n_chain - 1) x thinning + 1
 *
 * @param y_                       SEXP Numeric vector of observed binomial counts [length n]
 * @param n_trials_                SEXP Double scalar, number of trials per observation
 * @param burnin_                  SEXP Integer scalar, number of burn-in iterations
 * @param thinning_                SEXP Integer scalar, thinning interval
 * @param n_chain_                 SEXP Integer scalar, target number of retained samples
 * @param prior_theta01_mean_      SEXP Double scalar, prior mean for initial state theta_{0,1}
 * @param prior_theta01_prec_      SEXP Double scalar, prior precision for initial state
 * @param prior_prec1_shape_       SEXP Double scalar, shape parameter for Gamma prior on 1/W_1
 * @param prior_prec1_rate_        SEXP Double scalar, rate parameter for Gamma prior on 1/W_1
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
 *         - theta_1:     Numeric matrix [n_chain x n] of state trajectory samples
 *         - theta_01:    Numeric vector [n_chain] of initial state samples
 *         - prec_1:      Numeric vector [n_chain] of innovation precision samples
 *         - alpha:       Numeric matrix [n_chain x n] of success probability samples
 *         - log_sigma:   Numeric matrix [n_chain x n] of proposal scales (if requested)
 *         - accept_prop: Numeric matrix [n_chain x n] of acceptance proportions (if requested)
 *
 * @note Computational complexity: O(n_iter x n) for n_iter total iterations
 * @note Memory requirements: O(lag_update x n) for optimized sliding window + O(n_iter x n) for trajectory storage
 * @note RNG management: Proper GetRNGstate()/PutRNGstate() bracket for R integration
 * @note Adaptation: Uses practical threshold of 1.0/lag_update for optimal sensitivity
 * @note Memory optimization: theta_1_updated uses sliding window instead of full B x n matrix
 *
 * @warning Minimum sample size n >= 3 enforced for numerical stability
 * @warning Each y[i] must satisfy 0 <= y[i] <= n_trials
 * @warning Memory allocation failures will terminate R session via R_Calloc errors
 *
 * @see generate_alpha_logit_binomial_locallevel
 * @see generate_precision_theta_p
 * @see generate_theta_01_locallevel
 */
SEXP C_MCMC_logit_binomial_locallevel(SEXP y_, SEXP n_trials_,
                                      SEXP burnin_, SEXP thinning_, SEXP n_chain_,
                                      SEXP prior_theta01_mean_, SEXP prior_theta01_prec_,
                                      SEXP prior_prec1_shape_, SEXP prior_prec1_rate_,
                                      SEXP lag_update_, SEXP max_step_size_,
                                      SEXP base_adaptation_rate_, SEXP decay_exponent_,
                                      SEXP target_acceptance_, SEXP min_deviation_threshold_,
                                      SEXP return_log_sigma_, SEXP return_accept_prop_);

#endif /* MCMC_BINOMIAL_LOCALLEVEL_H */
