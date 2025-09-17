/**
 * @file mcmc_binomial_localacceleration.h
 * @brief Header file for MCMC sampling of local-acceleration binomial dynamic models
 * @details Function declarations for Gibbs sampling of binomial state-space models
 *          with logit link and local-acceleration structure using component-wise
 *          Metropolis-Hastings algorithms.
 * @author Michel H. Montoril
 * @date 2025-08-18
 * @version 1.0
 */

#ifndef MCMC_BINOMIAL_LOCALACCELERATION_H
#define MCMC_BINOMIAL_LOCALACCELERATION_H

#include <R.h>
#include <Rinternals.h>

/**
 * @brief Gibbs sampler for local-acceleration binomial dynamic model with logit link
 *
 * @details Implements a complete Gibbs MCMC algorithm for the local-acceleration binomial model:
 *
 *          Observation equation:
 *          y_t ~ Binomial(n_trials, alpha_t)
 *          where alpha_t = logit^(-1)(theta_{1,t})
 *
 *          State equations:
 *          theta_{1,t} = theta_{1,t-1} + theta_{2,t-1} + u_{1,t},  u_{1,t} ~ N(0, W_1)
 *          theta_{2,t} = theta_{2,t-1} + theta_{3,t-1} + u_{2,t},  u_{2,t} ~ N(0, W_2)
 *          theta_{3,t} = theta_{3,t-1} + u_{3,t},                  u_{3,t} ~ N(0, W_3)
 *
 *          Uses component-wise Metropolis-Hastings for non-linear observation model
 *          with adaptive proposal tuning based on acceptance proportions.
 *
 *          The sampler cycles through conditional posteriors in the following order:
 *          1. State vector theta_3 | theta_2, theta_0, W_3 -> Gaussian posterior
 *          2. Innovation precision 1/W_3 | theta_3, theta_03 -> Gamma posterior
 *          3. Initial state theta_03 | theta_3, theta_02, W_3 -> Gaussian posterior
 *          4. State vector theta_2 | theta_1, theta_3, theta_0, W_2 -> Gaussian posterior
 *          5. Innovation precision 1/W_2 | theta_2, theta_02, theta_03 -> Gamma posterior
 *          6. Initial state theta_02 | theta_2, theta_01, theta_03, W_2 -> Gaussian posterior
 *          7. State vector theta_1 | y, theta_2, theta_0, W_1 -> Component-wise Metropolis-Hastings
 *          8. Innovation precision 1/W_1 | theta_1, theta_01, theta_02 -> Gamma posterior
 *          9. Initial state theta_01 | theta_1, theta_02, W_1 -> Gaussian posterior
 *
 *          Priors:
 *          - theta_{0,1} ~ N(mu_01, tau_01^{-1})
 *          - theta_{0,2} ~ N(mu_02, tau_02^{-1})
 *          - theta_{0,3} ~ N(mu_03, tau_03^{-1})
 *          - 1/W_1 ~ Gamma(nu_1, eta_1)
 *          - 1/W_2 ~ Gamma(nu_2, eta_2)
 *          - 1/W_3 ~ Gamma(nu_3, eta_3)
 *
 * @param y_                    SEXP Numeric vector of observed binomial counts [length n]
 * @param n_trials_             SEXP Double scalar, number of trials per observation
 * @param burnin_               SEXP Integer scalar, number of burn-in iterations
 * @param thinning_             SEXP Integer scalar, thinning interval
 * @param n_chain_              SEXP Integer scalar, target number of retained samples
 * @param prior_theta01_mean_   SEXP Double scalar, prior mean for initial state theta_{0,1}
 * @param prior_theta01_prec_   SEXP Double scalar, prior precision for initial state theta_{0,1}
 * @param prior_theta02_mean_   SEXP Double scalar, prior mean for initial state theta_{0,2}
 * @param prior_theta02_prec_   SEXP Double scalar, prior precision for initial state theta_{0,2}
 * @param prior_theta03_mean_   SEXP Double scalar, prior mean for initial state theta_{0,3}
 * @param prior_theta03_prec_   SEXP Double scalar, prior precision for initial state theta_{0,3}
 * @param prior_prec1_shape_    SEXP Double scalar, shape parameter for Gamma prior on 1/W_1
 * @param prior_prec1_rate_     SEXP Double scalar, rate parameter for Gamma prior on 1/W_1
 * @param prior_prec2_shape_    SEXP Double scalar, shape parameter for Gamma prior on 1/W_2
 * @param prior_prec2_rate_     SEXP Double scalar, rate parameter for Gamma prior on 1/W_2
 * @param prior_prec3_shape_    SEXP Double scalar, shape parameter for Gamma prior on 1/W_3
 * @param prior_prec3_rate_     SEXP Double scalar, rate parameter for Gamma prior on 1/W_3
 * @param lag_update_           SEXP Integer scalar, adaptation frequency (iterations)
 * @param max_step_size_        SEXP Double scalar, maximum proposal step size
 * @param base_adaptation_rate_ SEXP Double scalar, base adaptation rate
 * @param decay_exponent_       SEXP Double scalar, adaptation decay exponent
 * @param target_acceptance_    SEXP Double scalar, target acceptance proportion
 * @param return_log_sigma_     SEXP Logical scalar, whether to return log_sigma diagnostics
 * @param return_accept_prop_   SEXP Logical scalar, whether to return accept_prop diagnostics
 *
 * @return SEXP R list containing posterior samples with named components:
 *         - theta_1: Numeric matrix [n_chain x n] of level state trajectory samples
 *         - theta_2: Numeric matrix [n_chain x n] of trend state trajectory samples
 *         - theta_3: Numeric matrix [n_chain x n] of acceleration state trajectory samples
 *         - theta_01: Numeric vector [n_chain] of initial level state samples
 *         - theta_02: Numeric vector [n_chain] of initial trend state samples
 *         - theta_03: Numeric vector [n_chain] of initial acceleration state samples
 *         - prec_1: Numeric vector [n_chain] of level innovation precision samples
 *         - prec_2: Numeric vector [n_chain] of trend innovation precision samples
 *         - prec_3: Numeric vector [n_chain] of acceleration innovation precision samples
 *         - alpha: Numeric matrix [n_chain x n] of success probability samples
 *         - log_sigma: Numeric matrix [n_chain x n] of proposal scales (if requested)
 *         - accept_prop: Numeric matrix [n_chain x n] of acceptance proportions (if requested)
 *
 * @note Computational complexity: O(n_iter x n) for n_iter total iterations
 * @note Memory requirements: O(n_iter x n) for trajectory and adaptation storage
 * @note Each y[i] must satisfy 0 <= y[i] <= n_trials
 * @note Sample size n >= 3 required for numerical stability
 * @note Uses diminishing adaptation with sliding window acceptance proportions
 *
 * @warning Prior parameters must be positive for proper Gamma distributions
 * @warning Large sample sizes may require substantial memory allocation
 * @warning No convergence diagnostics implemented; user must assess convergence
 *
 * @see generate_alpha_logit_binomial
 * @see generate_precision_theta_p
 * @see generate_theta_0p
 * @see generate_theta_k
 * @see generate_precision_theta_k
 * @see generate_theta_0k
 */
SEXP C_MCMC_logit_binomial_localacceleration(SEXP y_, SEXP n_trials_,
                                            SEXP burnin_, SEXP thinning_, SEXP n_chain_,
                                            SEXP prior_theta01_mean_, SEXP prior_theta01_prec_,
                                            SEXP prior_theta02_mean_, SEXP prior_theta02_prec_,
                                            SEXP prior_theta03_mean_, SEXP prior_theta03_prec_,
                                            SEXP prior_prec1_shape_, SEXP prior_prec1_rate_,
                                            SEXP prior_prec2_shape_, SEXP prior_prec2_rate_,
                                            SEXP prior_prec3_shape_, SEXP prior_prec3_rate_,
                                            SEXP lag_update_, SEXP max_step_size_,
                                            SEXP base_adaptation_rate_, SEXP decay_exponent_,
                                            SEXP target_acceptance_,
                                            SEXP return_log_sigma_, SEXP return_accept_prop_);

#endif /* MCMC_BINOMIAL_LOCALACCELERATION_H */