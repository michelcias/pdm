/**
 * @file mcmc_localacceleration.h
 * @brief MCMC implementation for local-acceleration dynamic models
 * @details This module provides Gibbs sampling functionality for polynomial dynamic models
 *          with local-acceleration structure (p=3), implementing Bayesian estimation through
 *          conditional posterior distributions for level, trend, and acceleration components.
 * @author Michel H. Montoril
 * @date 2025-01-11
 * @version 1.0
 */

#ifndef MCMC_LOCALACCELERATION_H
#define MCMC_LOCALACCELERATION_H

#include <R.h>
#include <Rinternals.h>
#include <Rmath.h>

/**
 * @brief Gibbs sampler for local-acceleration dynamic model with polynomial structure
 *
 * @details Implements a complete Gibbs MCMC algorithm for the local-acceleration dynamic model:
 *
 *          Observation equation:
 *          y_t = theta_{t,1} + e_t,     e_t ~ N(0, V)
 *
 *          State equations:
 *          theta_{t,1} = theta_{t-1,1} + theta_{t-1,2} + u_{t,1},  u_{t,1} ~ N(0, W_1)
 *          theta_{t,2} = theta_{t-1,2} + theta_{t-1,3} + u_{t,2},  u_{t,2} ~ N(0, W_2)
 *          theta_{t,3} = theta_{t-1,3} + u_{t,3},                  u_{t,3} ~ N(0, W_3)
 *
 *          The sampler cycles through conditional posteriors in the following order:
 *          1. State vector theta_3 | theta_2, theta_0, W_3
 *          2. Innovation precision 1/W_3 | theta_3, theta_03
 *          3. Initial state theta_03 | theta_3, theta_02, W_3
 *          4. State vector theta_2 | theta_1, theta_3, theta_0, W_2
 *          5. Innovation precision 1/W_2 | theta_2, theta_02, theta_03
 *          6. Initial state theta_02 | theta_2, theta_01, theta_03, W_2
 *          7. State vector theta_1 | y, theta_2, theta_0, W_1, V
 *          8. Innovation precision 1/W_1 | theta_1, theta_01, theta_02
 *          9. Initial state theta_01 | theta_1, theta_02, W_1
 *          10. Data precision 1/V | y, theta_1
 *
 *          Priors:
 *          - theta_{0,1} ~ N(mu_01, tau_01^{-1})
 *          - theta_{0,2} ~ N(mu_02, tau_02^{-1})
 *          - theta_{0,3} ~ N(mu_03, tau_03^{-1})
 *          - 1/W_1 ~ Gamma(nu_1, eta_1)
 *          - 1/W_2 ~ Gamma(nu_2, eta_2)
 *          - 1/W_3 ~ Gamma(nu_3, eta_3)
 *          - 1/V ~ Gamma(nu_y, eta_y)
 *
 * @param y                    SEXP Numeric vector of observed time series data [length n]
 * @param burnin               SEXP Integer scalar, number of burn-in iterations to discard
 * @param thinning             SEXP Integer scalar, thinning interval for posterior samples
 * @param n_chain              SEXP Integer scalar, target number of retained posterior samples
 * @param prior_theta01_mean   SEXP Double scalar, prior mean mu_01 for initial state theta_{0,1}
 * @param prior_theta01_prec   SEXP Double scalar, prior precision tau_01 for initial state theta_{0,1}
 * @param prior_theta02_mean   SEXP Double scalar, prior mean mu_02 for initial state theta_{0,2}
 * @param prior_theta02_prec   SEXP Double scalar, prior precision tau_02 for initial state theta_{0,2}
 * @param prior_theta03_mean   SEXP Double scalar, prior mean mu_03 for initial state theta_{0,3}
 * @param prior_theta03_prec   SEXP Double scalar, prior precision tau_03 for initial state theta_{0,3}
 * @param prior_prec1_shape    SEXP Double scalar, shape parameter nu_1 for Gamma prior on 1/W_1
 * @param prior_prec1_rate     SEXP Double scalar, rate parameter eta_1 for Gamma prior on 1/W_1
 * @param prior_prec2_shape    SEXP Double scalar, shape parameter nu_2 for Gamma prior on 1/W_2
 * @param prior_prec2_rate     SEXP Double scalar, rate parameter eta_2 for Gamma prior on 1/W_2
 * @param prior_prec3_shape    SEXP Double scalar, shape parameter nu_3 for Gamma prior on 1/W_3
 * @param prior_prec3_rate     SEXP Double scalar, rate parameter eta_3 for Gamma prior on 1/W_3
 * @param prior_prec_y_shape   SEXP Double scalar, shape parameter nu_y for Gamma prior on 1/V
 * @param prior_prec_y_rate    SEXP Double scalar, rate parameter eta_y for Gamma prior on 1/V
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
 *         - prec_y: Numeric vector [n_chain] of observation precision samples
 *
 * @note Computational complexity: O(n_iter x n) where n_iter = burnin + (n_chain-1)*thinning + 1
 * @note Memory allocation: Requires O(n_iter x n) temporary storage for full MCMC trajectory
 * @note Numerical stability: Uses R's built-in random number generators with proper state management
 * @note Thread safety: Not thread-safe due to shared RNG state; use GetRNGstate()/PutRNGstate()
 *
 * @warning Minimum sample size n >= 3 required for numerical stability
 * @warning Large sample sizes (n > INT_MAX) not supported due to R integer limitations
 * @warning Prior parameters must be positive for proper Gamma distributions
 * @warning No convergence diagnostics implemented; user must assess chain convergence
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
SEXP C_MCMC_localacceleration(SEXP y_, SEXP burnin_, SEXP thinning_, SEXP n_chain_,
                              SEXP prior_theta01_mean_, SEXP prior_theta01_prec_,
                              SEXP prior_theta02_mean_, SEXP prior_theta02_prec_,
                              SEXP prior_theta03_mean_, SEXP prior_theta03_prec_,
                              SEXP prior_prec1_shape_, SEXP prior_prec1_rate_,
                              SEXP prior_prec2_shape_, SEXP prior_prec2_rate_,
                              SEXP prior_prec3_shape_, SEXP prior_prec3_rate_,
                              SEXP prior_prec_y_shape_, SEXP prior_prec_y_rate_);

#endif /* MCMC_LOCALACCELERATION_H */
