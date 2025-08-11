/**
 * @file mcmc_locallevel.h
 * @brief MCMC implementation for local-level dynamic models
 * @details This module provides Gibbs sampling functionality for polynomial dynamic models
 *          with local-level structure (p=1), implementing Bayesian estimation through
 *          conditional posterior distributions.
 * @author Michel H. Montoril
 * @date 2025-01-11
 * @version 1.0
 */

#ifndef MCMC_LOCALLEVEL_H
#define MCMC_LOCALLEVEL_H

#include <R.h>
#include <Rinternals.h>
#include <Rmath.h>

/**
 * @brief Gibbs sampler for local-level dynamic model with polynomial structure
 *
 * @details Implements a complete Gibbs MCMC algorithm for the local-level dynamic model:
 *
 *          Observation equation:
 *          y_t = theta_{t,1} + e_t,     e_t ~ N(0, V)
 *
 *          State equation:
 *          theta_{t,1} = theta_{t-1,1} + u_{t,1},  u_{t,1} ~ N(0, W_1)
 *
 *          The algorithm employs the Markovian structure of polynomial dynamic models
 *          to enable efficient block sampling of the state vector through forward-backward
 *          recursions. Innovation and observation precisions are sampled from their
 *          conjugate Gamma posteriors.
 *
 *          Sampling sequence per iteration:
 *          1. theta_1 | y, theta_0, W_1, V -> Forward-Backward algorithm
 *          2. 1/W_1 | theta_1, theta_0 -> Gamma posterior
 *          3. theta_{0,1} | theta_1, W_1 -> Normal posterior
 *          4. 1/V | y, theta_1 -> Gamma posterior
 *
 *          Total iterations computed as: burnin + (n_chain - 1) x thinning + 1
 *
 * @param y                    SEXP Numeric vector of observed time series data [length n]
 * @param burnin               SEXP Integer scalar, number of burn-in iterations to discard for chain convergence
 * @param thinning             SEXP Integer scalar, thinning interval to reduce autocorrelation in samples
 * @param n_chain              SEXP Integer scalar, target number of retained posterior samples
 * @param prior_theta01_mean   SEXP Double scalar, prior mean mu_0 for initial state theta_{0,1}
 * @param prior_theta01_prec   SEXP Double scalar, prior precision tau_0 = 1/sigma_0^2 for initial state
 * @param prior_prec1_shape    SEXP Double scalar, shape parameter nu_1 for Gamma(nu_1, eta_1) prior on 1/W_1
 * @param prior_prec1_rate     SEXP Double scalar, rate parameter eta_1 for Gamma(nu_1, eta_1) prior on 1/W_1
 * @param prior_prec_y_shape   SEXP Double scalar, shape parameter nu_y for Gamma(nu_y, eta_y) prior on 1/V
 * @param prior_prec_y_rate    SEXP Double scalar, rate parameter eta_y for Gamma(nu_y, eta_y) prior on 1/V
 *
 * @return SEXP R list containing posterior samples with named components:
 *         - theta_1: Numeric matrix [n_chain x n] of complete state trajectory samples
 *         - theta_01: Numeric vector [n_chain] of initial state theta_{0,1} samples
 *         - prec_1: Numeric vector [n_chain] of innovation precision 1/W_1 samples
 *         - prec_y: Numeric vector [n_chain] of observation precision 1/V samples
 *
 * @note Computational complexity: O(n_iter x n) for n_iter total iterations
 * @note Memory requirements: O(n_iter x n) for temporary trajectory storage
 * @note RNG management: Proper GetRNGstate()/PutRNGstate() bracket for R integration
 * @note Initialization: Uses prior-based random initialization for all parameters
 *
 * @warning Minimum sample size n >= 3 enforced for numerical stability of recursions
 * @warning Integer overflow protection: n <= INT_MAX due to R's integer limitations
 * @warning Memory allocation failures will terminate R session via R_Calloc errors
 * @warning No input validation for prior hyperparameters; negative values may cause crashes
 *
 * @see generate_theta_1_locallevel
 * @see generate_precision_theta_p
 * @see generate_theta_01_locallevel
 * @see generate_precision_data
 */
SEXP C_MCMC_locallevel(
    SEXP y_,
    SEXP burnin_,
    SEXP thinning_,
    SEXP n_chain_,
    SEXP prior_theta01_mean_,
    SEXP prior_theta01_prec_,
    SEXP prior_prec1_shape_,
    SEXP prior_prec1_rate_,
    SEXP prior_prec_y_shape_,
    SEXP prior_prec_y_rate_
);

#endif /* MCMC_LOCALLEVEL_H */
