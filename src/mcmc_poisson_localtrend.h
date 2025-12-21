/**
 * @file mcmc_poisson_localtrend.h
 * @brief Header for MCMC sampling in local-trend Poisson dynamic models
 * @author Michel H. Montoril
 * @date 2025-12-20
 * @version 1.0
 *
 * @details This header declares complete Gibbs samplers for Bayesian estimation of
 *          Poisson dynamic models with local-trend structure. 
 *
 *          **Key features:**
 *          - Log-Poisson:  Component-wise MH with adaptive tuning
 *          - Memory-efficient O(n) temporary storage
 *          - Conditional alpha computation for performance
 *          - Conjugate posterior updates for variance and initial state
 *          - Flexible burn-in and thinning controls
 *
 *          **Model specification:**
 *          Local-trend state equations:
 *          theta_{t,1} = theta_{t-1,1} + theta_{t-1,2} + u_{t,1}, u_{t,1} ~ N(0, W_1)
 *          theta_{t,2} = theta_{t-1,2} + u_{t,2},                 u_{t,2} ~ N(0, W_2)
 *
 *          Observation equation with log link:
 *          y_t ~ Poisson(exp(theta_{t,1}))
 */

#ifndef MCMC_POISSON_LOCALTREND_H
#define MCMC_POISSON_LOCALTREND_H

#include <R.h>
#include <Rinternals.h>

/**
 * @brief Gibbs sampler for local-trend Poisson dynamic model with log link
 *
 * @details Implements a complete Gibbs MCMC algorithm for the local-trend Poisson model: 
 *
 *          **Observation equation:**
 *          y_t ~ Poisson(alpha_t), where alpha_t = exp(theta_{t,1})
 *
 *          **State equations:**
 *          theta_{t,1} = theta_{t-1,1} + theta_{t-1,2} + u_{t,1}, u_{t,1} ~ N(0, W_1)
 *          theta_{t,2} = theta_{t-1,2} + u_{t,2},                 u_{t,2} ~ N(0, W_2)
 *
 *          **Prior distributions:**
 *          - theta_{0,1} ~ N(mu_{0,1}, sigma_{0,1}^2)
 *          - theta_{0,2} ~ N(mu_{0,2}, sigma_{0,2}^2)
 *          - 1/W_1 ~ Gamma(nu_1, eta_1)
 *          - 1/W_2 ~ Gamma(nu_2, eta_2)
 *
 *          The algorithm employs component-wise Metropolis-Hastings for the non-linear
 *          observation model with adaptive proposal tuning based on acceptance proportions.
 *
 *          **Optimizations implemented:**
 *          - Memory-efficient current/previous iteration buffers (O(n) storage)
 *          - Conditional alpha computation (skip during burn-in/thinning)
 *          - Configurable adaptation threshold (practical default: 1. 0/lag_update)
 *          - Scalar parameter passing to avoid array indexing
 *          - Efficient initialization with zeros and neutral starting values
 *          - Optional visual progress bar with adaptive update frequency
 *
 *          **Sampling sequence per iteration:**
 *          1. theta_2 | theta_1, theta_02, W_2 -> Gaussian posterior
 *          2. 1/W_2 | theta_2, theta_02 -> Gamma posterior
 *          3. theta_{0,2} | theta_2, theta_01, W_2 -> Gaussian posterior
 *          4. theta_1, alpha | y, theta_2, theta_01, W_1 -> Component-wise MH with adaptive tuning
 *          5. 1/W_1 | theta_1, theta_01, theta_02 -> Gamma posterior
 *          6. theta_{0,1} | theta_1, theta_02, W_1 -> Gaussian posterior
 *
 *          Total iterations: burnin + (n_chain - 1) * thinning + 1
 *
 * @param y_                       Numeric vector [n] of observed Poisson counts. 
 * @param burnin_                  Number of burn-in iterations (discarded).
 * @param thinning_                Thinning interval for autocorrelation reduction.
 * @param n_chain_                 Number of retained posterior samples.
 * @param prior_theta01_mean_      Prior mean for theta_{0,1}. 
 * @param prior_theta01_prec_      Prior precision for theta_{0,1}. 
 * @param prior_theta02_mean_      Prior mean for theta_{0,2}.
 * @param prior_theta02_prec_      Prior precision for theta_{0,2}.
 * @param prior_prec1_shape_       Gamma shape for 1/W_1.
 * @param prior_prec1_rate_        Gamma rate for 1/W_1.
 * @param prior_prec2_shape_       Gamma shape for 1/W_2.
 * @param prior_prec2_rate_        Gamma rate for 1/W_2.
 * @param lag_update_              Adaptation frequency (iterations).
 * @param max_step_size_           Maximum proposal step size.
 * @param base_adaptation_rate_    Base adaptation rate. 
 * @param decay_exponent_          Adaptation decay exponent.
 * @param target_acceptance_       Target acceptance proportion.
 * @param min_deviation_threshold_ Minimum deviation to trigger adaptation (>= 0).
 * @param return_log_sigma_        Flag to return log_sigma diagnostics.
 * @param return_accept_prop_      Flag to return accept_prop diagnostics.
 * @param verbose_                 Logical:  display progress bar (0 = FALSE, 1 = TRUE).
 * @param bar_width_               Integer: width of progress bar in characters (10-120 recommended).
 *
 * @return R list with components:
 *         - theta_1:     Matrix [n_chain * n] of level state trajectory samples
 *         - theta_2:     Matrix [n_chain * n] of trend state trajectory samples
 *         - theta_01:    Vector [n_chain] of initial level state samples
 *         - theta_02:    Vector [n_chain] of initial trend state samples
 *         - prec_theta1: Vector [n_chain] of level innovation precision samples
 *         - prec_theta2: Vector [n_chain] of trend innovation precision samples
 *         - alpha:       Matrix [n_chain * n] of rate parameter samples
 *         - log_sigma:   Matrix [n_chain * n] of proposal scales (if requested)
 *         - accept_prop:  Matrix [n_chain * n] of acceptance proportions (if requested)
 *
 * @note Complexity: O(n_iter * n) time, O(n) space
 * @note Requires n >= 3 for numerical stability
 * @note Proper RNG state management via GetRNGstate()/PutRNGstate()
 * @note Adaptation threshold: practical default is 1.0/lag_update
 * @note Progress bar updates approximately once per bar segment (adaptive frequency)
 * @note Minimal performance overhead from progress bar (~0.01% for typical runs)
 *
 * @warning Each y[t] must be a non-negative integer
 * @warning n must not exceed INT_MAX
 * @warning Memory allocation failures terminate R session
 *
 * @see generate_alpha_log_poisson
 * @see generate_theta_p
 * @see generate_precision_theta_p
 * @see generate_theta_0p
 * @see generate_precision_theta_k
 * @see generate_theta_01
 */
SEXP C_MCMC_log_poisson_localtrend(SEXP y_,
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
                                   SEXP prior_prec2_rate_,
                                   SEXP lag_update_,
                                   SEXP max_step_size_,
                                   SEXP base_adaptation_rate_,
                                   SEXP decay_exponent_,
                                   SEXP target_acceptance_,
                                   SEXP min_deviation_threshold_,
                                   SEXP return_log_sigma_,
                                   SEXP return_accept_prop_,
                                   SEXP verbose_,
                                   SEXP bar_width_);

#endif /* MCMC_POISSON_LOCALTREND_H */