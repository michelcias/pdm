/**
 * @file mcmc_poisson_locallevel.h
 * @brief Header for MCMC sampling in local-level Poisson dynamic models
 * @author Michel H. Montoril
 * @date 2025-12-20
 * @version 1.0
 *
 * @details This header declares complete Gibbs samplers for Bayesian estimation of
 *          Poisson dynamic models with local-level structure. 
 *
 *          **Key features:**
 *          - Log-Poisson:  Component-wise MH with adaptive tuning
 *          - Memory-efficient O(n) temporary storage
 *          - Conditional alpha computation for performance
 *          - Conjugate posterior updates for variance and initial state
 *          - Flexible burn-in and thinning controls
 *
 *          **Model specification:**
 *          Local-level state equation:
 *          theta_{t,1} = theta_{t-1,1} + u_{t,1}, u_{t,1} ~ N(0, W_1)
 *
 *          Observation equation with log link:
 *          y_t ~ Poisson(exp(theta_{t,1}))
 */

#ifndef MCMC_POISSON_LOCALLEVEL_H
#define MCMC_POISSON_LOCALLEVEL_H

#include <R.h>
#include <Rinternals.h>

/**
 * @brief Gibbs sampler for local-level Poisson dynamic model with log link
 *
 * @details Complete Gibbs MCMC for Poisson observations with log link.
 *          Uses component-wise Metropolis-Hastings for non-linear observation model
 *          with adaptive proposal tuning.
 *
 *          Model:  y_t ~ Poisson(exp(theta_{t,1}))
 *          State:  theta_{t,1} = theta_{t-1,1} + u_{t,1}
 *
 * @param y_                       Observed Poisson counts [n]
 * @param burnin_                  Burn-in iterations (discarded)
 * @param thinning_                Thinning interval
 * @param n_chain_                 Number of retained samples
 * @param prior_theta01_mean_      Prior mean for theta_{0,1}
 * @param prior_theta01_prec_      Prior precision for theta_{0,1}
 * @param prior_prec1_shape_       Gamma shape for 1/W_1
 * @param prior_prec1_rate_        Gamma rate for 1/W_1
 * @param lag_update_              Adaptation frequency
 * @param max_step_size_           Maximum proposal step size
 * @param base_adaptation_rate_    Base adaptation rate
 * @param decay_exponent_          Adaptation decay exponent
 * @param target_acceptance_       Target acceptance rate
 * @param min_deviation_threshold_ Minimum deviation to trigger adaptation
 * @param return_log_sigma_        Flag for log_sigma diagnostics
 * @param return_accept_prop_      Flag for accept_prop diagnostics
 * @param verbose_                 Flag for progress bar display (0 = off, non-zero = on)
 * @param bar_width_               Width of progress bar in characters (10-120 recommended)
 *
 * @return List with components:
 *         - theta_1: Matrix [n_chain * n] of state samples
 *         - theta_01: Vector [n_chain] of initial state samples
 *         - prec_theta1: Vector [n_chain] of precision samples
 *         - alpha:  Matrix [n_chain * n] of rate samples
 *         - log_sigma: Matrix [n_chain * n] (if requested)
 *         - accept_prop: Matrix [n_chain * n] (if requested)
 *
 * @note Complexity: O(n_iter * n) time, O(n) space
 * @note Requires n >= 3 for stability
 * @note Each y[t] must be a non-negative integer
 *
 * @see generate_alpha_log_poisson_locallevel
 * @see generate_precision_theta_p
 * @see generate_theta_01_locallevel
 */
SEXP C_MCMC_log_poisson_locallevel(SEXP y_,
                                   SEXP burnin_,
                                   SEXP thinning_,
                                   SEXP n_chain_,
                                   SEXP prior_theta01_mean_,
                                   SEXP prior_theta01_prec_,
                                   SEXP prior_prec1_shape_,
                                   SEXP prior_prec1_rate_,
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

#endif /* MCMC_POISSON_LOCALLEVEL_H */