/**
 * @file mcmc_binomial_locallevel.h
 * @brief Header for MCMC sampling in local-level binomial and Bernoulli dynamic models
 * @author Michel H. Montoril
 * @date 2026-07-26
 * @version 1.2
 *
 * @details This header declares complete Gibbs samplers for Bayesian estimation of
 *          binomial and Bernoulli dynamic models with local-level structure.
 *
 *          **Key features:**
 *          - Logit-binomial: Component-wise MH with adaptive tuning
 *          - Probit-Bernoulli: Gibbs sampling via Albert-Chib augmentation
 *          - Memory-efficient O(n) temporary storage
 *          - Conditional alpha computation for performance
 *          - Conjugate posterior updates for variance and initial state
 *          - Flexible burn-in and thinning controls
 *
 *          **Model specifications:**
 *          Both models share the local-level state equation:
 *          theta_{t,1} = theta_{t-1,1} + u_{t,1}, u_{t,1} ~ N(0, W_1)
 *
 *          Observation equations differ by link function:
 *          - Logit-binomial:   y_t ~ Binomial(n_trials, logit^{-1}(theta_{t,1}))
 *          - Probit-Bernoulli: y_t ~ Bernoulli(Phi(theta_{t,1}))
 */

#ifndef MCMC_BINOMIAL_LOCALLEVEL_H
#define MCMC_BINOMIAL_LOCALLEVEL_H

#include <R.h>
#include <Rinternals.h>

/**
 * @brief Gibbs sampler for local-level binomial dynamic model with logit link
 *
 * @details Complete Gibbs MCMC for binomial observations with logit link.
 *          Uses component-wise Metropolis-Hastings for non-linear observation model
 *          with adaptive proposal tuning.
 *
 *          Model: y_t ~ Binomial(n_trials, logit^{-1}(theta_{t,1}))
 *          State: theta_{t,1} = theta_{t-1,1} + u_{t,1}
 *
 * @param y_                       Observed binomial counts [n]
 * @param n_trials_                Number of trials per observation
 * @param burnin_                  Burn-in iterations (discarded)
 * @param thinning_                Thinning interval
 * @param n_draws_                 Number of retained samples
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
 * @param init_                    Double vector [3] of starting values, resolved in R by
 *                                 resolve_init(): theta_{0,1}, then 1/W_1, then the Half-t
 *                                 auxiliary of that precision (0 under a Gamma prior,
 *                                 where it is never read).
 * @param verbose_                 Flag for progress bar display (0 = off, non-zero = on)
 * @param bar_width_               Width of progress bar in characters (10-120 recommended)
 *
 * @return List with components:
 *         - theta_1: Matrix [n_draws * n] of state samples
 *         - theta_01: Vector [n_draws] of initial state samples
 *         - prec_theta1: Vector [n_draws] of precision samples
 *         - alpha: Matrix [n_draws * n] of probability samples
 *         - log_sigma: Matrix [n_draws * n] (if requested)
 *         - accept_prop: Matrix [n_draws * n] (if requested)
 *
 * @note Complexity: O(n_iter * n) time, O(n) space
 * @note Requires n >= 3 for stability
 * @note Each y[t] must satisfy 0 <= y[t] <= n_trials
 *
 * @see generate_alpha_logit_binomial_locallevel
 * @see generate_precision_theta_p
 * @see generate_theta_01_locallevel
 */
SEXP C_MCMC_logit_binomial_locallevel(SEXP y_,
                                      SEXP n_trials_,
                                      SEXP burnin_,
                                      SEXP thinning_,
                                      SEXP n_draws_,
                                      SEXP prior_theta01_mean_,
                                      SEXP prior_theta01_prec_,
                                      SEXP prior_prec1_type_,
                                      SEXP prior_prec1_shape_,
                                      SEXP prior_prec1_rate_,
                                      SEXP prior_prec1_scale_,
                                      SEXP prior_prec1_df_,
                                      SEXP lag_update_,
                                      SEXP max_step_size_,
                                      SEXP base_adaptation_rate_,
                                      SEXP decay_exponent_,
                                      SEXP target_acceptance_,
                                      SEXP min_deviation_threshold_,
                                      SEXP return_log_sigma_,
                                      SEXP return_accept_prop_,
                                      SEXP init_,
                                      SEXP verbose_,
                                      SEXP bar_width_);

/**
 * @brief Gibbs sampler for local-level Bernoulli dynamic model with probit link
 *
 * @details Complete Gibbs MCMC for Bernoulli observations with probit link.
 *          Uses Albert-Chib latent variable augmentation for efficient sampling.
 *
 *          Model: y_t ~ Bernoulli(Phi(theta_{t,1}))
 *          State: theta_{t,1} = theta_{t-1,1} + u_{t,1}
 *
 * @param y_                  Observed Bernoulli outcomes [n] (0 or 1)
 * @param burnin_             Burn-in iterations (discarded)
 * @param thinning_           Thinning interval
 * @param n_draws_            Number of retained samples
 * @param prior_theta01_mean_ Prior mean for theta_{0,1}
 * @param prior_theta01_prec_ Prior precision for theta_{0,1}
 * @param prior_prec1_shape_  Gamma shape for 1/W_1
 * @param prior_prec1_rate_   Gamma rate for 1/W_1
 * @param verbose_            Flag for progress bar display (0 = off, non-zero = on)
 * @param bar_width_          Width of progress bar in characters (10-120 recommended)
 *
 * @return List with components:
 *         - theta_1: Matrix [n_draws * n] of state samples
 *         - theta_01: Vector [n_draws] of initial state samples
 *         - prec_theta1: Vector [n_draws] of precision samples
 *         - alpha: Matrix [n_draws * n] of probability samples
 *
 * @note Complexity: O(n_iter * n) time, O(n) space
 * @note Requires n >= 3 for stability
 * @note Acceptance rate: Always 1.0 (Gibbs sampling)
 * @note Each y[t] must be exactly 0 or 1
 *
 * @see Albert & Chib (1993), JASA
 * @see generate_alpha_probit_bernoulli_locallevel
 * @see generate_precision_theta_p
 * @see generate_theta_01_locallevel
 */
SEXP C_MCMC_probit_bernoulli_locallevel(SEXP y_,
                                        SEXP burnin_,
                                        SEXP thinning_,
                                        SEXP n_draws_,
                                        SEXP prior_theta01_mean_,
                                        SEXP prior_theta01_prec_,
                                        SEXP prior_prec1_type_,
                                        SEXP prior_prec1_shape_,
                                        SEXP prior_prec1_rate_,
                                        SEXP prior_prec1_scale_,
                                        SEXP prior_prec1_df_,
                                        SEXP verbose_,
                                        SEXP bar_width_);

#endif /* MCMC_BINOMIAL_LOCALLEVEL_H */
