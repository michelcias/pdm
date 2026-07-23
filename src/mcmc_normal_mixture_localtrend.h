/**
 * @file mcmc_normal_mixture_localtrend.h
 * @brief MCMC sampling for Gaussian mixture models with dynamic mixture weights
 * @author Michel H. Montoril
 * @date 2025-10-22
 * @version 1.0
 *
 * @details This file declares the complete Gibbs sampler for Bayesian estimation of
 *          two-component Gaussian mixture models with time-varying mixture weights
 *          following a local-trend dynamic structure.
 *
 *          The implementation features:
 *          - Memory-efficient O(n) temporary storage using current/previous buffers
 *          - Support for both logit and probit link functions
 *          - Conditional alpha computation for performance optimization
 *          - Configurable adaptive tuning for Metropolis-Hastings (logit only)
 *          - Conjugate posterior updates for all parameters
 *          - Label switching constraint enforcement (mu_1 < mu_2)
 *
 *          **Model specification:**
 *          Observation equation: y_t | z_t, mu, phi ~ N(z_t * mu_2 + (1-z_t) * mu_1,
 *                                                        [z_t * phi_2 + (1-z_t) * phi_1]^{-1})
 *          Latent indicators: z_t | alpha_t ~ Bernoulli(alpha_t)
 *
 *          **Link functions:**
 *          Logit:  alpha_t = logit^{-1}(theta_{t,1})
 *          Probit: alpha_t = Phi(theta_{t,1})
 *
 *          **State equations:**
 *          theta_{t,1} = theta_{t-1,1} + theta_{t-1,2} + u_{t,1}, u_{t,1} ~ N(0, W_1)
 *          theta_{t,2} = theta_{t-1,2} + u_{t,2},                 u_{t,2} ~ N(0, W_2)
 *
 *          **Gibbs sampling sequence per iteration:**
 *          1. (mu_1, phi_1, mu_2, phi_2) | y, z
 *          2. z | y, alpha, mu, phi
 *          3. theta_2 | theta_1, theta_{0,2}, W_2
 *          4. 1/W_2 | theta_2, theta_{0,2}
 *          5. theta_{0,2} | theta_1, theta_2, theta_{0,1}, W_1, W_2
 *          6. theta_1, alpha | z, theta_2, theta_{0,1}, W_1
 *          7. 1/W_1 | theta_1, theta_{0,1}, theta_{0,2}
 *          8. theta_{0,1} | theta_1, theta_{0,2}, W_1
 *
 *          **Link-specific methods:**
 *          Logit:  Component-wise Metropolis-Hastings with adaptive tuning
 *          Probit: Gibbs sampling via Albert-Chib data augmentation
 */

#ifndef MCMC_NORMAL_MIXTURE_LOCALTREND_H
#define MCMC_NORMAL_MIXTURE_LOCALTREND_H

#include <R.h>
#include <Rinternals.h>

/**
 * @brief Gibbs sampler for Gaussian mixture model with dynamic weights and local trend
 *
 * @details Implements a complete Gibbs MCMC algorithm for the two-component Gaussian
 *          mixture model with time-varying mixture weights following local trend dynamics.
 *          Supports both logit and probit link functions.
 *
 *          **Observation equation:**
 *          y_t | z_t, mu, phi ~ N(z_t * mu_2 + (1-z_t) * mu_1,
 *                                  [z_t * phi_2 + (1-z_t) * phi_1]^{-1})
 *          z_t | alpha_t ~ Bernoulli(alpha_t)
 *
 *          **State equations:**
 *          theta_{t,1} = theta_{t-1,1} + theta_{t-1,2} + u_{t,1}, u_{t,1} ~ N(0, W_1)
 *          theta_{t,2} = theta_{t-1,2} + u_{t,2},                 u_{t,2} ~ N(0, W_2)
 *
 *          **Link functions:**
 *          Logit:  alpha_t = exp(theta_{t,1}) / [1 + exp(theta_{t,1})]
 *                  Uses component-wise Metropolis-Hastings with adaptive tuning
 *                  Returns optional diagnostics: log_sigma, accept_prop
 *
 *          Probit: alpha_t = Phi(theta_{t,1}) where Phi is standard normal CDF
 *                  Uses Albert-Chib data augmentation (always accepts)
 *                  No adaptation parameters needed
 *
 *          **Optimizations:**
 *          - Memory-efficient current/previous iteration buffers (O(n) storage)
 *          - Conditional alpha computation (skip during burn-in/thinning)
 *          - Link-specific buffer allocation (only allocate what's needed)
 *          - Scalar parameter passing to avoid array indexing
 *
 *          **Label switching:**
 *          Enforces mu_1 < mu_2 constraint via component swapping
 *
 *          **Iteration count:**
 *          Total iterations = burnin + (n_chain - 1) × thinning + 1
 *
 * @param y_                          Numeric vector [n] of observations.
 * @param link_                       Character string: "logit" or "probit".
 * @param burnin_                     Integer scalar, number of burn-in iterations.
 * @param thinning_                   Integer scalar, thinning interval.
 * @param n_chain_                    Integer scalar, number of retained samples.
 * @param prior_mu01_mean_            Double scalar, prior mean for mu_1.
 * @param prior_mu01_prec_            Double scalar, prior precision for mu_1.
 * @param prior_prec01_type_          Integer, prior kind on phi_1 (0 = Gamma, 1 = Half-t).
 * @param prior_prec01_shape_         Double scalar, Gamma shape for phi_1 (Gamma kind).
 * @param prior_prec01_rate_          Double scalar, Gamma rate for phi_1 (Gamma kind).
 * @param prior_prec01_scale_         Double scalar, Half-t scale A > 0 for phi_1 (Half-t kind).
 * @param prior_prec01_df_            Double scalar, Half-t df > 0 for phi_1 (Half-t kind; 1 = Half-Cauchy).
 * @param prior_mu02_mean_            Double scalar, prior mean for mu_2.
 * @param prior_mu02_prec_            Double scalar, prior precision for mu_2.
 * @param prior_prec02_type_          Integer, prior kind on phi_2 (0 = Gamma, 1 = Half-t).
 * @param prior_prec02_shape_         Double scalar, Gamma shape for phi_2 (Gamma kind).
 * @param prior_prec02_rate_          Double scalar, Gamma rate for phi_2 (Gamma kind).
 * @param prior_prec02_scale_         Double scalar, Half-t scale A > 0 for phi_2 (Half-t kind).
 * @param prior_prec02_df_            Double scalar, Half-t df > 0 for phi_2 (Half-t kind; 1 = Half-Cauchy).
 * @param prior_theta01_mean_         Double scalar, prior mean for theta_{0,1}.
 * @param prior_theta01_prec_         Double scalar, prior precision for theta_{0,1}.
 * @param prior_theta02_mean_         Double scalar, prior mean for theta_{0,2}.
 * @param prior_theta02_prec_         Double scalar, prior precision for theta_{0,2}.
 * @param prior_prec1_type_           Integer, prior kind on 1/W_1 (0 = Gamma, 1 = Half-t).
 * @param prior_prec1_shape_          Double scalar, Gamma shape for 1/W_1 (Gamma kind).
 * @param prior_prec1_rate_           Double scalar, Gamma rate for 1/W_1 (Gamma kind).
 * @param prior_prec1_scale_          Double scalar, Half-t scale A_1 > 0 for 1/W_1 (Half-t kind).
 * @param prior_prec1_df_             Double scalar, Half-t df nu_1 > 0 for 1/W_1 (Half-t kind; 1 = Half-Cauchy).
 * @param prior_prec2_type_           Integer, prior kind on 1/W_2 (0 = Gamma, 1 = Half-t).
 * @param prior_prec2_shape_          Double scalar, Gamma shape for 1/W_2 (Gamma kind).
 * @param prior_prec2_rate_           Double scalar, Gamma rate for 1/W_2 (Gamma kind).
 * @param prior_prec2_scale_          Double scalar, Half-t scale A_2 > 0 for 1/W_2 (Half-t kind).
 * @param prior_prec2_df_             Double scalar, Half-t df nu_2 > 0 for 1/W_2 (Half-t kind; 1 = Half-Cauchy).
 * @param lag_update_                 Integer scalar, adaptation frequency (logit only).
 * @param max_step_size_              Double scalar, max proposal step (logit only).
 * @param base_adaptation_rate_       Double scalar, base adaptation rate (logit only).
 * @param decay_exponent_             Double scalar, adaptation decay (logit only).
 * @param target_acceptance_          Double scalar, target acceptance rate (logit only).
 * @param min_deviation_threshold_    Double scalar, adaptation trigger threshold (logit only).
 * @param return_log_sigma_           Logical, return log_sigma diagnostics (logit only).
 * @param return_accept_prop_         Logical, return accept_prop diagnostics (logit only).
 * @param verbose_                    Logical, display progress bar (0 = FALSE, 1 = TRUE).
 * @param bar_width_                  Integer, progress bar width in characters (10-120).
 *
 * @return R list with components:
 *         **Always returned:**
 *         - mu_1:        Vector [n_chain] of component 1 mean samples
 *         - prec_1:      Vector [n_chain] of component 1 precision samples
 *         - mu_2:        Vector [n_chain] of component 2 mean samples
 *         - prec_2:      Vector [n_chain] of component 2 precision samples
 *         - theta_1:     Matrix [n_chain × n] of level state samples
 *         - theta_2:     Matrix [n_chain × n] of trend state samples
 *         - theta_01:    Vector [n_chain] of initial level state samples
 *         - theta_02:    Vector [n_chain] of initial trend state samples
 *         - prec_theta1: Vector [n_chain] of level precision samples
 *         - prec_theta2: Vector [n_chain] of trend precision samples
 *         - alpha:       Matrix [n_chain × n] of mixture weight samples
 *         - z:           Matrix [n_chain × n] of latent indicator samples
 *
 *         **Conditionally returned (logit only):**
 *         - log_sigma:   Matrix [n_chain × n] of proposal scales
 *         - accept_prop: Matrix [n_chain × n] of acceptance proportions
 *
 * @note Computational complexity: O(n_iter × n) time, O(n) space.
 * @note Memory requirements: O(n) temporary storage for efficient buffer management.
 * @note RNG management: Proper GetRNGstate()/PutRNGstate() bracket for R integration.
 * @note Minimum sample size: n >= 3 for numerical stability.
 * @note Link functions: Must be exactly "logit" or "probit" (case-sensitive).
 * @note Logit-specific parameters: Ignored when link="probit".
 * @note Diagnostic outputs: NULL when link="probit".
 *
 * @warning Minimum sample size n >= 3 enforced for numerical stability.
 * @warning Integer overflow protection: n <= INT_MAX.
 * @warning Memory allocation failures will terminate R session.
 * @warning link must be exactly "logit" or "probit" (case-sensitive).
 *
 * @see conditional_mixture_normal_parameters_k2
 * @see conditional_mixture_normal_indicators_k2
 * @see generate_alpha_logit_binomial
 * @see generate_alpha_probit_bernoulli
 * @see generate_theta_p
 * @see generate_precision_theta_p
 * @see generate_theta_0p
 * @see generate_precision_theta_k
 * @see generate_theta_01
 */
SEXP C_MCMC_normal_mixture_localtrend(SEXP y_,
                                      SEXP link_,
                                      SEXP burnin_,
                                      SEXP thinning_,
                                      SEXP n_chain_,
                                      SEXP prior_mu01_mean_,
                                      SEXP prior_mu01_prec_,
                                      SEXP prior_prec01_type_,
                                      SEXP prior_prec01_shape_,
                                      SEXP prior_prec01_rate_,
                                      SEXP prior_prec01_scale_,
                                      SEXP prior_prec01_df_,
                                      SEXP prior_mu02_mean_,
                                      SEXP prior_mu02_prec_,
                                      SEXP prior_prec02_type_,
                                      SEXP prior_prec02_shape_,
                                      SEXP prior_prec02_rate_,
                                      SEXP prior_prec02_scale_,
                                      SEXP prior_prec02_df_,
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

#endif /* MCMC_NORMAL_MIXTURE_LOCALTREND_H */
