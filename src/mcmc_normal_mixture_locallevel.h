/**
 * @file mcmc_normal_mixture_locallevel.h
 * @brief MCMC sampling for Gaussian mixture models with dynamic mixture weights
 *        following a local-level structure
 * @author Michel H. Montoril
 * @date 2026-07-26
 * @version 1.2
 *
 * @details Declares the complete Gibbs sampler for Bayesian estimation of
 *          two-component Gaussian mixture models with time-varying mixture
 *          weights evolving as a local-level stochastic process. The
 *          implementation mirrors the local-trend variant while simplifying
 *          the state evolution to a single random-walk component.
 *
 *          The sampler supports both logit and probit link functions for the
 *          dynamic mixture weights and leverages the same memory-efficient
 *          buffer strategy used across the pdm package.
 *
 *          **Observation equation:**
 *          y_t | z_t, mu, phi ~ N(z_t * mu_2 + (1 - z_t) * mu_1,
 *                                  [z_t * phi_2 + (1 - z_t) * phi_1]^{-1})
 *          z_t | alpha_t ~ Bernoulli(alpha_t)
 *
 *          **State equation (local level):**
 *          theta_{t,1} = theta_{t-1,1} + u_{t,1},  u_{t,1} ~ N(0, W_1)
 *
 *          **Link functions:**
 *          - Logit:  alpha_t = logit^{-1}(theta_{t,1})
 *          - Probit: alpha_t = Phi(theta_{t,1})
 *
 *          **Gibbs sampling sequence per iteration:**
 *          1. (mu_1, phi_1, mu_2, phi_2) | y, z
 *          2. z | y, alpha, mu, phi
 *          3. theta_1, alpha | z, theta_{0,1}, W_1
 *             - Logit: Component-wise MH with adaptive tuning
 *             - Probit: Gibbs sampling via Albert-Chib augmentation
 *          4. 1/W_1 | theta_1, theta_{0,1}
 *          5. theta_{0,1} | theta_1, W_1
 */

#ifndef MCMC_NORMAL_MIXTURE_LOCALLEVEL_H
#define MCMC_NORMAL_MIXTURE_LOCALLEVEL_H

#include <R.h>
#include <Rinternals.h>

/**
 * @brief Gibbs sampler for Gaussian mixture model with dynamic weights and
 *        local-level evolution
 *
 * @details Implements a complete Gibbs MCMC algorithm for the two-component
 *          Gaussian mixture model with time-varying mixture weights following
 *          a local-level dynamic structure. Supports both logit and probit
 *          link functions through a single entry point.
 *
 *          **Observation equation:**
 *          y_t | z_t, mu, phi ~ N(z_t * mu_2 + (1 - z_t) * mu_1,
 *                                  [z_t * phi_2 + (1 - z_t) * phi_1]^{-1})
 *          z_t | alpha_t ~ Bernoulli(alpha_t)
 *
 *          **State equation (local level):**
 *          theta_{t,1} = theta_{t-1,1} + u_{t,1},  u_{t,1} ~ N(0, W_1)
 *
 *          **Link functions:**
 *          - "logit":  alpha_t = exp(theta_{t,1}) / [1 + exp(theta_{t,1})]
 *                      Uses component-wise Metropolis-Hastings with adaptive
 *                      tuning and optional diagnostics (log_sigma,
 *                      accept_prop).
 *          - "probit": alpha_t = Phi(theta_{t,1})
 *                      Uses Albert-Chib data augmentation (no adaptation
 *                      parameters required).
 *
 *          **Optimizations inherited from the local-trend implementation:**
 *          - Memory-efficient current/previous iteration buffers (O(n) storage)
 *          - Conditional alpha computation to skip expensive transformations
 *            during burn-in/thinning (controlled via compute_alpha flag)
 *          - Scalar parameter passing to minimize indexing overhead
 *          - Automatic enforcement of label-switching constraint (mu_1 < mu_2)
 *
 *          **Iteration count:**
 *          Total iterations = burnin + (n_draws - 1) × thinning + 1
 *
 * @param y_                       Numeric vector [n] of observations.
 * @param link_                    Character scalar: "logit" or "probit".
 * @param burnin_                  Integer scalar, number of burn-in iterations.
 * @param thinning_                Integer scalar, thinning interval.
 * @param n_draws_                 Integer scalar, number of retained samples.
 * @param prior_mu01_mean_         Double scalar, prior mean for mu_1.
 * @param prior_mu01_prec_         Double scalar, prior precision for mu_1.
 * @param prior_prec01_type_       Integer, prior kind on phi_1 (0 = Gamma, 1 = Half-t).
 * @param prior_prec01_shape_      Double scalar, Gamma shape for phi_1 (Gamma kind).
 * @param prior_prec01_rate_       Double scalar, Gamma rate for phi_1 (Gamma kind).
 * @param prior_prec01_scale_      Double scalar, Half-t scale A > 0 for phi_1 (Half-t kind).
 * @param prior_prec01_df_         Double scalar, Half-t df > 0 for phi_1 (Half-t kind; 1 = Half-Cauchy).
 * @param prior_mu02_mean_         Double scalar, prior mean for mu_2.
 * @param prior_mu02_prec_         Double scalar, prior precision for mu_2.
 * @param prior_prec02_type_       Integer, prior kind on phi_2 (0 = Gamma, 1 = Half-t).
 * @param prior_prec02_shape_      Double scalar, Gamma shape for phi_2 (Gamma kind).
 * @param prior_prec02_rate_       Double scalar, Gamma rate for phi_2 (Gamma kind).
 * @param prior_prec02_scale_      Double scalar, Half-t scale A > 0 for phi_2 (Half-t kind).
 * @param prior_prec02_df_         Double scalar, Half-t df > 0 for phi_2 (Half-t kind; 1 = Half-Cauchy).
 * @param prior_theta01_mean_      Double scalar, prior mean for theta_{0,1}.
 * @param prior_theta01_prec_      Double scalar, prior precision for theta_{0,1}.
 * @param prior_prec1_type_        Integer, prior kind on 1/W_1 (0 = Gamma, 1 = Half-t).
 * @param prior_prec1_shape_       Double scalar, Gamma shape for 1/W_1 (Gamma kind).
 * @param prior_prec1_rate_        Double scalar, Gamma rate for 1/W_1 (Gamma kind).
 * @param prior_prec1_scale_       Double scalar, Half-t scale A_1 > 0 for 1/W_1 (Half-t kind).
 * @param prior_prec1_df_          Double scalar, Half-t df nu_1 > 0 for 1/W_1 (Half-t kind; 1 = Half-Cauchy).
 * @param lag_update_              Integer scalar, adaptation frequency (logit only).
 * @param max_step_size_           Double scalar, maximum proposal step (logit only).
 * @param base_adaptation_rate_    Double scalar, base adaptation rate (logit only).
 * @param decay_exponent_          Double scalar, adaptation decay exponent (logit only).
 * @param target_acceptance_       Double scalar, target acceptance rate (logit only).
 * @param min_deviation_threshold_ Double scalar, adaptation trigger (logit only).
 * @param return_log_sigma_        Logical, return log_sigma diagnostics (logit only).
 * @param return_accept_prop_      Logical, return accept_prop diagnostics (logit only).
 * @param init_                    Double vector [9] of starting values, resolved in R by
 *                                 resolve_init(): mu_1 and mu_2, then theta_{0,1}, then the
 *                                 component precisions phi_1 and phi_2, then 1/W_1, then one
 *                                 Half-t auxiliary per precision in that same order (0 under a Gamma
 *                                 prior, never read). R also enforces mu_1 <= mu_2.
 * @param verbose_                 Logical, display progress bar (0 = FALSE, 1 = TRUE).
 * @param bar_width_               Integer, progress bar width (10-120 recommended).
 *
 * @return R list with components:
 *         **Always returned:**
 *         - mu_1:        Vector [n_draws] of component 1 mean samples
 *         - prec_1:      Vector [n_draws] of component 1 precision samples
 *         - mu_2:        Vector [n_draws] of component 2 mean samples
 *         - prec_2:      Vector [n_draws] of component 2 precision samples
 *         - theta_1:     Matrix [n_draws × n] of level state samples
 *         - theta_01:    Vector [n_draws] of initial level state samples
 *         - prec_theta1: Vector [n_draws] of level innovation precision samples
 *         - alpha:       Matrix [n_draws × n] of mixture weight samples
 *         - z:           Matrix [n_draws × n] of latent indicator samples
 *
 *         **Conditionally returned (logit only):**
 *         - log_sigma:   Matrix [n_draws × n] of proposal log-scales
 *         - accept_prop: Matrix [n_draws × n] of acceptance proportions
 *
 * @note Computational complexity: O(n_iter × n) time, O(n) space.
 * @note Requires n >= 3 for numerical stability (consistent with dynamic GLM samplers).
 * @note Logit-specific parameters are ignored when link = "probit".
 * @note Diagnostic outputs (log_sigma, accept_prop) are NULL when link = "probit".
 *
 * @warning Sample size must satisfy 3 <= n <= INT_MAX.
 * @warning link must be exactly "logit" or "probit" (case-sensitive).
 * @warning Memory allocation failures terminate the R session via R's allocator.
 *
 * @see conditional_mixture_normal_parameters_k2
 * @see conditional_mixture_normal_indicators_k2
 * @see generate_alpha_logit_binomial_locallevel
 * @see generate_alpha_probit_bernoulli_locallevel
 * @see generate_precision_theta_p
 * @see generate_theta_01_locallevel
 */
SEXP C_MCMC_normal_mixture_locallevel(SEXP y_,
                                      SEXP link_,
                                      SEXP burnin_,
                                      SEXP thinning_,
                                      SEXP n_draws_,
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

#endif /* MCMC_NORMAL_MIXTURE_LOCALLEVEL_H */
