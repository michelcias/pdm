/**
 * @file mcmc_poisson_mixture_locallevel.h
 * @brief MCMC sampling for Poisson mixture models with dynamic mixture weights
 *        following a local-level structure
 * @author Michel H. Montoril
 * @date 2026-08-15
 * @version 1.0
 *
 * @details Declares the complete Gibbs sampler for Bayesian estimation of
 *          two-component Poisson mixture models with time-varying mixture
 *          weights evolving as a local-level stochastic process. The driver
 *          mirrors the Gaussian mixture sampler of the same order, differing
 *          only in the observation model: the component step draws two rates
 *          from conjugate Gammas instead of two mean/precision pairs, and the
 *          indicator step weighs Poisson densities instead of Normal ones.
 *
 *          **Observation equation:**
 *          y_t | z_t, lambda ~ Poisson(z_t * lambda_2 + (1 - z_t) * lambda_1)
 *          z_t | alpha_t     ~ Bernoulli(alpha_t)
 *
 *          **State equation (local level):**
 *          theta_{t,1} = theta_{t-1,1} + u_{t,1},  u_{t,1} ~ N(0, W_1)
 *
 *          **Link functions:**
 *          - Logit:  alpha_t = logit^{-1}(theta_{t,1})
 *          - Probit: alpha_t = Phi(theta_{t,1})
 *
 *          Note which quantity the link carries. In `mcmc_poisson_*` the latent
 *          state drives the *rate*, through a log link. Here it drives the
 *          *mixture weight*, exactly as in the Gaussian mixture, and the rates
 *          are two constants. The two families answer different questions and
 *          neither is a special case of the other.
 *
 *          **Gibbs sampling sequence per iteration:**
 *          1. (lambda_1, lambda_2) | y, z
 *          2. z | y, alpha, lambda
 *          3. theta_1, alpha | z, theta_{0,1}, W_1
 *             - Logit: Component-wise MH with adaptive tuning
 *             - Probit: Gibbs sampling via Albert-Chib augmentation
 *          4. 1/W_1 | theta_1, theta_{0,1}
 *          5. theta_{0,1} | theta_1, W_1
 */

#ifndef MCMC_POISSON_MIXTURE_LOCALLEVEL_H
#define MCMC_POISSON_MIXTURE_LOCALLEVEL_H

#include <R.h>
#include <Rinternals.h>

/**
 * @brief Gibbs sampler for a Poisson mixture model with dynamic weights and
 *        local-level evolution
 *
 * @details Implements the complete Gibbs MCMC algorithm for the two-component
 *          Poisson mixture model with time-varying mixture weights following a
 *          local-level dynamic structure. Both link functions are served through
 *          this single entry point.
 *
 *          **Optimizations shared with the Gaussian mixture driver:**
 *          - Memory-efficient current/previous iteration buffers (O(n) storage)
 *          - Scalar parameter passing to minimize indexing overhead
 *          - Automatic enforcement of the label constraint (lambda_1 < lambda_2)
 *
 *          **Iteration count:**
 *          Total iterations = burnin + (n_draws - 1) * thinning + 1
 *
 * @param y_                       Numeric vector [n] of observed counts.
 * @param link_                    Character scalar: "logit" or "probit".
 * @param burnin_                  Integer scalar, number of burn-in iterations.
 * @param thinning_                Integer scalar, thinning interval.
 * @param n_draws_                 Integer scalar, number of retained samples.
 * @param prior_lambda01_shape_    Double scalar, Gamma shape a_01 for lambda_1.
 * @param prior_lambda01_rate_     Double scalar, Gamma rate b_01 for lambda_1.
 * @param prior_lambda02_shape_    Double scalar, Gamma shape a_02 for lambda_2.
 * @param prior_lambda02_rate_     Double scalar, Gamma rate b_02 for lambda_2.
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
 * @param init_                    Double vector [7] of starting values, resolved in R by
 *                                 resolve_init(): theta_{0,1}, then the component rates
 *                                 lambda_1 and lambda_2, then 1/W_1, then one Half-t
 *                                 auxiliary per entry of that precision block (the two
 *                                 rate slots are 0 and never read; only the W_1 auxiliary
 *                                 carries a value, and only under a Half-t prior). R also
 *                                 enforces lambda_1 <= lambda_2.
 * @param verbose_                 Logical, display progress bar (0 = FALSE, 1 = TRUE).
 * @param bar_width_               Integer, progress bar width (10-120 recommended).
 *
 * @return R list with components:
 *         **Always returned:**
 *         - lambda_1:    Vector [n_draws] of component 1 rate samples
 *         - lambda_2:    Vector [n_draws] of component 2 rate samples
 *         - theta_1:     Matrix [n_draws x n] of level state samples
 *         - theta_01:    Vector [n_draws] of initial level state samples
 *         - prec_theta1: Vector [n_draws] of level innovation precision samples
 *         - alpha:       Matrix [n_draws x n] of mixture weight samples
 *         - z:           Matrix [n_draws x n] of latent indicator samples
 *
 *         **Conditionally returned (logit only):**
 *         - log_sigma:   Matrix [n_draws x n] of proposal log-scales
 *         - accept_prop: Matrix [n_draws x n] of acceptance proportions
 *
 * @note Computational complexity: O(n_iter * n) time, O(n) space.
 * @note Requires n >= 3, consistent with the other dynamic GLM samplers.
 * @note Logit-specific parameters are ignored when link = "probit".
 * @note Diagnostic outputs (log_sigma, accept_prop) are NULL when link = "probit".
 *
 * @warning Sample size must satisfy 3 <= n <= INT_MAX.
 * @warning link must be exactly "logit" or "probit" (case-sensitive).
 * @warning Counts are validated in R; this driver assumes y is non-negative.
 *
 * @see conditional_mixture_poisson_parameters_k2
 * @see conditional_mixture_poisson_indicators_k2
 * @see generate_alpha_logit_binomial_locallevel
 * @see generate_alpha_probit_bernoulli_locallevel
 * @see generate_theta_01_locallevel
 */
SEXP C_MCMC_poisson_mixture_locallevel(SEXP y_,
                                       SEXP link_,
                                       SEXP burnin_,
                                       SEXP thinning_,
                                       SEXP n_draws_,
                                       SEXP prior_lambda01_shape_,
                                       SEXP prior_lambda01_rate_,
                                       SEXP prior_lambda02_shape_,
                                       SEXP prior_lambda02_rate_,
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

#endif /* MCMC_POISSON_MIXTURE_LOCALLEVEL_H */
