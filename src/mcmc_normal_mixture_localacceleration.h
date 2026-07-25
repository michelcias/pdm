/**
 * @file mcmc_normal_mixture_localacceleration.h
 * @brief MCMC sampling for Gaussian mixture models with local-acceleration weights
 * @author Michel H. Montoril
 * @date 2026-07-25
 * @version 1.1
 *
 * @details Declares the Gibbs sampler for a two-component Gaussian mixture model whose
 *          mixture weights evolve according to a local-acceleration polynomial dynamic.
 *          The implementation mirrors the local-trend sampler while adding the
 *          acceleration component (theta_3) together with its innovation precision and
 *          initial state. Both logit and probit link functions are supported via the
 *          same adaptive Metropolis-Hastings and Albert-Chib routines used elsewhere in
 *          the package.
 *
 *          **Model specification:**
 *          Observation:   y_t | z_t, mu, phi ~ N(z_t * mu_2 + (1 - z_t) * mu_1,
 *                                              [z_t * phi_2 + (1 - z_t) * phi_1]^{-1})
 *          Indicators:    z_t | alpha_t ~ Bernoulli(alpha_t)
 *          Link options:  alpha_t = logit^{-1}(theta_{t,1})  ("logit")
 *                         alpha_t = Phi(theta_{t,1})          ("probit")
 *
 *          **State equations (local acceleration):**
 *          theta_{t,1} = theta_{t-1,1} + theta_{t-1,2} + u_{t,1},  u_{t,1} ~ N(0, W_1)
 *          theta_{t,2} = theta_{t-1,2} + theta_{t-1,3} + u_{t,2},  u_{t,2} ~ N(0, W_2)
 *          theta_{t,3} = theta_{t-1,3} + u_{t,3},                  u_{t,3} ~ N(0, W_3)
 *
 *          **Gibbs sampling steps per iteration:**
 *          1.  (mu_1, phi_1, mu_2, phi_2) | y, z
 *          2.  z | y, alpha, mu, phi
 *          3.  theta_3 | theta_2, theta_{0,3}, W_2, W_3
 *          4.  1/W_3 | theta_3, theta_{0,3}
 *          5.  theta_{0,3} | theta_2, theta_3, theta_{0,2}, W_2, W_3
 *          6.  theta_2 | theta_1, theta_3, theta_{0,2}, theta_{0,3}, W_1, W_2
 *          7.  1/W_2 | theta_2, theta_3, theta_{0,2}, theta_{0,3}
 *          8.  theta_{0,2} | theta_1, theta_2, theta_{0,1}, theta_{0,3}, W_1, W_2
 *          9.  theta_1, alpha | z, theta_2, theta_{0,1}, theta_{0,2}, W_1
 *          10. 1/W_1 | theta_1, theta_2, theta_{0,1}, theta_{0,2}
 *          11. theta_{0,1} | theta_1, theta_{0,2}, W_1
 *
 *          Each iteration requires only O(n) temporary storage thanks to the
 *          current/previous buffer strategy used throughout the package.
 */

#ifndef MCMC_NORMAL_MIXTURE_LOCALACCELERATION_H
#define MCMC_NORMAL_MIXTURE_LOCALACCELERATION_H

#include <R.h>
#include <Rinternals.h>

/**
 * @brief Gibbs sampler for Gaussian mixture model with local-acceleration weights
 *
 * @param y_                       Numeric vector [n] of observations.
 * @param link_                    Character string: "logit" or "probit".
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
 * @param prior_theta02_mean_      Double scalar, prior mean for theta_{0,2}.
 * @param prior_theta02_prec_      Double scalar, prior precision for theta_{0,2}.
 * @param prior_theta03_mean_      Double scalar, prior mean for theta_{0,3}.
 * @param prior_theta03_prec_      Double scalar, prior precision for theta_{0,3}.
 * @param prior_prec1_type_        Integer, prior kind on 1/W_1 (0 = Gamma, 1 = Half-t).
 * @param prior_prec1_shape_       Double scalar, Gamma shape for 1/W_1 (Gamma kind).
 * @param prior_prec1_rate_        Double scalar, Gamma rate for 1/W_1 (Gamma kind).
 * @param prior_prec1_scale_       Double scalar, Half-t scale A_1 > 0 for 1/W_1 (Half-t kind).
 * @param prior_prec1_df_          Double scalar, Half-t df nu_1 > 0 for 1/W_1 (Half-t kind; 1 = Half-Cauchy).
 * @param prior_prec2_type_        Integer, prior kind on 1/W_2 (0 = Gamma, 1 = Half-t).
 * @param prior_prec2_shape_       Double scalar, Gamma shape for 1/W_2 (Gamma kind).
 * @param prior_prec2_rate_        Double scalar, Gamma rate for 1/W_2 (Gamma kind).
 * @param prior_prec2_scale_       Double scalar, Half-t scale A_2 > 0 for 1/W_2 (Half-t kind).
 * @param prior_prec2_df_          Double scalar, Half-t df nu_2 > 0 for 1/W_2 (Half-t kind; 1 = Half-Cauchy).
 * @param prior_prec3_type_        Integer, prior kind on 1/W_3 (0 = Gamma, 1 = Half-t).
 * @param prior_prec3_shape_       Double scalar, Gamma shape for 1/W_3 (Gamma kind).
 * @param prior_prec3_rate_        Double scalar, Gamma rate for 1/W_3 (Gamma kind).
 * @param prior_prec3_scale_       Double scalar, Half-t scale A_3 > 0 for 1/W_3 (Half-t kind).
 * @param prior_prec3_df_          Double scalar, Half-t df nu_3 > 0 for 1/W_3 (Half-t kind; 1 = Half-Cauchy).
 * @param lag_update_              Integer scalar, adaptation frequency (logit only).
 * @param max_step_size_           Double scalar, max proposal step (logit only).
 * @param base_adaptation_rate_    Double scalar, base adaptation rate (logit only).
 * @param decay_exponent_          Double scalar, adaptation decay exponent (logit only).
 * @param target_acceptance_       Double scalar, target acceptance rate (logit only).
 * @param min_deviation_threshold_ Double scalar, adaptation trigger threshold (logit only).
 * @param return_log_sigma_        Logical, return log_sigma diagnostics (logit only).
 * @param return_accept_prop_      Logical, return accept_prop diagnostics (logit only).
 * @param verbose_                 Logical, display progress bar (0 = FALSE, 1 = TRUE).
 * @param bar_width_               Integer, progress bar width in characters (10-120).
 *
 * @return R list with components:
 *         - mu_1, prec_1, mu_2, prec_2 (component parameters)
 *         - theta_1, theta_2, theta_3 (state trajectories)
 *         - theta_01, theta_02, theta_03 (initial states)
 *         - prec_theta1, prec_theta2, prec_theta3 (state precisions)
 *         - alpha, z (dynamic weights and indicators)
 *         - log_sigma, accept_prop (conditionally for logit link)
 */
SEXP C_MCMC_normal_mixture_localacceleration(SEXP y_,
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
                                             SEXP prior_theta02_mean_,
                                             SEXP prior_theta02_prec_,
                                             SEXP prior_theta03_mean_,
                                             SEXP prior_theta03_prec_,
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
                                             SEXP prior_prec3_type_,
                                             SEXP prior_prec3_shape_,
                                             SEXP prior_prec3_rate_,
                                             SEXP prior_prec3_scale_,
                                             SEXP prior_prec3_df_,
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

#endif /* MCMC_NORMAL_MIXTURE_LOCALACCELERATION_H */
