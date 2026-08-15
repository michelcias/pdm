/**
 * @file mcmc_poisson_mixture_localacceleration.c
 * @brief MCMC sampling for Poisson mixture models with dynamic mixture weights
 * under a local-acceleration evolution
 * @author Michel H. Montoril
 * @date 2026-08-15
 * @version 1.0
 *
 * @details Implements the complete Gibbs sampler for Bayesian estimation of
 *          two-component Poisson mixture models with time-varying mixture
 *          weights that follow a local-acceleration stochastic process (level +
 *          trend + acceleration). The dynamic weight machinery is shared
 *          verbatim with the Gaussian mixture sampler of the same order; only
 *          the observation model differs.
 *
 * **Supported link functions:**
 * - Logit:  Component-wise Metropolis-Hastings with adaptive tuning
 * - Probit: Gibbs sampling via Albert-Chib data augmentation
 *
 * **Poisson mixture model with dynamic weights:**
 * Observation: y_t | z_t, lambda ~ Poisson(z_t*lambda_2 + (1-z_t)*lambda_1)
 * Indicators:  z_t | alpha_t ~ Bernoulli(alpha_t)
 *
 * **Dynamic weight model (local acceleration):**
 * theta_{t,1} = theta_{t-1,1} + theta_{t-1,2} + u_{t,1},  u_{t,1} ~ N(0, W_1)
 * theta_{t,2} = theta_{t-1,2} + theta_{t-1,3} + u_{t,2},  u_{t,2} ~ N(0, W_2)
 * theta_{t,3} = theta_{t-1,3} + u_{t,3},                  u_{t,3} ~ N(0, W_3)
 * alpha_t = T^{-1}(theta_{t,1}),  T in {logit, probit}
 *
 * **Prior distributions:**
 * - lambda_k ~ Gamma(a_0k, b_0k),             k = 1, 2
 * - theta_{0,j} ~ N(mu_{0,j}, sigma^2_{0,j}), j = 1, 2, 3
 * - 1/W_j ~ Gamma(nu_j, eta_j), or a Half-t prior on sqrt(W_j) (Gelman, 2006),
 *   selected independently via the prior_prec*_type_ codes
 *
 * The component rates carry a Gamma prior and nothing else; see
 * mcmc_poisson_mixture_locallevel.c for why the Half-t option stops at the
 * state precisions.
 *
 * **Sampling sequence per iteration:**
 * 1. (lambda_1, lambda_2) | y, z -> Conjugate Gamma updates
 * 2. z | y, alpha, lambda -> Bernoulli with weighted Poisson densities
 * 3. theta_3 | theta_2, theta_{0,3}, W_2, W_3 -> Gaussian posterior (tridiagonal)
 * 4. 1/W_3 | theta_3, theta_{0,3} -> Gamma or Half-t posterior
 * 5. theta_{0,3} | theta_2, theta_3, theta_{0,2}, W_2, W_3 -> Normal posterior
 * 6. theta_2 | theta_1, theta_3, theta_{0,2}, theta_{0,3}, W_1, W_2 -> Gaussian posterior
 * 7. 1/W_2 | theta_2, theta_3, theta_{0,2}, theta_{0,3} -> Gamma or Half-t posterior
 * 8. theta_{0,2} | theta_1, theta_2, theta_{0,1}, theta_{0,3}, W_1, W_2 -> Normal posterior
 * 9. theta_1, alpha | z, theta_2, theta_{0,1}, theta_{0,2}, W_1 -> Link-specific
 * 10. 1/W_1 | theta_1, theta_2, theta_{0,1}, theta_{0,2} -> Gamma or Half-t posterior
 * 11. theta_{0,1} | theta_1, theta_{0,2}, W_1 -> Normal posterior
 */

#include <R.h>
#include <Rinternals.h>
#include <Rmath.h>
#include <string.h>  /* memcpy, strcmp */
#include "conditional_state.h"
#include "conditional_precision.h"
#include "conditional_theta0.h"
#include "conditional_mixture_poisson_parameters.h"
#include "conditional_mixture_poisson_indicators.h"
#include "generate_alpha_binomial.h"
#include "utils.h"
#include "link_guard.h"  /* ilogit_guarded, clamp_link_alpha */
#include "prec_prior_dispatch.h" /* prec_prior_t, step_prec_*, pdm_init_prec_prior */
#include "mcmc_progress_bar.h"
#include "mcmc_poisson_mixture_localacceleration.h"

/**
 * @brief Unified Gibbs sampler for a Poisson mixture model with
 *        local-acceleration weights
 *
 * @details Executes the full Gibbs sampling cycle for a two-component Poisson
 *          mixture model whose mixture weights follow a local-acceleration
 *          process. Supports both logit and probit link functions via the
 *          `link` argument.
 *
 * @param y_                       Numeric vector [n] of observed counts.
 * @param link_                    Character string: "logit" or "probit".
 * @param burnin_                  Number of burn-in iterations (discarded).
 * @param thinning_                Thinning interval for retained samples.
 * @param n_draws_                 Number of retained posterior samples.
 * @param prior_lambda01_shape_    Gamma shape a_01 for lambda_1.
 * @param prior_lambda01_rate_     Gamma rate b_01 for lambda_1.
 * @param prior_lambda02_shape_    Gamma shape a_02 for lambda_2.
 * @param prior_lambda02_rate_     Gamma rate b_02 for lambda_2.
 * @param prior_theta01_mean_      Prior mean for theta_{0,1}.
 * @param prior_theta01_prec_      Prior precision for theta_{0,1}.
 * @param prior_theta02_mean_      Prior mean for theta_{0,2}.
 * @param prior_theta02_prec_      Prior precision for theta_{0,2}.
 * @param prior_theta03_mean_      Prior mean for theta_{0,3}.
 * @param prior_theta03_prec_      Prior precision for theta_{0,3}.
 * @param prior_prec1_type_        Prior kind on 1/W_1 (0 = Gamma, 1 = Half-t).
 * @param prior_prec1_shape_       Gamma shape for 1/W_1 (Gamma kind).
 * @param prior_prec1_rate_        Gamma rate for 1/W_1 (Gamma kind).
 * @param prior_prec1_scale_       Half-t scale A_1 > 0 for 1/W_1 (Half-t kind).
 * @param prior_prec1_df_          Half-t df nu_1 > 0 for 1/W_1 (Half-t kind).
 * @param prior_prec2_type_        Prior kind on 1/W_2 (0 = Gamma, 1 = Half-t).
 * @param prior_prec2_shape_       Gamma shape for 1/W_2 (Gamma kind).
 * @param prior_prec2_rate_        Gamma rate for 1/W_2 (Gamma kind).
 * @param prior_prec2_scale_       Half-t scale A_2 > 0 for 1/W_2 (Half-t kind).
 * @param prior_prec2_df_          Half-t df nu_2 > 0 for 1/W_2 (Half-t kind).
 * @param prior_prec3_type_        Prior kind on 1/W_3 (0 = Gamma, 1 = Half-t).
 * @param prior_prec3_shape_       Gamma shape for 1/W_3 (Gamma kind).
 * @param prior_prec3_rate_        Gamma rate for 1/W_3 (Gamma kind).
 * @param prior_prec3_scale_       Half-t scale A_3 > 0 for 1/W_3 (Half-t kind).
 * @param prior_prec3_df_          Half-t df nu_3 > 0 for 1/W_3 (Half-t kind).
 * @param lag_update_              Adaptation frequency (logit only).
 * @param max_step_size_           Maximum proposal step size (logit only).
 * @param base_adaptation_rate_    Base adaptation rate (logit only).
 * @param decay_exponent_          Adaptation decay exponent (logit only).
 * @param target_acceptance_       Target acceptance proportion (logit only).
 * @param min_deviation_threshold_ Minimum deviation to trigger adaptation (logit only).
 * @param return_log_sigma_        Flag to return log_sigma diagnostics (logit only).
 * @param return_accept_prop_      Flag to return accept_prop diagnostics (logit only).
 * @param init_                    Double vector [13] of starting values, resolved in R by
 *                                 resolve_init(): the three initial states, then the two
 *                                 component rates and the three state precisions, then one
 *                                 Half-t auxiliary per entry of that second block.
 * @param verbose_                 Logical: display progress bar (0 = FALSE, 1 = TRUE).
 * @param bar_width_               Integer: width of progress bar (10-120 recommended).
 *
 * @return R list with posterior samples and optional diagnostics.
 *
 * @note Complexity: O(n_iter * n) time, O(n) space
 * @note Requires n >= 3 for numerical stability
 * @note Proper RNG state management via GetRNGstate()/PutRNGstate()
 * @note Logit-specific parameters are ignored when link="probit"
 *
 * @warning n must not exceed INT_MAX
 * @warning link must be exactly "logit" or "probit" (case-sensitive)
 *
 * @see conditional_mixture_poisson_parameters_k2
 * @see conditional_mixture_poisson_indicators_k2
 * @see generate_alpha_logit_binomial
 * @see generate_alpha_probit_bernoulli
 * @see generate_theta_p
 * @see generate_theta_k
 * @see generate_theta_0p
 * @see generate_theta_0k
 * @see generate_theta_01
 */
SEXP C_MCMC_poisson_mixture_localacceleration(SEXP y_,
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
                                              SEXP init_,
                                              SEXP verbose_,
                                              SEXP bar_width_) {

  /* ========== Parse and Validate Link Function ========== */
  const char *link = CHAR(STRING_ELT(link_, 0));

  if (strcmp(link, "logit") != 0 && strcmp(link, "probit") != 0) {
    Rf_error("C_MCMC_poisson_mixture_localacceleration: link must be 'logit' or 'probit', got '%s'",
             link);
  }

  int use_logit = (strcmp(link, "logit") == 0);

  /* ========== Parse Data Vector and Validate Length ========== */
  double   *y   = REAL(y_);
  R_xlen_t  len = LENGTH(y_);

  if (len < 3) {
    Rf_error("C_MCMC_poisson_mixture_localacceleration: sample size 'n' must be at least 3, got %lld",
             (long long) len);
  }

  if (len > INT_MAX) {
    Rf_error("C_MCMC_poisson_mixture_localacceleration: sample size too large (%lld > %d)",
             (long long) len, INT_MAX);
  }

  int n = (int) len;

  /* ========== Parse MCMC Control Parameters ========== */
  int burnin   = INTEGER(burnin_)[0];
  int thinning = INTEGER(thinning_)[0];
  int n_draws  = INTEGER(n_draws_)[0];
  int n_iter   = burnin + (n_draws - 1) * thinning + 1;

  /* ========== Parse Mixture Component Prior Hyperparameters ========== */
  /* Both component rates carry a conjugate Gamma; there is no prior kind to
   * dispatch on, unlike the Gaussian mixture's component precisions. */
  double lambda_01_shape = REAL(prior_lambda01_shape_)[0];
  double lambda_01_rate  = REAL(prior_lambda01_rate_)[0];
  double lambda_02_shape = REAL(prior_lambda02_shape_)[0];
  double lambda_02_rate  = REAL(prior_lambda02_rate_)[0];

  /* ========== Parse Dynamic State Prior Hyperparameters ========== */
  double mean_theta01 = REAL(prior_theta01_mean_)[0];
  double prec_theta01 = REAL(prior_theta01_prec_)[0];
  double mean_theta02 = REAL(prior_theta02_mean_)[0];
  double prec_theta02 = REAL(prior_theta02_prec_)[0];
  double mean_theta03 = REAL(prior_theta03_mean_)[0];
  double prec_theta03 = REAL(prior_theta03_prec_)[0];
  int          prec1_kind = asInteger(prior_prec1_type_);   /* prior on 1/W_1 */
  prec_prior_t prior_W1   = {
    .shape    = REAL(prior_prec1_shape_)[0],
    .rate     = REAL(prior_prec1_rate_)[0],
    .df       = REAL(prior_prec1_df_)[0],
    .hc_scale = REAL(prior_prec1_scale_)[0]
  };
  int          prec2_kind = asInteger(prior_prec2_type_);   /* prior on 1/W_2 */
  prec_prior_t prior_W2   = {
    .shape    = REAL(prior_prec2_shape_)[0],
    .rate     = REAL(prior_prec2_rate_)[0],
    .df       = REAL(prior_prec2_df_)[0],
    .hc_scale = REAL(prior_prec2_scale_)[0]
  };
  int          prec3_kind = asInteger(prior_prec3_type_);   /* prior on 1/W_3 */
  prec_prior_t prior_W3   = {
    .shape    = REAL(prior_prec3_shape_)[0],
    .rate     = REAL(prior_prec3_rate_)[0],
    .df       = REAL(prior_prec3_df_)[0],
    .hc_scale = REAL(prior_prec3_scale_)[0]
  };
  /* W_1 and W_2 are intermediate components (theta_k sampler), W_3 the terminal
   * random-walk component (theta_p sampler). */
  prec_thetak_step_t update_prec_W1 =
    (prec1_kind == PDM_PREC_PRIOR_HALFT) ? step_prec_thetak_halft
                                         : step_prec_thetak_gamma;
  prec_thetak_step_t update_prec_W2 =
    (prec2_kind == PDM_PREC_PRIOR_HALFT) ? step_prec_thetak_halft
                                         : step_prec_thetak_gamma;
  prec_thetap_step_t update_prec_W3 =
    (prec3_kind == PDM_PREC_PRIOR_HALFT) ? step_prec_thetap_halft
                                         : step_prec_thetap_gamma;

  /* ========== Parse Adaptation Parameters (Logit Only) ========== */
  int    lag_update              = 0;
  double max_step_size           = 0.0;
  double base_adaptation_rate    = 0.0;
  double decay_exponent          = 0.0;
  double target_acceptance       = 0.0;
  double min_deviation_threshold = 0.0;

  if (use_logit) {
    lag_update              = INTEGER(lag_update_)[0];
    max_step_size           = REAL(max_step_size_)[0];
    base_adaptation_rate    = REAL(base_adaptation_rate_)[0];
    decay_exponent          = REAL(decay_exponent_)[0];
    target_acceptance       = REAL(target_acceptance_)[0];
    min_deviation_threshold = REAL(min_deviation_threshold_)[0];
  }

  /* ========== Parse Diagnostic Output Options (Logit Only) ========== */
  int return_log_sigma   = 0;
  int return_accept_prop = 0;

  if (use_logit) {
    return_log_sigma   = LOGICAL(return_log_sigma_)[0];
    return_accept_prop = LOGICAL(return_accept_prop_)[0];
  }

  /* ========== Parse Progress Bar Parameters ========== */
  int verbose   = asLogical(verbose_);
  int bar_width = asInteger(bar_width_);

  ProgressBar pb = progress_bar_init(n_iter, bar_width, burnin, thinning, verbose);
  progress_bar_start(&pb);

  /* ========== Allocate Output Storage (Retained Samples Only) ========== */
  SEXP lambda_1_samples    = PROTECT(allocVector(REALSXP, n_draws));
  SEXP lambda_2_samples    = PROTECT(allocVector(REALSXP, n_draws));
  SEXP theta_1_samples     = PROTECT(allocMatrix(REALSXP, n_draws, n));
  SEXP theta_2_samples     = PROTECT(allocMatrix(REALSXP, n_draws, n));
  SEXP theta_3_samples     = PROTECT(allocMatrix(REALSXP, n_draws, n));
  SEXP theta_01_samples    = PROTECT(allocVector(REALSXP, n_draws));
  SEXP theta_02_samples    = PROTECT(allocVector(REALSXP, n_draws));
  SEXP theta_03_samples    = PROTECT(allocVector(REALSXP, n_draws));
  SEXP prec_theta1_samples = PROTECT(allocVector(REALSXP, n_draws));
  SEXP prec_theta2_samples = PROTECT(allocVector(REALSXP, n_draws));
  SEXP prec_theta3_samples = PROTECT(allocVector(REALSXP, n_draws));
  SEXP alpha_samples       = PROTECT(allocMatrix(REALSXP, n_draws, n));
  SEXP z_samples           = PROTECT(allocMatrix(REALSXP, n_draws, n));

  SEXP log_sigma_samples   = R_NilValue;
  SEXP accept_prop_samples = R_NilValue;
  int n_outputs = 13;
  int n_protect = 13;

  if (use_logit) {
    if (return_log_sigma) {
      log_sigma_samples = PROTECT(allocMatrix(REALSXP, n_draws, n));
      n_outputs++;
      n_protect++;
    }
    if (return_accept_prop) {
      accept_prop_samples = PROTECT(allocMatrix(REALSXP, n_draws, n));
      n_outputs++;
      n_protect++;
    }
  }

  /* ========== Allocate Temporary Buffers (O(n) Storage) ========== */
  double *theta_1_current  = (double *) R_Calloc(n, double);
  double *theta_1_previous = (double *) R_Calloc(n, double);
  double *theta_2_current  = (double *) R_Calloc(n, double);
  double *theta_2_previous = (double *) R_Calloc(n, double);
  double *theta_3_current  = (double *) R_Calloc(n, double);
  double *theta_3_previous = (double *) R_Calloc(n, double);
  double *alpha_current    = (double *) R_Calloc(n, double);
  double *z_current        = (double *) R_Calloc(n, double);

  /* Mixture component parameters (structure: [lambda_1, lambda_2]). No
   * `previous` companion: the conjugate Gamma conditional depends on the data
   * and the indicators alone, never on the preceding draw. */
  double *params_current = (double *) R_Calloc(2, double);

  double theta_01_current, theta_01_previous;
  double theta_02_current, theta_02_previous;
  double theta_03_current, theta_03_previous;
  double prec_theta1_current, prec_theta1_previous;
  double prec_theta2_current, prec_theta2_previous;
  double prec_theta3_current, prec_theta3_previous;

  /* Half-t auxiliaries b = 1/a for the state precisions W_1, W_2 and W_3
   * (refreshed when the matching prior is Half-t; left at 0 and never read
   * under the Gamma prior). The component rates carry no auxiliary. */
  double aux_W1, aux_W2, aux_W3;

  double *theta_1_updated = NULL;
  double *accept_prop     = NULL;
  double *log_sigma       = NULL;
  double *hat_theta_1     = NULL;
  double *theta_1_new     = NULL;
  double *log_accept_prob = NULL;
  double *rhs_vector      = NULL;

  if (use_logit) {
    theta_1_updated = (double *) R_Calloc(lag_update * n, double);
    accept_prop     = (double *) R_Calloc(n, double);
    log_sigma       = (double *) R_Calloc(n, double);
    hat_theta_1     = (double *) R_Calloc(n, double);
    theta_1_new     = (double *) R_Calloc(n, double);
    log_accept_prob = (double *) R_Calloc(n, double);

    for (int t = 0; t < n; t++) {
      log_sigma[t] = log(0.1);
    }
  } else {
    rhs_vector = (double *) R_Calloc(n, double);
  }

  /* ========== Initialize RNG State ========== */
  GetRNGstate();

  /* ========== Initialize Parameters (Iteration 0) ========== */
  /* Starting values arrive already decided from R (see resolve_init() in
   * R/init_values.R): whatever the caller pinned through `init`, and a draw
   * from the corresponding prior for everything else. R also applies the
   * lambda_1 <= lambda_2 relabelling. `init_` is laid out as the initial states
   * in order, then the component rates and the state precisions, then one
   * auxiliary per entry of that second block. */
  const double *init = REAL(init_);

  theta_01_previous    = init[0];
  theta_02_previous    = init[1];
  theta_03_previous    = init[2];
  params_current[0]    = init[3];   /* lambda_1 */
  params_current[1]    = init[4];   /* lambda_2 */
  prec_theta1_previous = init[5];
  prec_theta2_previous = init[6];
  prec_theta3_previous = init[7];
  /* init[8], init[9] are the (unused) rate auxiliaries. */
  aux_W1               = init[10];
  aux_W2               = init[11];
  aux_W3               = init[12];

  /* The starting weight is the one the model implies for the starting state,
   * alpha_t = link(theta_{t,1}) with the trajectory flat at theta_{0,1}. This
   * sampler *reads* alpha before writing it -- step 2 draws z | alpha -- so the
   * value matters. z stays Bernoulli(0.5), the maximum-entropy start and a
   * source of the between-chain dispersion the Gelman-Rubin statistic needs;
   * see docs/starting-values.md. */
  const double alpha_0 = use_logit
    ? ilogit_guarded(theta_01_previous)
    : clamp_link_alpha(pnorm(theta_01_previous, 0.0, 1.0, 1, 0));

  for (int t = 0; t < n; t++) {
    theta_1_previous[t] = theta_01_previous;
    theta_2_previous[t] = theta_02_previous;
    theta_3_previous[t] = theta_03_previous;
    alpha_current[t]    = alpha_0;
    z_current[t]        = (unif_rand() < 0.5) ? 1.0 : 0.0;
  }

  /* ========== Main Gibbs Sampling Loop ========== */
  int chain_idx = 0;

  for (int ii = 1; ii < n_iter; ii++) {
    int compute_alpha = (ii >= burnin && ((ii - burnin) % thinning) == 0) ? 1 : 0;

    /* ===== Step 1: Sample Mixture Component Rates (lambda) ===== */
    /* Draw (lambda_1, lambda_2) | y, z from conjugate Gamma posteriors.
     * Enforces the label constraint lambda_1 < lambda_2 by swapping. */
    conditional_mixture_poisson_parameters_k2(
      y,                  /* observed counts [n] */
      z_current,          /* latent indicators [n] from previous iteration */
      params_current,     /* output: current [lambda_1, lambda_2] */
      lambda_01_shape,    /* prior shape for lambda_1 */
      lambda_01_rate,     /* prior rate for lambda_1 */
      lambda_02_shape,    /* prior shape for lambda_2 */
      lambda_02_rate,     /* prior rate for lambda_2 */
      n                   /* sample size */
    );

    /* ===== Step 2: Sample Latent Indicators z ===== */
    /* Draw z_t | y, alpha, lambda ~ Bernoulli(alpha*_t) using the rates just
     * sampled and alpha from the previous iteration. */
    conditional_mixture_poisson_indicators_k2(
      y,                  /* observed counts [n] */
      z_current,          /* output: latent indicators [n] */
      params_current,     /* current [lambda_1, lambda_2] */
      alpha_current,      /* current mixture weights [n] from previous iteration */
      n                   /* sample size */
    );

    /* ===== Step 3: Sample Acceleration State Vector theta_3 ===== */
    generate_theta_p(
      theta_2_previous,     /* theta_{p-1}: trend from previous iteration [n] */
      theta_3_current,      /* output: current iteration theta_3 [n] */
      prec_theta2_previous, /* scalar: trend precision from previous iteration */
      prec_theta3_previous, /* scalar: acceleration precision from previous iteration */
      theta_03_previous,    /* scalar: initial acceleration from previous iteration */
      n                     /* sample size */
    );

    /* ===== Step 4: Sample Acceleration Innovation Precision 1/W_3 ===== */
    prec_theta3_current = update_prec_W3(
      theta_03_previous,  /* scalar: initial acceleration from previous iteration */
      theta_3_current,    /* vector: current theta_3 [n] */
      n,                  /* sample size */
      &prior_W3,          /* prior hyperparameters for the resolved kind */
      &aux_W3             /* Half-t auxiliary (updated in place; unused if Gamma) */
    );

    /* ===== Step 5: Sample Initial Acceleration State theta_{0,3} ===== */
    theta_03_current = generate_theta_0p(
      theta_2_previous,     /* theta_{p-1}: trend from previous iteration [n] */
      theta_3_current,      /* theta_p: current acceleration [n] */
      theta_02_previous,    /* theta_{0,p-1}: initial trend from previous iteration */
      prec_theta2_previous, /* W_{p-1}^{-1}: trend precision from previous iteration */
      prec_theta3_current,  /* W_p^{-1}: current acceleration precision */
      mean_theta03,         /* prior mean */
      prec_theta03,         /* prior precision */
      n                     /* sample size */
    );

    /* ===== Step 6: Sample Trend State Vector theta_2 ===== */
    generate_theta_k(
      theta_1_previous,     /* theta_{k-1}: level from previous iteration [n] */
      theta_2_current,      /* output: current iteration theta_2 [n] */
      theta_3_current,      /* theta_{k+1}: current acceleration [n] */
      prec_theta1_previous, /* scalar: level precision from previous iteration */
      prec_theta2_previous, /* scalar: trend precision from previous iteration */
      theta_02_previous,    /* scalar: initial trend from previous iteration */
      theta_03_current,     /* scalar: current initial acceleration */
      n                     /* sample size */
    );

    /* ===== Step 7: Sample Trend Innovation Precision 1/W_2 ===== */
    prec_theta2_current = update_prec_W2(
      theta_02_previous,  /* scalar: initial trend from previous iteration */
      theta_03_current,   /* scalar: current initial acceleration */
      theta_2_current,    /* vector: current theta_2 [n] */
      theta_3_current,    /* vector: current theta_3 [n] */
      n,                  /* sample size */
      &prior_W2,          /* prior hyperparameters for the resolved kind */
      &aux_W2             /* Half-t auxiliary (updated in place; unused if Gamma) */
    );

    /* ===== Step 8: Sample Initial Trend State theta_{0,2} ===== */
    theta_02_current = generate_theta_0k(
      theta_1_previous,     /* theta_{k-1}: level from previous iteration [n] */
      theta_2_current,      /* theta_k: current trend [n] */
      theta_01_previous,    /* theta_{0,k-1}: initial level from previous iteration */
      theta_03_current,     /* theta_{0,k+1}: current initial acceleration */
      prec_theta1_previous, /* prec_{k-1}: level precision from previous iteration */
      prec_theta2_current,  /* prec_k: current trend precision */
      mean_theta02,         /* prior mean */
      prec_theta02,         /* prior precision */
      n                     /* sample size */
    );

    /* ===== Step 9: Sample Level State theta_1 and Mixture Weights alpha ===== */
    /* The latent indicators z act as Bernoulli "pseudo-observations", which is
     * what lets the dynamic-GLM binomial samplers serve the mixture weight. */
    if (use_logit) {
      generate_alpha_logit_binomial(
        theta_1_previous,          /* theta_1: level from previous iteration [n] */
        theta_1_current,           /* output: current iteration theta_1 [n] */
        alpha_current,             /* alpha: always computed for next iteration's z sampling */
        theta_2_current,           /* theta_2: trend from current iteration [n] */
        theta_01_previous,         /* theta_{0,1}: initial level from previous iteration */
        theta_02_current,          /* theta_{0,2}: initial trend from current iteration */
        prec_theta1_previous,      /* W_1^{-1}: level precision from previous iteration */
        theta_1_updated,           /* sliding window workspace [lag_update * n] */
        z_current,                 /* z: latent indicators as "observations" [n] */
        accept_prop,               /* acceptance proportions workspace [n] */
        log_sigma,                 /* proposal scale parameters [n] */
        hat_theta_1,               /* conditional means workspace [n] */
        theta_1_new,               /* proposal states workspace [n] */
        log_accept_prob,           /* MH log-acceptance ratios [n] */
        lag_update,                /* adaptation lag */
        1.0,                       /* n_trials: 1 for Bernoulli (z in {0,1}) */
        n,                         /* series length */
        ii,                        /* current iteration */
        max_step_size,             /* proposal cap */
        base_adaptation_rate,      /* base adaptation weight */
        decay_exponent,            /* adaptation decay */
        target_acceptance,         /* desired acceptance rate */
        min_deviation_threshold,   /* adaptation trigger */
        1                          /* compute_alpha: force true */
      );
    } else {
      generate_alpha_probit_bernoulli(
        theta_1_previous,          /* theta_1: level from previous iteration [n] */
        theta_1_current,           /* output: current iteration theta_1 [n] */
        alpha_current,             /* alpha: always computed for next iteration's z sampling */
        theta_2_current,           /* theta_2: trend from current iteration [n] */
        theta_01_previous,         /* theta_{0,1}: initial level from previous iteration */
        theta_02_current,          /* theta_{0,2}: initial trend from current iteration */
        prec_theta1_previous,      /* W_1^{-1}: level precision from previous iteration */
        z_current,                 /* z: latent indicators [n] */
        rhs_vector,                /* workspace: solver right-hand side [n] */
        n,                         /* series length */
        1                          /* compute_alpha: force true */
      );
    }

    /* ===== Step 10: Sample Level Innovation Precision 1/W_1 ===== */
    prec_theta1_current = update_prec_W1(
      theta_01_previous,  /* scalar: initial level from previous iteration */
      theta_02_current,   /* scalar: current initial trend */
      theta_1_current,    /* vector: current level [n] */
      theta_2_current,    /* vector: current trend [n] */
      n,                  /* sample size */
      &prior_W1,          /* prior hyperparameters for the resolved kind */
      &aux_W1             /* Half-t auxiliary (updated in place; unused if Gamma) */
    );

    /* ===== Step 11: Sample Initial Level State theta_{0,1} ===== */
    theta_01_current = generate_theta_01(
      theta_1_current,     /* vector: current level [n] */
      theta_02_current,    /* scalar: current initial trend */
      prec_theta1_current, /* scalar: current level precision */
      mean_theta01,        /* prior mean */
      prec_theta01,        /* prior precision */
      n                    /* sample size */
    );

    /* ===== Store Post-Burn-in Samples with Thinning ===== */
    if (compute_alpha) {
      int idx = chain_idx++;

      REAL(lambda_1_samples)[idx]    = params_current[0];
      REAL(lambda_2_samples)[idx]    = params_current[1];
      REAL(theta_01_samples)[idx]    = theta_01_current;
      REAL(theta_02_samples)[idx]    = theta_02_current;
      REAL(theta_03_samples)[idx]    = theta_03_current;
      REAL(prec_theta1_samples)[idx] = prec_theta1_current;
      REAL(prec_theta2_samples)[idx] = prec_theta2_current;
      REAL(prec_theta3_samples)[idx] = prec_theta3_current;

      for (int t = 0; t < n; t++) {
        REAL(theta_1_samples)[idx + t * n_draws] = theta_1_current[t];
        REAL(theta_2_samples)[idx + t * n_draws] = theta_2_current[t];
        REAL(theta_3_samples)[idx + t * n_draws] = theta_3_current[t];
        REAL(alpha_samples)[idx + t * n_draws]   = alpha_current[t];
        REAL(z_samples)[idx + t * n_draws]       = z_current[t];

        if (use_logit) {
          if (return_log_sigma) {
            REAL(log_sigma_samples)[idx + t * n_draws] = log_sigma[t];
          }
          if (return_accept_prop) {
            REAL(accept_prop_samples)[idx + t * n_draws] = accept_prop[t];
          }
        }
      }
    }

    if (ii % pb.update_step == 0 || ii == n_iter - 1) {
      progress_bar_update(&pb, ii);
    }

    memcpy(theta_1_previous, theta_1_current, n * sizeof(double));
    memcpy(theta_2_previous, theta_2_current, n * sizeof(double));
    memcpy(theta_3_previous, theta_3_current, n * sizeof(double));

    theta_01_previous    = theta_01_current;
    theta_02_previous    = theta_02_current;
    theta_03_previous    = theta_03_current;
    prec_theta1_previous = prec_theta1_current;
    prec_theta2_previous = prec_theta2_current;
    prec_theta3_previous = prec_theta3_current;
  }

  progress_bar_finish(&pb, n_draws);
  PutRNGstate();

  /* ========== Free Temporary Buffers ========== */
  R_Free(theta_1_current);
  R_Free(theta_1_previous);
  R_Free(theta_2_current);
  R_Free(theta_2_previous);
  R_Free(theta_3_current);
  R_Free(theta_3_previous);
  R_Free(alpha_current);
  R_Free(z_current);
  R_Free(params_current);

  if (use_logit) {
    R_Free(theta_1_updated);
    R_Free(accept_prop);
    R_Free(log_sigma);
    R_Free(hat_theta_1);
    R_Free(theta_1_new);
    R_Free(log_accept_prob);
  } else {
    R_Free(rhs_vector);
  }

  /* ========== Package Results into Named List ========== */
  SEXP out = PROTECT(allocVector(VECSXP, n_outputs));
  SEXP nms = PROTECT(allocVector(STRSXP, n_outputs));

  int output_idx = 0;

  SET_VECTOR_ELT(out, output_idx, lambda_1_samples);
  SET_STRING_ELT(nms, output_idx++, mkChar("lambda_1"));

  SET_VECTOR_ELT(out, output_idx, lambda_2_samples);
  SET_STRING_ELT(nms, output_idx++, mkChar("lambda_2"));

  SET_VECTOR_ELT(out, output_idx, theta_1_samples);
  SET_STRING_ELT(nms, output_idx++, mkChar("theta_1"));

  SET_VECTOR_ELT(out, output_idx, theta_2_samples);
  SET_STRING_ELT(nms, output_idx++, mkChar("theta_2"));

  SET_VECTOR_ELT(out, output_idx, theta_3_samples);
  SET_STRING_ELT(nms, output_idx++, mkChar("theta_3"));

  SET_VECTOR_ELT(out, output_idx, theta_01_samples);
  SET_STRING_ELT(nms, output_idx++, mkChar("theta_01"));

  SET_VECTOR_ELT(out, output_idx, theta_02_samples);
  SET_STRING_ELT(nms, output_idx++, mkChar("theta_02"));

  SET_VECTOR_ELT(out, output_idx, theta_03_samples);
  SET_STRING_ELT(nms, output_idx++, mkChar("theta_03"));

  SET_VECTOR_ELT(out, output_idx, prec_theta1_samples);
  SET_STRING_ELT(nms, output_idx++, mkChar("prec_theta1"));

  SET_VECTOR_ELT(out, output_idx, prec_theta2_samples);
  SET_STRING_ELT(nms, output_idx++, mkChar("prec_theta2"));

  SET_VECTOR_ELT(out, output_idx, prec_theta3_samples);
  SET_STRING_ELT(nms, output_idx++, mkChar("prec_theta3"));

  SET_VECTOR_ELT(out, output_idx, alpha_samples);
  SET_STRING_ELT(nms, output_idx++, mkChar("alpha"));

  SET_VECTOR_ELT(out, output_idx, z_samples);
  SET_STRING_ELT(nms, output_idx++, mkChar("z"));

  if (use_logit) {
    if (return_log_sigma) {
      SET_VECTOR_ELT(out, output_idx, log_sigma_samples);
      SET_STRING_ELT(nms, output_idx++, mkChar("log_sigma"));
    }
    if (return_accept_prop) {
      SET_VECTOR_ELT(out, output_idx, accept_prop_samples);
      SET_STRING_ELT(nms, output_idx++, mkChar("accept_prop"));
    }
  }

  setAttrib(out, R_NamesSymbol, nms);

  UNPROTECT(n_protect + 2);
  return out;
}
