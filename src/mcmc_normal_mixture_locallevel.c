/**
 * @file mcmc_normal_mixture_locallevel.c
 * @brief MCMC sampling for Gaussian mixture models with dynamic mixture weights
 * under a local-level evolution
 * @author Michel H. Montoril
 * @date 2025-01-07
 * @version 1.0
 *
 * @details Implements the complete Gibbs sampler for Bayesian estimation of
 * two-component Gaussian mixture models with time-varying mixture
 * weights that follow a local-level stochastic process. The
 * implementation mirrors the local-trend variant while omitting the
 * trend component, making it suitable for applications where a single
 * random-walk latent process governs the mixture weights.
 *
 * **Supported link functions:**
 * - Logit:  Component-wise Metropolis-Hastings with adaptive tuning
 * - Probit: Gibbs sampling via Albert-Chib data augmentation
 *
 * **Key features inherited from the local-trend sampler:**
 * - Memory-efficient O(n) temporary storage using current/previous buffers
 * - Conditional alpha computation for performance optimization
 * - Configurable adaptive tuning for Metropolis-Hastings (logit only)
 * - Conjugate posterior updates for all parameters
 * - Label switching constraint enforcement (mu_1 < mu_2)
 *
 * **Gaussian mixture model with dynamic weights:**
 * Observation: y_t | z_t, mu, phi ~ N(z_t*mu_2 + (1-z_t)*mu_1, [z_t*phi_2 + (1-z_t)*phi_1]^{-1})
 * Indicators:  z_t | alpha_t ~ Bernoulli(alpha_t)
 *
 * **Dynamic weight model (local level):**
 * theta_{t,1} = theta_{t-1,1} + u_{t,1},  u_{t,1} ~ N(0, W_1)
 * alpha_t = T^{-1}(theta_{t,1}),  T in {logit, probit}
 *
 * **Prior distributions:**
 * - mu_k ~ N(mu_0k, sigma^2_0k),          k = 1, 2
 * - phi_k ~ Gamma(nu_0k, eta_0k),         k = 1, 2
 * - theta_{0,1} ~ N(mu_{0,1}, sigma^2_{0,1})
 * - 1/W_1 ~ Gamma(nu_1, eta_1)
 * - every precision (phi_1, phi_2, 1/W_1) may instead take a Half-t prior on its
 *   standard deviation (Gelman, 2006), selected independently via the
 *   prior_prec*_type_ codes (see prec_prior_dispatch.h)
 *
 * **Sampling sequence per iteration:**
 * 1. (mu_1, phi_1, mu_2, phi_2) | y, z -> Conjugate Normal-Gamma updates
 * 2. z | y, alpha, mu, phi -> Bernoulli with weighted densities
 * 3. theta_1, alpha | z, theta_{0,1}, W_1 -> Link-specific sampling
 * - Logit: Component-wise MH with adaptive tuning
 * - Probit: Gibbs via latent utilities
 * 4. 1/W_1 | theta_1, theta_{0,1} -> Gamma posterior
 * 5. theta_{0,1} | theta_1, W_1 -> Normal posterior
 */

#include <R.h>
#include <Rinternals.h>
#include <Rmath.h>
#include <string.h>  /* memcpy, strcmp */
#include "conditional_precision.h"
#include "conditional_theta0.h"
#include "conditional_mixture_normal_parameters.h"
#include "conditional_mixture_normal_indicators.h"
#include "generate_alpha_binomial.h"
#include "utils.h"
#include "prec_prior_dispatch.h" /* prec_prior_t, step_prec_*, pdm_init_prec_prior */
#include "mcmc_progress_bar.h"
#include "mcmc_normal_mixture_locallevel.h"

/**
 * @brief Unified Gibbs sampler for Gaussian mixture model with local-level weights
 *
 * @details Executes the full Gibbs sampling cycle for a two-component Gaussian
 * mixture model whose mixture weights evolve as a local-level random
 * walk. Supports both logit and probit link functions via the `link`
 * argument.
 *
 * @param y_                       Numeric vector [n] of observations.
 * @param link_                    Character string: "logit" or "probit".
 * @param burnin_                  Number of burn-in iterations (discarded).
 * @param thinning_                Thinning interval for retained samples.
 * @param n_chain_                 Number of retained posterior samples.
 * @param prior_mu01_mean_         Prior mean for mu_1.
 * @param prior_mu01_prec_         Prior precision for mu_1.
 * @param prior_prec01_type_       Prior kind on phi_1 (0 = Gamma, 1 = Half-t on sqrt(1/phi_1)).
 * @param prior_prec01_shape_      Gamma shape for phi_1 (Gamma kind).
 * @param prior_prec01_rate_       Gamma rate for phi_1 (Gamma kind).
 * @param prior_prec01_scale_      Half-t scale A > 0 for phi_1 (Half-t kind).
 * @param prior_prec01_df_         Half-t df > 0 for phi_1 (Half-t kind; 1 = Half-Cauchy).
 * @param prior_mu02_mean_         Prior mean for mu_2.
 * @param prior_mu02_prec_         Prior precision for mu_2.
 * @param prior_prec02_type_       Prior kind on phi_2 (0 = Gamma, 1 = Half-t).
 * @param prior_prec02_shape_      Gamma shape for phi_2 (Gamma kind).
 * @param prior_prec02_rate_       Gamma rate for phi_2 (Gamma kind).
 * @param prior_prec02_scale_      Half-t scale A > 0 for phi_2 (Half-t kind).
 * @param prior_prec02_df_         Half-t df > 0 for phi_2 (Half-t kind; 1 = Half-Cauchy).
 * @param prior_theta01_mean_      Prior mean for theta_{0,1}.
 * @param prior_theta01_prec_      Prior precision for theta_{0,1}.
 * @param prior_prec1_type_        Prior kind on 1/W_1 (0 = Gamma, 1 = Half-t on sqrt(W_1)).
 * @param prior_prec1_shape_       Gamma shape for 1/W_1 (Gamma kind).
 * @param prior_prec1_rate_        Gamma rate for 1/W_1 (Gamma kind).
 * @param prior_prec1_scale_       Half-t scale A_1 > 0 for 1/W_1 (Half-t kind).
 * @param prior_prec1_df_          Half-t df nu_1 > 0 for 1/W_1 (Half-t kind; 1 = Half-Cauchy).
 * @param lag_update_              Adaptation frequency (logit only).
 * @param max_step_size_           Maximum proposal step size (logit only).
 * @param base_adaptation_rate_    Base adaptation rate (logit only).
 * @param decay_exponent_          Adaptation decay exponent (logit only).
 * @param target_acceptance_       Target acceptance proportion (logit only).
 * @param min_deviation_threshold_ Minimum deviation to trigger adaptation (logit only).
 * @param return_log_sigma_        Flag to return log_sigma diagnostics (logit only).
 * @param return_accept_prop_      Flag to return accept_prop diagnostics (logit only).
 * @param verbose_                 Logical: display progress bar (0 = FALSE, 1 = TRUE).
 * @param bar_width_               Integer: width of progress bar (10-120 recommended).
 *
 * @return R list with posterior samples and optional diagnostics.
 *
 * @note Complexity: O(n_iter * n) time, O(n) space
 * @note Requires n >= 3 for numerical stability
 * @note Proper RNG state management via GetRNGstate()/PutRNGstate()
 * @note Adaptation threshold default: 1.0/lag_update
 * @note Logit-specific parameters are ignored when link="probit"
 * @note Diagnostic outputs (log_sigma, accept_prop) are NULL when link="probit"
 *
 * @warning n must not exceed INT_MAX
 * @warning Memory allocation failures terminate R session
 * @warning link must be exactly "logit" or "probit" (case-sensitive)
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
                                      SEXP verbose_,
                                      SEXP bar_width_) {

  /* ========== Parse and Validate Link Function ========== */
  const char *link = CHAR(STRING_ELT(link_, 0));

  if (strcmp(link, "logit") != 0 && strcmp(link, "probit") != 0) {
    Rf_error("C_MCMC_normal_mixture_locallevel: link must be 'logit' or 'probit', got '%s'", link);
  }

  int use_logit = (strcmp(link, "logit") == 0);

  /* ========== Parse Data Vector and Validate Length ========== */
  double   *y   = REAL(y_);
  R_xlen_t  len = LENGTH(y_);

  if (len < 3) {
    Rf_error("C_MCMC_normal_mixture_locallevel: sample size 'n' must be at least 3, got %lld",
             (long long) len);
  }

  if (len > INT_MAX) {
    Rf_error("C_MCMC_normal_mixture_locallevel: sample size too large (%lld > %d)",
             (long long) len, INT_MAX);
  }

  int n = (int) len;

  /* ========== Parse MCMC Control Parameters ========== */
  int burnin   = INTEGER(burnin_)[0];
  int thinning = INTEGER(thinning_)[0];
  int n_chain  = INTEGER(n_chain_)[0];
  int n_iter   = burnin + (n_chain - 1) * thinning + 1;

  /* ========== Parse Mixture Component Prior Hyperparameters ========== */
  /* Each component precision phi_k carries a Gamma or Half-t prior, resolved in
   * R to an integer code plus finite hyperparameters (unused fields are finite
   * placeholders). The mixture parameter sampler dispatches on the kind. */
  double mu_01_mean    = REAL(prior_mu01_mean_)[0];
  double mu_01_prec    = REAL(prior_mu01_prec_)[0];
  int    phi1_kind     = asInteger(prior_prec01_type_);
  prec_prior_t phi_prior_1 = {
    .shape    = REAL(prior_prec01_shape_)[0],
    .rate     = REAL(prior_prec01_rate_)[0],
    .df       = REAL(prior_prec01_df_)[0],
    .hc_scale = REAL(prior_prec01_scale_)[0]
  };
  double mu_02_mean    = REAL(prior_mu02_mean_)[0];
  double mu_02_prec    = REAL(prior_mu02_prec_)[0];
  int    phi2_kind     = asInteger(prior_prec02_type_);
  prec_prior_t phi_prior_2 = {
    .shape    = REAL(prior_prec02_shape_)[0],
    .rate     = REAL(prior_prec02_rate_)[0],
    .df       = REAL(prior_prec02_df_)[0],
    .hc_scale = REAL(prior_prec02_scale_)[0]
  };

  /* ========== Parse Dynamic State Prior Hyperparameters ========== */
  double mean_theta01 = REAL(prior_theta01_mean_)[0];
  double prec_theta01 = REAL(prior_theta01_prec_)[0];
  int    prec1_kind   = asInteger(prior_prec1_type_);   /* prior on 1/W_1 */
  prec_prior_t prior_W1   = {
    .shape    = REAL(prior_prec1_shape_)[0],
    .rate     = REAL(prior_prec1_rate_)[0],
    .df       = REAL(prior_prec1_df_)[0],
    .hc_scale = REAL(prior_prec1_scale_)[0]
  };
  /* W_1 is the terminal random-walk component (theta_p sampler). */
  prec_thetap_step_t update_prec_W1 =
    (prec1_kind == PDM_PREC_PRIOR_HALFT) ? step_prec_thetap_halft
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
  SEXP mu_1_samples        = PROTECT(allocVector(REALSXP, n_chain));
  SEXP prec_1_samples      = PROTECT(allocVector(REALSXP, n_chain));
  SEXP mu_2_samples        = PROTECT(allocVector(REALSXP, n_chain));
  SEXP prec_2_samples      = PROTECT(allocVector(REALSXP, n_chain));
  SEXP theta_1_samples     = PROTECT(allocMatrix(REALSXP, n_chain, n));
  SEXP theta_01_samples    = PROTECT(allocVector(REALSXP, n_chain));
  SEXP prec_theta1_samples = PROTECT(allocVector(REALSXP, n_chain));
  SEXP alpha_samples       = PROTECT(allocMatrix(REALSXP, n_chain, n));
  SEXP z_samples           = PROTECT(allocMatrix(REALSXP, n_chain, n));

  SEXP log_sigma_samples   = R_NilValue;
  SEXP accept_prop_samples = R_NilValue;
  int n_outputs = 9;
  int n_protect = 9;

  if (use_logit) {
    if (return_log_sigma) {
      log_sigma_samples = PROTECT(allocMatrix(REALSXP, n_chain, n));
      n_outputs++;
      n_protect++;
    }
    if (return_accept_prop) {
      accept_prop_samples = PROTECT(allocMatrix(REALSXP, n_chain, n));
      n_outputs++;
      n_protect++;
    }
  }

  /* ========== Allocate Temporary Buffers (O(n) Storage) ========== */
  double *theta_1_current  = (double *) R_Calloc(n, double);
  double *theta_1_previous = (double *) R_Calloc(n, double);
  double *alpha_current    = (double *) R_Calloc(n, double);
  double *z_current        = (double *) R_Calloc(n, double);

  double *params_current  = (double *) R_Calloc(4, double);
  double *params_previous = (double *) R_Calloc(4, double);

  double theta_01_current, theta_01_previous;
  double prec_theta1_current, prec_theta1_previous;

  /* Half-t auxiliaries b = 1/a: one for the state precision W_1 and one per
   * mixture component precision phi_k. Refreshed in place when the matching
   * prior is Half-t; left at 0 and never read under the Gamma prior. */
  double aux_W1 = 0.0, aux_phi1 = 0.0, aux_phi2 = 0.0;

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
  params_previous[0] = rnorm(mu_01_mean, sqrt(1.0 / mu_01_prec));
  params_previous[1] = pdm_init_prec_prior(phi1_kind, &phi_prior_1, &aux_phi1);
  params_previous[2] = rnorm(mu_02_mean, sqrt(1.0 / mu_02_prec));
  params_previous[3] = pdm_init_prec_prior(phi2_kind, &phi_prior_2, &aux_phi2);

  if (params_previous[0] > params_previous[2]) {
    double temp_mu = params_previous[0];
    params_previous[0] = params_previous[2];
    params_previous[2] = temp_mu;
    double temp_prec = params_previous[1];
    params_previous[1] = params_previous[3];
    params_previous[3] = temp_prec;
    /* Keep each Half-t auxiliary paired with its component precision. */
    double temp_aux = aux_phi1;
    aux_phi1 = aux_phi2;
    aux_phi2 = temp_aux;
  }

  theta_01_previous    = rnorm(mean_theta01, sqrt(1.0 / prec_theta01));
  prec_theta1_previous = pdm_init_prec_prior(prec1_kind, &prior_W1, &aux_W1);

  for (int t = 0; t < n; t++) {
    theta_1_previous[t] = 0.0;
    alpha_current[t]    = 0.5;
    z_current[t]        = 0.0;
  }

  /* ========== Main Gibbs Sampling Loop ========== */
  int chain_idx = 0;

  for (int ii = 1; ii < n_iter; ii++) {
    int compute_alpha = (ii >= burnin && ((ii - burnin) % thinning) == 0) ? 1 : 0;

    /* ===== Step 1: Sample Mixture Component Parameters (mu, phi) ===== */
    /* Draw (mu_1, phi_1, mu_2, phi_2) | y, z from conjugate Normal-Gamma posteriors.
     * Enforces label-switching constraint mu_1 < mu_2 via component swapping.
     * Uses z from previous iteration and updates parameters in params_current. */
    conditional_mixture_normal_parameters_k2(
      y,                  /* observed data [n] */
      z_current,          /* latent indicators [n] from previous iteration */
      params_previous,    /* previous [mu_1, prec_1, mu_2, prec_2] */
      params_current,     /* output: current [mu_1, prec_1, mu_2, prec_2] */
      mu_01_mean,         /* prior mean for mu_1 */
      mu_01_prec,         /* prior precision for mu_1 */
      phi1_kind,          /* prior kind for phi_1 (Gamma / Half-t) */
      &phi_prior_1,       /* phi_1 hyperparameters for the resolved kind */
      &aux_phi1,          /* Half-t auxiliary for phi_1 (in/out; unused if Gamma) */
      mu_02_mean,         /* prior mean for mu_2 */
      mu_02_prec,         /* prior precision for mu_2 */
      phi2_kind,          /* prior kind for phi_2 (Gamma / Half-t) */
      &phi_prior_2,       /* phi_2 hyperparameters for the resolved kind */
      &aux_phi2,          /* Half-t auxiliary for phi_2 (in/out; unused if Gamma) */
      n                   /* sample size */
    );

    /* ===== Step 2: Sample Latent Indicators z ===== */
    /* Draw z_t | y, alpha, mu, phi ~ Bernoulli(alpha*_t) where alpha*_t uses
     * freshly sampled component parameters to improve mixing. */
    conditional_mixture_normal_indicators_k2(
      y,                  /* observed data [n] */
      z_current,          /* output: latent indicators [n] */
      params_current,     /* current [mu_1, prec_1, mu_2, prec_2] */
      alpha_current,      /* mixture weights alpha_t */
      n                   /* sample size */
    );

    /* ===== Step 3: Sample Level State theta_1 and Mixture Weights alpha ===== */
    if (use_logit) {
      /* Draw theta_1 via component-wise MH under logit link.
       * Optionally compute alpha_t for retained iterations and update
       * adaptive proposal statistics. */
      generate_alpha_logit_binomial_locallevel(
        theta_1_previous,             /* theta_1_previous: level states from previous iter [n] */
        theta_1_current,              /* theta_1_current: output level states for current iter [n] */
        alpha_current,                /* alpha_current: mixture weights (always computed) */
        theta_01_previous,            /* theta_01_previous: initial level from previous iter */
        prec_theta1_previous,         /* prec_theta1_previous: level precision from previous iter */
        theta_1_updated,              /* theta_1_updated: workspace for lagged updates */
        z_current,                    /* z_current: Bernoulli indicators */
        accept_prop,                  /* accept_prop: acceptance proportions (diagnostics) */
        log_sigma,                    /* log_sigma: log proposal scales */
        hat_theta_1,                  /* hat_theta_1: conditional means workspace */
        theta_1_new,                  /* theta_1_new: proposal states workspace */
        log_accept_prob,              /* log_accept_prob: MH log-acceptance ratios */
        lag_update,                   /* lag_update: adaptation lag */
        1.0,                          /* n_trials: effective binomial trials (mixture indicators) */
        n,                            /* n: sample size */
        ii,                           /* iter: current iteration */
        max_step_size,                /* max_step_size: cap on proposal sd */
        base_adaptation_rate,         /* base_adaptation_rate: adaptation weight */
        decay_exponent,               /* decay_exponent: adaptation decay */
        target_acceptance,            /* target_acceptance: desired acceptance rate */
        min_deviation_threshold,      /* min_deviation_threshold: adaptation trigger */
        1                             /* compute_alpha: force true */
      );
    } else {
      /* Draw theta_1 via Gibbs sampling with probit link using Albert-Chib
       * augmentation. Optionally computes alpha_t for retained iterations. */
      generate_alpha_probit_bernoulli_locallevel(
        theta_1_previous,             /* theta_1_previous: level states from previous iter [n] */
        theta_1_current,              /* theta_1_current: output level states for current iter [n] */
        alpha_current,                /* alpha_current: mixture weights (always computed) */
        theta_01_previous,            /* theta_01_previous: initial level from previous iter */
        prec_theta1_previous,         /* prec_theta1_previous: level precision from previous iter */
        z_current,                    /* z_current: Bernoulli indicators */
        rhs_vector,                   /* rhs_vector: workspace for tridiagonal solver */
        n,                            /* n: sample size */
        1                             /* compute_alpha: force true */
      );
    }

    /* ===== Step 4: Sample Innovation Precision 1/W_1 ===== */
    /* Draw 1/W_1 | theta_1, theta_01 through the prior chosen before the loop
     * (Gamma posterior, or Half-t via its scale-mixture step, which also
     * refreshes aux_W1). */
    prec_theta1_current = update_prec_W1(
      theta_01_previous,  /* theta_0p: initial level from previous iteration */
      theta_1_current,    /* theta_p_current: current level trajectory [n] */
      n,                  /* n: number of time points */
      &prior_W1,          /* prior hyperparameters for the resolved kind */
      &aux_W1             /* Half-t auxiliary (updated in place; unused if Gamma) */
    );

    /* ===== Step 5: Sample Initial State theta_{0,1} ===== */
    /* Draw theta_{0,1} | theta_1, W_1 from Normal posterior. */
    theta_01_current = generate_theta_01_locallevel(
      theta_1_current,     /* theta_1_current: current level trajectory [n] */
      prec_theta1_current, /* prec_theta1: current level precision */
      mean_theta01,        /* mean_theta01: prior mean */
      prec_theta01,        /* prec_theta01: prior precision */
      n                    /* n: number of time points */
    );

    /* ===== Store Post-Burn-in Samples with Thinning ===== */
    if (compute_alpha) {
      int idx = chain_idx++;

      REAL(mu_1_samples)[idx]        = params_current[0];
      REAL(prec_1_samples)[idx]      = params_current[1];
      REAL(mu_2_samples)[idx]        = params_current[2];
      REAL(prec_2_samples)[idx]      = params_current[3];
      REAL(theta_01_samples)[idx]    = theta_01_current;
      REAL(prec_theta1_samples)[idx] = prec_theta1_current;

      for (int t = 0; t < n; t++) {
        REAL(theta_1_samples)[idx + t * n_chain] = theta_1_current[t];
        REAL(alpha_samples)[idx + t * n_chain] = alpha_current[t];
        REAL(z_samples)[idx + t * n_chain]       = z_current[t];

        if (use_logit) {
          if (return_log_sigma) {
            REAL(log_sigma_samples)[idx + t * n_chain] = log_sigma[t];
          }
          if (return_accept_prop) {
            REAL(accept_prop_samples)[idx + t * n_chain] = accept_prop[t];
          }
        }
      }
    }

    if (ii % pb.update_step == 0 || ii == n_iter - 1) {
      progress_bar_update(&pb, ii);
    }

    memcpy(theta_1_previous, theta_1_current, n * sizeof(double));
    memcpy(params_previous, params_current, 4 * sizeof(double));

    theta_01_previous    = theta_01_current;
    prec_theta1_previous = prec_theta1_current;
  }

  progress_bar_finish(&pb, n_chain);
  PutRNGstate();

  /* ========== Free Temporary Buffers ========== */
  R_Free(theta_1_current);
  R_Free(theta_1_previous);
  R_Free(alpha_current);
  R_Free(z_current);
  R_Free(params_current);
  R_Free(params_previous);

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

  SET_VECTOR_ELT(out, output_idx, mu_1_samples);
  SET_STRING_ELT(nms, output_idx++, mkChar("mu_1"));

  SET_VECTOR_ELT(out, output_idx, prec_1_samples);
  SET_STRING_ELT(nms, output_idx++, mkChar("prec_1"));

  SET_VECTOR_ELT(out, output_idx, mu_2_samples);
  SET_STRING_ELT(nms, output_idx++, mkChar("mu_2"));

  SET_VECTOR_ELT(out, output_idx, prec_2_samples);
  SET_STRING_ELT(nms, output_idx++, mkChar("prec_2"));

  SET_VECTOR_ELT(out, output_idx, theta_1_samples);
  SET_STRING_ELT(nms, output_idx++, mkChar("theta_1"));

  SET_VECTOR_ELT(out, output_idx, theta_01_samples);
  SET_STRING_ELT(nms, output_idx++, mkChar("theta_01"));

  SET_VECTOR_ELT(out, output_idx, prec_theta1_samples);
  SET_STRING_ELT(nms, output_idx++, mkChar("prec_theta1"));

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
