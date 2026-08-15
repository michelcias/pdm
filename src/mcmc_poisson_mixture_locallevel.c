/**
 * @file mcmc_poisson_mixture_locallevel.c
 * @brief MCMC sampling for Poisson mixture models with dynamic mixture weights
 * under a local-level evolution
 * @author Michel H. Montoril
 * @date 2026-08-15
 * @version 1.0
 *
 * @details Implements the complete Gibbs sampler for Bayesian estimation of
 *          two-component Poisson mixture models with time-varying mixture
 *          weights that follow a local-level stochastic process. The dynamic
 *          weight machinery is shared verbatim with the Gaussian mixture
 *          sampler; only the observation model differs.
 *
 * **Supported link functions:**
 * - Logit:  Component-wise Metropolis-Hastings with adaptive tuning
 * - Probit: Gibbs sampling via Albert-Chib data augmentation
 *
 * **Poisson mixture model with dynamic weights:**
 * Observation: y_t | z_t, lambda ~ Poisson(z_t*lambda_2 + (1-z_t)*lambda_1)
 * Indicators:  z_t | alpha_t ~ Bernoulli(alpha_t)
 *
 * **Dynamic weight model (local level):**
 * theta_{t,1} = theta_{t-1,1} + u_{t,1},  u_{t,1} ~ N(0, W_1)
 * alpha_t = T^{-1}(theta_{t,1}),  T in {logit, probit}
 *
 * **Prior distributions:**
 * - lambda_k ~ Gamma(a_0k, b_0k),         k = 1, 2
 * - theta_{0,1} ~ N(mu_{0,1}, sigma^2_{0,1})
 * - 1/W_1 ~ Gamma(nu_1, eta_1), or a Half-t prior on sqrt(W_1) (Gelman, 2006),
 *   selected via the prior_prec1_type_ code (see prec_prior_dispatch.h)
 *
 * The component rates carry a Gamma prior and nothing else. The Half-t option
 * exists for *precisions* -- parameters whose posterior has a likelihood
 * singularity at zero variance -- and a Poisson rate is not one: the conjugate
 * Gamma is proper for any positive shape and rate, and the mixture has no
 * degenerate direction to guard against. That is why this family has no
 * `prior_lambda0k_type` argument.
 *
 * **Sampling sequence per iteration:**
 * 1. (lambda_1, lambda_2) | y, z -> Conjugate Gamma updates
 * 2. z | y, alpha, lambda -> Bernoulli with weighted Poisson densities
 * 3. theta_1, alpha | z, theta_{0,1}, W_1 -> Link-specific sampling
 *    - Logit: Component-wise MH with adaptive tuning
 *    - Probit: Gibbs via latent utilities
 * 4. 1/W_1 | theta_1, theta_{0,1} -> Gamma or Half-t posterior
 * 5. theta_{0,1} | theta_1, W_1 -> Normal posterior
 */

#include <R.h>
#include <Rinternals.h>
#include <Rmath.h>
#include <string.h>  /* memcpy, strcmp */
#include "conditional_precision.h"
#include "conditional_theta0.h"
#include "conditional_mixture_poisson_parameters.h"
#include "conditional_mixture_poisson_indicators.h"
#include "generate_alpha_binomial.h"
#include "utils.h"
#include "link_guard.h"  /* ilogit_guarded, clamp_link_alpha */
#include "prec_prior_dispatch.h" /* prec_prior_t, step_prec_*, pdm_init_prec_prior */
#include "mcmc_progress_bar.h"
#include "mcmc_poisson_mixture_locallevel.h"

/**
 * @brief Unified Gibbs sampler for a Poisson mixture model with local-level weights
 *
 * @details Executes the full Gibbs sampling cycle for a two-component Poisson
 *          mixture model whose mixture weights evolve as a local-level random
 *          walk. Supports both logit and probit link functions via the `link`
 *          argument.
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
 * @param init_                    Double vector [7] of starting values, resolved in R by
 *                                 resolve_init(): theta_{0,1}, then lambda_1 and lambda_2,
 *                                 then 1/W_1, then one Half-t auxiliary per entry of that
 *                                 precision block. Only the W_1 auxiliary is ever read.
 * @param verbose_                 Logical: display progress bar (0 = FALSE, 1 = TRUE).
 * @param bar_width_               Integer: width of progress bar (10-120 recommended).
 *
 * @return R list with posterior samples and optional diagnostics.
 *
 * @note Complexity: O(n_iter * n) time, O(n) space
 * @note Requires n >= 3 for numerical stability
 * @note Proper RNG state management via GetRNGstate()/PutRNGstate()
 * @note Logit-specific parameters are ignored when link="probit"
 * @note Diagnostic outputs (log_sigma, accept_prop) are NULL when link="probit"
 *
 * @warning n must not exceed INT_MAX
 * @warning Memory allocation failures terminate R session
 * @warning link must be exactly "logit" or "probit" (case-sensitive)
 *
 * @see conditional_mixture_poisson_parameters_k2
 * @see conditional_mixture_poisson_indicators_k2
 * @see generate_alpha_logit_binomial_locallevel
 * @see generate_alpha_probit_bernoulli_locallevel
 * @see generate_precision_theta_p
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
                                       SEXP bar_width_) {

  /* ========== Parse and Validate Link Function ========== */
  const char *link = CHAR(STRING_ELT(link_, 0));

  if (strcmp(link, "logit") != 0 && strcmp(link, "probit") != 0) {
    Rf_error("C_MCMC_poisson_mixture_locallevel: link must be 'logit' or 'probit', got '%s'", link);
  }

  int use_logit = (strcmp(link, "logit") == 0);

  /* ========== Parse Data Vector and Validate Length ========== */
  double   *y   = REAL(y_);
  R_xlen_t  len = LENGTH(y_);

  if (len < 3) {
    Rf_error("C_MCMC_poisson_mixture_locallevel: sample size 'n' must be at least 3, got %lld",
             (long long) len);
  }

  if (len > INT_MAX) {
    Rf_error("C_MCMC_poisson_mixture_locallevel: sample size too large (%lld > %d)",
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
  SEXP lambda_1_samples    = PROTECT(allocVector(REALSXP, n_draws));
  SEXP lambda_2_samples    = PROTECT(allocVector(REALSXP, n_draws));
  SEXP theta_1_samples     = PROTECT(allocMatrix(REALSXP, n_draws, n));
  SEXP theta_01_samples    = PROTECT(allocVector(REALSXP, n_draws));
  SEXP prec_theta1_samples = PROTECT(allocVector(REALSXP, n_draws));
  SEXP alpha_samples       = PROTECT(allocMatrix(REALSXP, n_draws, n));
  SEXP z_samples           = PROTECT(allocMatrix(REALSXP, n_draws, n));

  SEXP log_sigma_samples   = R_NilValue;
  SEXP accept_prop_samples = R_NilValue;
  int n_outputs = 7;
  int n_protect = 7;

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
  double *alpha_current    = (double *) R_Calloc(n, double);
  double *z_current        = (double *) R_Calloc(n, double);

  /* Mixture component parameters (structure: [lambda_1, lambda_2]). No
   * `previous` companion: the conjugate Gamma conditional depends on the data
   * and the indicators alone, never on the preceding draw. */
  double *params_current = (double *) R_Calloc(2, double);

  double theta_01_current, theta_01_previous;
  double prec_theta1_current, prec_theta1_previous;

  /* Half-t auxiliary b = 1/a for the state precision W_1. Refreshed in place
   * when that prior is Half-t; left at 0 and never read under the Gamma prior.
   * The component rates carry no auxiliary of their own. */
  double aux_W1;

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
   * lambda_1 <= lambda_2 relabelling -- and refuses it, rather than silently
   * relabelling, when the caller pinned either rate. `init_` is laid out as the
   * initial states in order, then the component rates and the state precisions,
   * then one auxiliary per entry of that second block. */
  const double *init = REAL(init_);

  theta_01_previous    = init[0];
  params_current[0]    = init[1];   /* lambda_1 */
  params_current[1]    = init[2];   /* lambda_2 */
  prec_theta1_previous = init[3];
  /* init[4], init[5] are the (unused) rate auxiliaries; init[6] is W_1's. */
  aux_W1               = init[6];

  /* The starting weight is the one the model implies for the starting state,
   * alpha_t = link(theta_{t,1}) with the trajectory flat at theta_{0,1}, rather
   * than a fixed 0.5. This sampler *reads* alpha before writing it -- step 2
   * draws z | alpha -- so the value matters, and it is what makes the pair
   * (theta_1, alpha) a coherent state rather than two unrelated numbers.
   *
   * z stays Bernoulli(0.5), drawn here and deliberately not settable through
   * `init`: it is the between-chain dispersion the Gelman-Rubin statistic needs,
   * and it is the maximum-entropy start, so both components are guaranteed data
   * on the first sweep. The rationale is the Gaussian mixture's, measured in
   * docs/starting-values.md. */
  const double alpha_0 = use_logit
    ? ilogit_guarded(theta_01_previous)
    : clamp_link_alpha(pnorm(theta_01_previous, 0.0, 1.0, 1, 0));

  for (int t = 0; t < n; t++) {
    theta_1_previous[t] = theta_01_previous;
    alpha_current[t]    = alpha_0;
    z_current[t]        = (unif_rand() < 0.5) ? 1.0 : 0.0;
  }

  /* ========== Main Gibbs Sampling Loop ========== */
  int chain_idx = 0;

  for (int ii = 1; ii < n_iter; ii++) {
    int compute_alpha = (ii >= burnin && ((ii - burnin) % thinning) == 0) ? 1 : 0;

    /* ===== Step 1: Sample Mixture Component Rates (lambda) ===== */
    /* Draw (lambda_1, lambda_2) | y, z from conjugate Gamma posteriors.
     * Enforces the label constraint lambda_1 < lambda_2 by swapping.
     * Uses z from the previous iteration. */
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
     * sampled, which improves mixing over reusing the previous ones. */
    conditional_mixture_poisson_indicators_k2(
      y,                  /* observed counts [n] */
      z_current,          /* output: latent indicators [n] */
      params_current,     /* current [lambda_1, lambda_2] */
      alpha_current,      /* mixture weights alpha_t */
      n                   /* sample size */
    );

    /* ===== Step 3: Sample Level State theta_1 and Mixture Weights alpha ===== */
    if (use_logit) {
      /* Draw theta_1 via component-wise MH under the logit link, then transform
       * to alpha, updating the adaptive proposal statistics along the way. */
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
      /* Draw theta_1 via Gibbs sampling with the probit link, using Albert-Chib
       * augmentation. */
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

      REAL(lambda_1_samples)[idx]    = params_current[0];
      REAL(lambda_2_samples)[idx]    = params_current[1];
      REAL(theta_01_samples)[idx]    = theta_01_current;
      REAL(prec_theta1_samples)[idx] = prec_theta1_current;

      for (int t = 0; t < n; t++) {
        REAL(theta_1_samples)[idx + t * n_draws] = theta_1_current[t];
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

    theta_01_previous    = theta_01_current;
    prec_theta1_previous = prec_theta1_current;
  }

  progress_bar_finish(&pb, n_draws);
  PutRNGstate();

  /* ========== Free Temporary Buffers ========== */
  R_Free(theta_1_current);
  R_Free(theta_1_previous);
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
