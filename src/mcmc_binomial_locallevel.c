/**
 * @file mcmc_binomial_locallevel.c
 * @brief MCMC sampling for local-level binomial and Bernoulli dynamic models
 * @author Michel H. Montoril
 * @date 2025-10-11
 * @version 1.0
 *
 * @details Provides complete Gibbs samplers for Bayesian estimation of binomial and Bernoulli
 *          dynamic models with different link functions and local-level structure:
 *          - Logit-binomial: Component-wise Metropolis-Hastings with adaptive tuning
 *          - Probit-Bernoulli: Gibbs sampling via Albert-Chib data augmentation
 *
 *          All implementations utilize memory-efficient current/previous iteration buffers
 *          requiring only O(n) temporary storage regardless of chain length.
 *
 *          **Key features:**
 *          - Memory-efficient O(n) temporary storage using current/previous buffers
 *          - Conditional alpha computation for performance optimization
 *          - Configurable adaptive threshold for Metropolis-Hastings algorithms
 *          - Conjugate posterior updates for variance and initial state parameters
 *          - Flexible burn-in and thinning controls
 *
 *          **Logit-binomial model:**
 *          Observation: y_t ~ Binomial(n_trials, alpha_t), alpha_t = logit^{-1}(theta_{t,1})
 *          State: theta_{t,1} = theta_{t-1,1} + u_{t,1}, u_{t,1} ~ N(0, W_1)
 *
 *          **Probit-Bernoulli model:**
 *          Observation: y_t ~ Bernoulli(alpha_t), alpha_t = Phi(theta_{t,1})
 *          State: theta_{t,1} = theta_{t-1,1} + u_{t,1}, u_{t,1} ~ N(0, W_1)
 */

#include <R.h>
#include <Rinternals.h>
#include <Rmath.h>
#include <string.h>  /* memcpy */
#include "conditional_precision.h"
#include "conditional_theta0.h"
#include "generate_alpha_binomial.h"
#include "utils.h"
#include "prec_prior_dispatch.h" /* prec_prior_t, step_prec_*, pdm_init_prec_prior */
#include "mcmc_progress_bar.h"
#include "mcmc_binomial_locallevel.h"

/**
 * @brief Gibbs sampler for local-level binomial dynamic model with logit link
 *
 * @details Implements a complete Gibbs MCMC algorithm for the local-level binomial model:
 *
 *          **Observation equation:**
 *          y_t ~ Binomial(n_trials, alpha_t), where alpha_t = logit^{-1}(theta_{t,1})
 *
 *          **State equation:**
 *          theta_{t,1} = theta_{t-1,1} + u_{t,1}, u_{t,1} ~ N(0, W_1)
 *
 *          **Prior distributions:**
 *          - theta_{0,1} ~ N(mu_{0,1}, sigma_{0,1}^2)
 *          - 1/W_1       ~ Gamma(nu_1, eta_1)
 *
 *          The algorithm employs component-wise Metropolis-Hastings for the non-linear
 *          observation model with adaptive proposal tuning based on acceptance proportions.
 *          Innovation precision and initial state are sampled from conjugate posteriors.
 *
 *          **Optimizations implemented:**
 *          - Memory-efficient current/previous iteration buffers (O(n) storage)
 *          - Conditional alpha computation (skip during burn-in/thinning)
 *          - Configurable adaptation threshold (practical default: 1.0/lag_update)
 *          - Scalar parameter passing to avoid array indexing
 *          - Efficient initialization with zeros and neutral starting values
 *
 *          **Sampling sequence per iteration:**
 *          1. theta_1, alpha | y, theta_01, W_1 -> Component-wise MH with adaptive tuning
 *          2. 1/W_1 | theta_1, theta_01 -> Gamma posterior
 *          3. theta_{0,1} | theta_1, W_1 -> Normal posterior
 *
 *          Total iterations: burnin + (n_chain - 1) * thinning + 1
 *
 * @param y_                       Numeric vector [n] of observed binomial counts.
 * @param n_trials_                Number of trials per observation.
 * @param burnin_                  Number of burn-in iterations (discarded).
 * @param thinning_                Thinning interval for autocorrelation reduction.
 * @param n_chain_                 Number of retained posterior samples.
 * @param prior_theta01_mean_      Prior mean for theta_{0,1}.
 * @param prior_theta01_prec_      Prior precision for theta_{0,1}.
 * @param prior_prec1_type_        Integer prior kind on 1/W_1 (0 = Gamma on the
 *                                 precision, 1 = Half-t on sqrt(W_1)).
 * @param prior_prec1_shape_       Gamma shape for 1/W_1 (Gamma kind).
 * @param prior_prec1_rate_        Gamma rate for 1/W_1 (Gamma kind).
 * @param prior_prec1_scale_       Half-t scale A_1 > 0 (Half-t kind).
 * @param prior_prec1_df_          Half-t df nu_1 > 0 (Half-t kind; 1 = Half-Cauchy).
 * @param lag_update_              Adaptation frequency (iterations).
 * @param max_step_size_           Maximum proposal step size.
 * @param base_adaptation_rate_    Base adaptation rate.
 * @param decay_exponent_          Adaptation decay exponent.
 * @param target_acceptance_       Target acceptance proportion.
 * @param min_deviation_threshold_ Minimum deviation to trigger adaptation (>= 0).
 * @param return_log_sigma_        Flag to return log_sigma diagnostics.
 * @param return_accept_prop_      Flag to return accept_prop diagnostics.
 * @param verbose_                 Logical: display progress bar (0 = FALSE, 1 = TRUE).
 * @param bar_width_               Integer: width of progress bar in characters (10-120).
 *
 * @return R list with components:
 *         - theta_1:     Matrix [n_chain * n] of state trajectory samples
 *         - theta_01:    Vector [n_chain] of initial state samples
 *         - prec_theta1:      Vector [n_chain] of innovation precision samples
 *         - alpha:       Matrix [n_chain * n] of success probability samples
 *         - log_sigma:   Matrix [n_chain * n] of proposal scales (if requested)
 *         - accept_prop: Matrix [n_chain * n] of acceptance proportions (if requested)
 *
 * @note Complexity: O(n_iter * n) time, O(n) space
 * @note Requires n >= 3 for numerical stability
 * @note Proper RNG state management via GetRNGstate()/PutRNGstate()
 * @note Adaptation threshold: practical default is 1.0/lag_update
 *
 * @warning Each y[t] must satisfy 0 <= y[t] <= n_trials
 * @warning n must not exceed INT_MAX
 * @warning Memory allocation failures terminate R session
 *
 * @see generate_alpha_logit_binomial_locallevel
 * @see generate_precision_theta_p
 * @see generate_theta_01_locallevel
 */
SEXP C_MCMC_logit_binomial_locallevel(SEXP y_,
                                      SEXP n_trials_,
                                      SEXP burnin_,
                                      SEXP thinning_,
                                      SEXP n_chain_,
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

  /* ========== Parse Data Vector and Validate Length ========== */
  double   *y   = REAL(y_);
  R_xlen_t  len = LENGTH(y_);

  /* Enforce minimum sample size for numerical stability */
  if (len < 3) {
    Rf_error("C_MCMC_logit_binomial_locallevel: sample size 'n' must be at least 3, got %lld",
             (long long) len);
  }

  /* Check for integer overflow (R limitation) */
  if (len > INT_MAX) {
    Rf_error("C_MCMC_logit_binomial_locallevel: sample size too large (%lld > %d)",
             (long long) len, INT_MAX);
  }

  int n = (int) len;

  /* ========== Parse Observation Model Parameters ========== */
  double n_trials = REAL(n_trials_)[0];

  /* Validate binomial constraints */
  for (int t = 0; t < n; t++) {
    if (y[t] < 0 || y[t] > n_trials) {
      Rf_error("C_MCMC_logit_binomial_locallevel: y[%d] = %f violates 0 <= y <= n_trials = %f",
               t, y[t], n_trials);
    }
  }

  /* ========== Parse MCMC Control Parameters ========== */
  int burnin   = INTEGER(burnin_)[0];     /* Burn-in iterations to discard */
  int thinning = INTEGER(thinning_)[0];   /* Thinning interval */
  int n_chain  = INTEGER(n_chain_)[0];    /* Number of retained samples */

  /* Compute total iterations needed */
  int n_iter = burnin + (n_chain - 1) * thinning + 1;

  /* ========== Parse Prior Hyperparameters ========== */
  double mean_theta01 = REAL(prior_theta01_mean_)[0]; /* Prior mean for theta_{0,1} */
  double prec_theta01 = REAL(prior_theta01_prec_)[0]; /* Prior precision for theta_{0,1} */
  /* Innovation precision 1/W_1 prior. The R wrapper resolves the "halfcauchy"
   * alias to Half-t with df = 1 and passes finite placeholders for the unused
   * fields, so no field is ever NA regardless of the selected kind. */
  int          prec1_kind = asInteger(prior_prec1_type_);   /* prior on 1/W_1 */
  prec_prior_t prior_W1   = {
    .shape    = REAL(prior_prec1_shape_)[0],
    .rate     = REAL(prior_prec1_rate_)[0],
    .df       = REAL(prior_prec1_df_)[0],
    .hc_scale = REAL(prior_prec1_scale_)[0]
  };
  /* W_1 is the terminal random-walk component (theta_p sampler). Resolve the
   * prior dispatch ONCE, before the Gibbs loop. */
  prec_thetap_step_t update_prec_W1 =
    (prec1_kind == PDM_PREC_PRIOR_HALFT) ? step_prec_thetap_halft
                                         : step_prec_thetap_gamma;

  /* ========== Parse Adaptation Parameters ========== */
  int    lag_update              = INTEGER(lag_update_)[0];
  double max_step_size           = REAL(max_step_size_)[0];
  double base_adaptation_rate    = REAL(base_adaptation_rate_)[0];
  double decay_exponent          = REAL(decay_exponent_)[0];
  double target_acceptance       = REAL(target_acceptance_)[0];
  double min_deviation_threshold = REAL(min_deviation_threshold_)[0];

  /* ========== Parse Diagnostic Output Options ========== */
  int return_log_sigma   = LOGICAL(return_log_sigma_)[0];
  int return_accept_prop = LOGICAL(return_accept_prop_)[0];

  /* ========== Parse Progress Bar Parameters ========== */
  int verbose   = asLogical(verbose_);
  int bar_width = asInteger(bar_width_);

  /* ===== Initiate Progress Bar ===== */
  ProgressBar pb = progress_bar_init(n_iter, bar_width, burnin, thinning, verbose);
  progress_bar_start(&pb);

  /* ========== Allocate Output Storage (Retained Samples Only) ========== */
  SEXP theta_1_samples  = PROTECT(allocMatrix(REALSXP, n_chain, n));
  SEXP theta_01_samples = PROTECT(allocVector(REALSXP, n_chain));
  SEXP prec_theta1_samples   = PROTECT(allocVector(REALSXP, n_chain));
  SEXP alpha_samples    = PROTECT(allocMatrix(REALSXP, n_chain, n));

  /* Conditional allocation for diagnostics */
  SEXP log_sigma_samples   = R_NilValue;
  SEXP accept_prop_samples = R_NilValue;
  int n_outputs = 4;  /* Base outputs: theta_1, theta_01, prec_theta1, alpha */
  int n_protect = 4;  /* Base protection count */

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

  /* ========== Allocate Temporary Buffers (Memory-Efficient O(n) Storage) ========== */
  /* Uses current/previous iteration buffers instead of full trajectory storage */
  double *theta_1_current  = (double *) R_Calloc(n, double);
  double *theta_1_previous = (double *) R_Calloc(n, double);
  double *alpha_current    = (double *) R_Calloc(n, double);

  /* Scalar parameters for current and previous iterations */
  double theta_01_current, theta_01_previous;
  double prec_theta1_current,   prec_theta1_previous;

  /* Half-t auxiliary b = 1/a for the W_1 precision (refreshed in place when its
   * prior is Half-t; left at 0 and never read under the Gamma prior). */
  double aux_W1 = 0.0;

  /* Sliding window buffer for acceptance tracking (optimized) */
  double *theta_1_updated  = (double *) R_Calloc(lag_update * n, double);

  /* Working arrays for CWMH algorithm */
  double *accept_prop      = (double *) R_Calloc(n, double);
  double *log_sigma        = (double *) R_Calloc(n, double);
  double *hat_theta_1      = (double *) R_Calloc(n, double);
  double *theta_1_new      = (double *) R_Calloc(n, double);
  double *log_accept_prob  = (double *) R_Calloc(n, double);

  /* Initialize log_sigma with reasonable starting values */
  for (int t = 0; t < n; t++) {
    log_sigma[t] = log(0.1);  /* Initial proposal sd = 0.1 */
  }

  /* ========== Initialize RNG State ========== */
  GetRNGstate();

  /* ========== Initialize Parameters (Iteration 0) ========== */
  /* Draw initial values from priors to start the Markov chain */
  theta_01_previous = rnorm(mean_theta01, sqrt(1.0 / prec_theta01));
  prec_theta1_previous   = pdm_init_prec_prior(prec1_kind, &prior_W1, &aux_W1);

  /* Initialize theta_1 and alpha with efficient neutral starting values */
  for (int t = 0; t < n; t++) {
    theta_1_previous[t] = 0.0;   /* Zeros for state trajectory */
    alpha_current[t]    = 0.5;   /* Neutral probability for success rates */
  }

  /* ========== Main Gibbs Sampling Loop ========== */
  /* Uses current/previous buffers for memory efficiency.
   * Only retained samples are copied to output (post burn-in, thinned). */
  int chain_idx = 0;

  for (int ii = 1; ii < n_iter; ii++) {

    /* Determine whether to compute alpha transformations */
    int compute_alpha = (ii >= burnin && ((ii - burnin) % thinning) == 0) ? 1 : 0;

    /* ===== Step 1: Sample State Vector theta_1 and Success Probabilities alpha ===== */
    /* Draw theta_1 | y, theta_01, W_1 using component-wise Metropolis-Hastings.
     * Alpha is computed conditionally based on whether this iteration will be retained. */
    generate_alpha_logit_binomial_locallevel(
      theta_1_previous,       /* theta_1_previous: state from previous iteration [n] */
      theta_1_current,        /* theta_1_current: output for current iteration [n] */
      compute_alpha ? alpha_current : NULL,  /* alpha_current: NULL if not retained */
      theta_01_previous,      /* theta_01_previous: initial level from previous iteration */
      prec_theta1_previous,        /* prec_theta1_previous: level precision from previous iteration */
      theta_1_updated,        /* theta_1_updated: sliding window workspace */
      y,                      /* y: observed binomial counts */
      accept_prop,            /* accept_prop: acceptance proportions workspace */
      log_sigma,              /* log_sigma: proposal scale parameters */
      hat_theta_1,            /* hat_theta_1: conditional means workspace */
      theta_1_new,            /* theta_1_new: proposal states workspace */
      log_accept_prob,        /* log_accept_prob: MH log-acceptance ratios */
      lag_update,             /* lag_update: adaptation lag */
      n_trials,               /* n_trials: number of binomial trials */
      n,                      /* n: series length */
      ii,                     /* iter: current iteration */
      max_step_size,          /* max_step_size: proposal cap */
      base_adaptation_rate,   /* base_adaptation_rate: base adaptation weight */
      decay_exponent,         /* decay_exponent: adaptation decay */
      target_acceptance,      /* target_acceptance: desired acceptance rate */
      min_deviation_threshold,/* min_deviation_threshold: adaptation trigger */
      compute_alpha           /* compute_alpha: flag for alpha computation */
    );

    /* ===== Step 2: Sample Innovation Precision 1/W_1 ===== */
    /* Draw 1/W_1 | theta_1, theta_01 from Gamma posterior */
    prec_theta1_current = update_prec_W1(
      theta_01_previous,  /* theta_0p: initial level from previous iteration */
      theta_1_current,    /* theta_p_current: current level trajectory [n] */
      n,                  /* n: number of time points */
      &prior_W1,          /* prior hyperparameters for the resolved kind */
      &aux_W1             /* Half-t auxiliary (updated in place; unused if Gamma) */
    );

    /* ===== Step 3: Sample Initial State theta_{0,1} ===== */
    /* Draw theta_{0,1} | theta_1, W_1 from Normal posterior */
    theta_01_current = generate_theta_01_locallevel(
      theta_1_current,    /* theta_1_current: current level trajectory [n] */
      prec_theta1_current,     /* prec_theta1: current level precision */
      mean_theta01,       /* mean_theta01: prior mean */
      prec_theta01,       /* prec_theta01: prior precision */
      n                   /* n: number of time points */
    );

    /* ===== Store Post-Burn-in Samples with Thinning ===== */
    /* Only copy to output matrices for retained iterations */
    if (compute_alpha) {
      int idx = chain_idx++;

      /* Copy current theta_1 and alpha to output matrices (column-major) */
      for (int t = 0; t < n; t++) {
        REAL(theta_1_samples)[idx + t * n_chain] = theta_1_current[t];
        REAL(alpha_samples)[idx + t * n_chain] = alpha_current[t];

        /* Store diagnostics if requested */
        if (return_log_sigma) {
          REAL(log_sigma_samples)[idx + t * n_chain] = log_sigma[t];
        }
        if (return_accept_prop) {
          REAL(accept_prop_samples)[idx + t * n_chain] = accept_prop[t];
        }
      }

      /* Copy scalar parameters to output vectors */
      REAL(theta_01_samples)[idx] = theta_01_current;
      REAL(prec_theta1_samples)[idx] = prec_theta1_current;
    }

    /* ===== Update Progress Bar ===== */
    if (ii % pb.update_step == 0 || ii == n_iter - 1) {
      progress_bar_update(&pb, ii);
    }

    /* ===== Update Previous Values for Next Iteration ===== */
    /* Efficient element-wise copy for theta_1 trajectory */
    memcpy(theta_1_previous, theta_1_current, n * sizeof(double));
    theta_01_previous = theta_01_current;
    prec_theta1_previous = prec_theta1_current;
  }

  /* ========== Finalize Progress Bar ========== */
  progress_bar_finish(&pb, n_chain);

  /* ========== Restore RNG State ========== */
  PutRNGstate();

  /* ========== Free Temporary Buffers ========== */
  R_Free(theta_1_current);
  R_Free(theta_1_previous);
  R_Free(alpha_current);
  R_Free(theta_1_updated);
  R_Free(accept_prop);
  R_Free(log_sigma);
  R_Free(hat_theta_1);
  R_Free(theta_1_new);
  R_Free(log_accept_prob);

  /* ========== Package Results into Named List ========== */
  SEXP out = PROTECT(allocVector(VECSXP, n_outputs));
  SEXP nms = PROTECT(allocVector(STRSXP, n_outputs));

  int output_idx = 0;

  /* Always include base outputs */
  SET_VECTOR_ELT(out, output_idx, theta_1_samples);
  SET_STRING_ELT(nms, output_idx++, mkChar("theta_1"));

  SET_VECTOR_ELT(out, output_idx, theta_01_samples);
  SET_STRING_ELT(nms, output_idx++, mkChar("theta_01"));

  SET_VECTOR_ELT(out, output_idx, prec_theta1_samples);
  SET_STRING_ELT(nms, output_idx++, mkChar("prec_theta1"));

  SET_VECTOR_ELT(out, output_idx, alpha_samples);
  SET_STRING_ELT(nms, output_idx++, mkChar("alpha"));

  /* Conditionally add diagnostic outputs */
  if (return_log_sigma) {
    SET_VECTOR_ELT(out, output_idx, log_sigma_samples);
    SET_STRING_ELT(nms, output_idx++, mkChar("log_sigma"));
  }
  if (return_accept_prop) {
    SET_VECTOR_ELT(out, output_idx, accept_prop_samples);
    SET_STRING_ELT(nms, output_idx++, mkChar("accept_prop"));
  }

  setAttrib(out, R_NamesSymbol, nms);

  /* Adjust UNPROTECT count: n_protect + 2 (out and nms) */
  UNPROTECT(n_protect + 2);
  return out;
}

//----------------------------------------------------------------------

/**
 * @brief Gibbs sampler for local-level Bernoulli dynamic model with probit link
 *
 * @details Implements the Albert-Chib (1993) latent-variable augmentation to obtain
 *          fully Gibbs updates for Bernoulli observations governed by a local-level
 *          Gaussian state evolution:
 *
 *          **Observation equation:**
 *          y_t ~ Bernoulli(alpha_t), where alpha_t = Phi(theta_{t,1})
 *
 *          **State equation:**
 *          theta_{t,1} = theta_{t-1,1} + u_{t,1}, u_{t,1} ~ N(0, W_1)
 *
 *          **Prior distributions:**
 *          - theta_{0,1} ~ N(mu_{0,1}, sigma_{0,1}^2)
 *          - 1/W_1       ~ Gamma(nu_1, eta_1)
 *
 *          The augmentation introduces latent Gaussian utilities v_t whose signs match y_t,
 *          yielding closed-form conditional distributions for theta_{t,1}, theta_{0,1},
 *          and the innovation precision 1/W_1.
 *
 *          **Optimizations implemented:**
 *          - Memory-efficient current/previous iteration buffers (O(n) storage)
 *          - Conditional alpha computation (skip during burn-in/thinning)
 *          - Single rhs_vector buffer for linear system construction
 *          - Scalar parameter passing to avoid array indexing
 *          - Efficient initialization with zeros and neutral starting values
 *
 *          **Sampling sequence per iteration:**
 *          1. v_t, theta_{t,1}, alpha_t | y_t, theta_{0,1}, 1/W_1 -> Gibbs via augmentation
 *          2. 1/W_1 | theta_{t,1}, theta_{0,1} -> Gamma posterior
 *          3. theta_{0,1} | theta_{t,1}, 1/W_1 -> Normal posterior
 *
 *          Total iterations: burnin + (n_chain - 1) * thinning + 1
 *
 * @param y_                  Numeric vector [n] of Bernoulli observations.
 * @param burnin_             Number of burn-in iterations (discarded).
 * @param thinning_           Thinning interval for autocorrelation reduction.
 * @param n_chain_            Number of retained posterior samples.
 * @param prior_theta01_mean_ Prior mean for theta_{0,1}.
 * @param prior_theta01_prec_ Prior precision for theta_{0,1}.
 * @param prior_prec1_type_   Integer prior kind on 1/W_1 (0 = Gamma, 1 = Half-t on sqrt(W_1)).
 * @param prior_prec1_shape_  Gamma shape for 1/W_1 (Gamma kind).
 * @param prior_prec1_rate_   Gamma rate for 1/W_1 (Gamma kind).
 * @param prior_prec1_scale_  Half-t scale A_1 > 0 (Half-t kind).
 * @param prior_prec1_df_     Half-t df nu_1 > 0 (Half-t kind; 1 = Half-Cauchy).
 * @param verbose_            Logical: display progress bar (0 = FALSE, 1 = TRUE).
 * @param bar_width_          Integer: width of progress bar in characters (10-120).
 *
 * @return R list with components:
 *         - theta_1:  Matrix [n_chain * n] of state trajectory samples
 *         - theta_01: Vector [n_chain] of initial state samples
 *         - prec_theta1:   Vector [n_chain] of innovation precision samples
 *         - alpha:    Matrix [n_chain * n] of Bernoulli probabilities
 *
 * @note Complexity: O(n_iter * n) time, O(n) space
 * @note Requires n >= 3 for numerical stability
 * @note Acceptance rate: Always 1.0 (Gibbs sampling)
 * @note Conditional alpha computation eliminates unnecessary pnorm calls
 *
 * @warning Each y[t] must be either 0 or 1
 * @warning n must not exceed INT_MAX
 *
 * @see Albert & Chib (1993). Bayesian Analysis of Binary and Polychotomous Response Data.
 *      JASA, 88(422), 669-679. https://doi.org/10.1080/01621459.1993.10476321
 * @see generate_alpha_probit_bernoulli_locallevel
 * @see generate_precision_theta_p
 * @see generate_theta_01_locallevel
 */
SEXP C_MCMC_probit_bernoulli_locallevel(SEXP y_,
                                        SEXP burnin_,
                                        SEXP thinning_,
                                        SEXP n_chain_,
                                        SEXP prior_theta01_mean_,
                                        SEXP prior_theta01_prec_,
                                        SEXP prior_prec1_type_,
                                        SEXP prior_prec1_shape_,
                                        SEXP prior_prec1_rate_,
                                        SEXP prior_prec1_scale_,
                                        SEXP prior_prec1_df_,
                                        SEXP verbose_,
                                        SEXP bar_width_) {

  /* ========== Parse Data Vector and Validate Length ========== */
  double   *y   = REAL(y_);
  R_xlen_t  len = LENGTH(y_);

  /* Enforce minimum sample size for numerical stability */
  if (len < 3) {
    Rf_error("C_MCMC_probit_bernoulli_locallevel: sample size 'n' must be at least 3, got %lld",
             (long long) len);
  }

  /* Check for integer overflow (R limitation) */
  if (len > INT_MAX) {
    Rf_error("C_MCMC_probit_bernoulli_locallevel: sample size too large (%lld > %d)",
             (long long) len, INT_MAX);
  }

  int n = (int) len;

  /* Validate Bernoulli support */
  for (int t = 0; t < n; t++) {
    if (!(y[t] == 0.0 || y[t] == 1.0)) {
      Rf_error("C_MCMC_probit_bernoulli_locallevel: y[%d] = %f must be 0 or 1", t, y[t]);
    }
  }

  /* ========== Parse MCMC Control Parameters ========== */
  int burnin   = INTEGER(burnin_)[0];     /* Burn-in iterations to discard */
  int thinning = INTEGER(thinning_)[0];   /* Thinning interval */
  int n_chain  = INTEGER(n_chain_)[0];    /* Number of retained samples */

  /* Compute total iterations needed */
  int n_iter = burnin + (n_chain - 1) * thinning + 1;

  /* ========== Parse Prior Hyperparameters ========== */
  double mean_theta01 = REAL(prior_theta01_mean_)[0]; /* Prior mean for theta_{0,1} */
  double prec_theta01 = REAL(prior_theta01_prec_)[0]; /* Prior precision for theta_{0,1} */
  /* Innovation precision 1/W_1 prior. The R wrapper resolves the "halfcauchy"
   * alias to Half-t with df = 1 and passes finite placeholders for the unused
   * fields, so no field is ever NA regardless of the selected kind. */
  int          prec1_kind = asInteger(prior_prec1_type_);   /* prior on 1/W_1 */
  prec_prior_t prior_W1   = {
    .shape    = REAL(prior_prec1_shape_)[0],
    .rate     = REAL(prior_prec1_rate_)[0],
    .df       = REAL(prior_prec1_df_)[0],
    .hc_scale = REAL(prior_prec1_scale_)[0]
  };
  /* W_1 is the terminal random-walk component (theta_p sampler). Resolve the
   * prior dispatch ONCE, before the Gibbs loop. */
  prec_thetap_step_t update_prec_W1 =
    (prec1_kind == PDM_PREC_PRIOR_HALFT) ? step_prec_thetap_halft
                                         : step_prec_thetap_gamma;

  /* ========== Parse Progress Bar Parameters ========== */
  int verbose   = asLogical(verbose_);
  int bar_width = asInteger(bar_width_);

  /* ===== Initiate Progress Bar ===== */
  ProgressBar pb = progress_bar_init(n_iter, bar_width, burnin, thinning, verbose);
  progress_bar_start(&pb);

  /* ========== Allocate Output Storage (Retained Samples Only) ========== */
  SEXP theta_1_samples  = PROTECT(allocMatrix(REALSXP, n_chain, n));
  SEXP theta_01_samples = PROTECT(allocVector(REALSXP, n_chain));
  SEXP prec_theta1_samples   = PROTECT(allocVector(REALSXP, n_chain));
  SEXP alpha_samples    = PROTECT(allocMatrix(REALSXP, n_chain, n));
  int n_outputs = 4;
  int n_protect = 4;

  /* ========== Allocate Temporary Buffers (Memory-Efficient O(n) Storage) ========== */
  /* Uses current/previous iteration buffers instead of full trajectory storage */
  double *theta_1_current  = (double *) R_Calloc(n, double);
  double *theta_1_previous = (double *) R_Calloc(n, double);
  double *alpha_current    = (double *) R_Calloc(n, double);

  /* Scalar parameters for current and previous iterations */
  double theta_01_current, theta_01_previous;
  double prec_theta1_current,   prec_theta1_previous;

  /* Half-t auxiliary b = 1/a for the W_1 precision (refreshed in place when its
   * prior is Half-t; left at 0 and never read under the Gamma prior). */
  double aux_W1 = 0.0;

  /* Working array for right-hand side of linear system */
  double *rhs_vector = (double *) R_Calloc(n, double);

  /* ========== Initialize RNG State ========== */
  GetRNGstate();

  /* ========== Initialize Parameters (Iteration 0) ========== */
  /* Draw initial values from priors to start the Markov chain */
  theta_01_previous = rnorm(mean_theta01, sqrt(1.0 / prec_theta01));
  prec_theta1_previous   = pdm_init_prec_prior(prec1_kind, &prior_W1, &aux_W1);

  /* Initialize theta_1 and alpha with efficient neutral starting values */
  for (int t = 0; t < n; t++) {
    theta_1_previous[t] = 0.0;   /* Zeros for state trajectory */
    alpha_current[t]    = 0.5;   /* Neutral probability for success rates */
  }

  /* ========== Main Gibbs Sampling Loop ========== */
  /* Uses current/previous buffers for memory efficiency.
   * Only retained samples are copied to output (post burn-in, thinned). */
  int chain_idx = 0;

  for (int ii = 1; ii < n_iter; ii++) {

    /* Determine whether to compute alpha transformations */
    int compute_alpha = (ii >= burnin && ((ii - burnin) % thinning) == 0) ? 1 : 0;

    /* ===== Step 1: Sample Latent Utilities, theta_1, and alpha (conditionally) ===== */
    /* Draw v_t, theta_1, alpha | y, theta_01, W_1 via Albert-Chib augmentation */
    generate_alpha_probit_bernoulli_locallevel(
      theta_1_previous,       /* theta_1_previous: level from previous iteration [n] */
      theta_1_current,        /* theta_1_current: output for current iteration [n] */
      compute_alpha ? alpha_current : NULL,  /* alpha_current: NULL if not retained */
      theta_01_previous,      /* theta_01_previous: initial level from previous iteration */
      prec_theta1_previous,        /* prec_theta1_previous: level precision from previous iteration */
      y,                      /* y: Bernoulli observations */
      rhs_vector,             /* rhs_vector: solver right-hand side */
      n,                      /* n: number of time points */
      compute_alpha           /* compute_alpha: flag for alpha computation */
    );

    /* ===== Step 2: Sample Innovation Precision 1/W_1 ===== */
    /* Draw 1/W_1 | theta_1, theta_01 from Gamma posterior */
    prec_theta1_current = update_prec_W1(
      theta_01_previous,  /* theta_0p: initial level from previous iteration */
      theta_1_current,    /* theta_p_current: current level trajectory [n] */
      n,                  /* n: number of time points */
      &prior_W1,          /* prior hyperparameters for the resolved kind */
      &aux_W1             /* Half-t auxiliary (updated in place; unused if Gamma) */
    );

    /* ===== Step 3: Sample Initial State theta_{0,1} ===== */
    /* Draw theta_{0,1} | theta_1, W_1 from Normal posterior */
    theta_01_current = generate_theta_01_locallevel(
      theta_1_current,    /* theta_1_current: current level trajectory [n] */
      prec_theta1_current,     /* prec_theta1: current level precision */
      mean_theta01,       /* mean_theta01: prior mean */
      prec_theta01,       /* prec_theta01: prior precision */
      n                   /* n: number of time points */
    );

    /* ===== Store Post-Burn-in Samples with Thinning ===== */
    /* Only copy to output matrices for retained iterations */
    if (compute_alpha) {
      int idx = chain_idx++;

      /* Copy current theta_1 and alpha to output matrices (column-major) */
      for (int t = 0; t < n; t++) {
        REAL(theta_1_samples)[idx + t * n_chain] = theta_1_current[t];
        REAL(alpha_samples)[idx + t * n_chain] = alpha_current[t];
      }

      /* Copy scalar parameters to output vectors */
      REAL(theta_01_samples)[idx] = theta_01_current;
      REAL(prec_theta1_samples)[idx] = prec_theta1_current;
    }

    /* ===== Update Progress Bar ===== */
    if (ii % pb.update_step == 0 || ii == n_iter - 1) {
      progress_bar_update(&pb, ii);
    }

    /* ===== Update Previous Values for Next Iteration ===== */
    /* Efficient element-wise copy for theta_1 trajectory */
    memcpy(theta_1_previous, theta_1_current, n * sizeof(double));
    theta_01_previous = theta_01_current;
    prec_theta1_previous = prec_theta1_current;
  }

  /* ========== Finalize Progress Bar ========== */
  progress_bar_finish(&pb, n_chain);

  /* ========== Restore RNG State ========== */
  PutRNGstate();

  /* ========== Free Temporary Buffers ========== */
  R_Free(theta_1_current);
  R_Free(theta_1_previous);
  R_Free(alpha_current);
  R_Free(rhs_vector);

  /* ========== Package Results into Named List ========== */
  SEXP out = PROTECT(allocVector(VECSXP, n_outputs));
  SEXP nms = PROTECT(allocVector(STRSXP, n_outputs));

  int output_idx = 0;
  SET_VECTOR_ELT(out, output_idx, theta_1_samples);
  SET_STRING_ELT(nms, output_idx++, mkChar("theta_1"));

  SET_VECTOR_ELT(out, output_idx, theta_01_samples);
  SET_STRING_ELT(nms, output_idx++, mkChar("theta_01"));

  SET_VECTOR_ELT(out, output_idx, prec_theta1_samples);
  SET_STRING_ELT(nms, output_idx++, mkChar("prec_theta1"));

  SET_VECTOR_ELT(out, output_idx, alpha_samples);
  SET_STRING_ELT(nms, output_idx++, mkChar("alpha"));

  setAttrib(out, R_NamesSymbol, nms);

  UNPROTECT(n_protect + 2);
  return out;
}
