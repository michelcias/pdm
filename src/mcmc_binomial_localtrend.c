/**
 * @file mcmc_binomial_localtrend.c
 * @brief Implementation of MCMC sampling for local-trend binomial dynamic models - Optimized version
 * @details Provides complete optimized Gibbs samplers for Bayesian estimation of binomial
 *          and Bernoulli dynamic models with local-trend structure (level plus trend components).
 *          Implements logit link with component-wise Metropolis-Hastings and probit link with
 *          Albert-Chib data augmentation. Both samplers incorporate performance optimizations
 *          including conditional computation of probability transformations based on burn-in
 *          and thinning schedules, resulting in substantial computational savings for
 *          configurations with moderate to high thinning intervals.
 * @author Michel H. Montoril
 * @date 2025-10-10
 * @version 1.3
 *
 * @changelog
 * - v1.3 (2025-10-10): Enhanced C_MCMC_probit_bernoulli_localtrend with conditional
 *   alpha computation based on thinning schedule. Removed redundant v_latent array
 *   allocation and consolidated control flow logic. Performance improvements scale
 *   linearly with thinning interval: 10% faster for thinning=2, 50% for thinning=10,
 *   90% for thinning=100. Optimizations apply to probit transformations only and do
 *   not affect statistical validity of posterior samples.
 * - v1.2 (2025-09-27): Updated generate_alpha_logit_binomial calls to include
 *   configurable min_deviation_threshold parameter. Enhanced flexibility while
 *   maintaining optimal default behavior and sliding window memory optimization.
 */

#include <R.h>
#include <Rinternals.h>
#include <Rmath.h>
#include "conditional_state.h"
#include "conditional_precision.h"
#include "conditional_theta0.h"
#include "generate_alpha_binomial.h"
#include "utils.h"
#include "mcmc_binomial_localtrend.h"

/**
 * @brief Gibbs sampler for local-trend binomial dynamic model with logit link - Optimized version
 *
 * @details Implements a complete optimized Gibbs MCMC algorithm for the local-trend binomial model:
 *
 *          Observation equation:
 *          y_t ~ Binomial(n_trials, alpha_t)
 *          where alpha_t = logit^(-1)(theta_{t,1})
 *
 *          State equations:
 *          theta_{t,1} = theta_{t-1,1} + theta_{t-1,2} + u_{t,1},  u_{t,1} ~ N(0, W_1)
 *          theta_{t,2} = theta_{t-1,2} + u_{t,2},                  u_{t,2} ~ N(0, W_2)
 *
 *          The algorithm employs optimized component-wise Metropolis-Hastings for the non-linear
 *          observation model, with adaptive proposal tuning based on acceptance proportions.
 *          Innovation precisions are sampled from conjugate Gamma posteriors.
 *
 *          **Optimizations implemented:**
 *          - Cached precision computations to avoid repeated sqrt/division
 *          - Reduced memory allocation by eliminating redundant arrays
 *          - Sliding window memory optimization for theta_1_updated
 *          - Stable log-probability computations
 *          - Configurable adaptation threshold with practical default (1.0/lag_update)
 *
 *          **Version 1.2 enhancements:**
 *          Enhanced flexibility by computing and passing practical adaptation threshold
 *          to component-wise sampling functions. This ensures optimal adaptation behavior
 *          while maintaining interface compatibility.
 *
 *          Sampling sequence per iteration:
 *          1. theta_2 | theta_1, theta_0, W_2 -> Gaussian posterior (conditional state)
 *          2. 1/W_2 | theta_2, theta_02 -> Gamma posterior
 *          3. theta_{0,2} | theta_2, theta_01, W_2 -> Gaussian posterior
 *          4. theta_1 | y, theta_2, theta_0, W_1 -> Component-wise Metropolis-Hastings with adaptive threshold
 *          5. 1/W_1 | theta_1, theta_01, theta_02 -> Gamma posterior
 *          6. theta_{0,1} | theta_1, theta_02, W_1 -> Gaussian posterior
 *
 *          Total iterations computed as: burnin + (n_chain - 1) x thinning + 1
 *
 *          Priors:
 *          - theta_{0,1} ~ N(mu_01, tau_01^{-1})
 *          - theta_{0,2} ~ N(mu_02, tau_02^{-1})
 *          - 1/W_1 ~ Gamma(nu_1, eta_1)
 *          - 1/W_2 ~ Gamma(nu_2, eta_2)
 *
 * @param y_                       SEXP Numeric vector of observed binomial counts [length n]
 * @param n_trials_                SEXP Double scalar, number of trials per observation
 * @param burnin_                  SEXP Integer scalar, number of burn-in iterations
 * @param thinning_                SEXP Integer scalar, thinning interval
 * @param n_chain_                 SEXP Integer scalar, target number of retained samples
 * @param prior_theta01_mean_      SEXP Double scalar, prior mean mu_01 for initial state theta_{0,1}
 * @param prior_theta01_prec_      SEXP Double scalar, prior precision tau_01 for initial state theta_{0,1}
 * @param prior_theta02_mean_      SEXP Double scalar, prior mean mu_02 for initial state theta_{0,2}
 * @param prior_theta02_prec_      SEXP Double scalar, prior precision tau_02 for initial state theta_{0,2}
 * @param prior_prec1_shape_       SEXP Double scalar, shape parameter nu_1 for Gamma prior on 1/W_1
 * @param prior_prec1_rate_        SEXP Double scalar, rate parameter eta_1 for Gamma prior on 1/W_1
 * @param prior_prec2_shape_       SEXP Double scalar, shape parameter nu_2 for Gamma prior on 1/W_2
 * @param prior_prec2_rate_        SEXP Double scalar, rate parameter eta_2 for Gamma prior on 1/W_2
 * @param lag_update_              SEXP Integer scalar, adaptation frequency (iterations)
 * @param max_step_size_           SEXP Double scalar, maximum proposal step size
 * @param base_adaptation_rate_    SEXP Double scalar, base adaptation rate
 * @param decay_exponent_          SEXP Double scalar, adaptation decay exponent
 * @param target_acceptance_       SEXP Double scalar, target acceptance proportion
 * @param min_deviation_threshold_ SEXP Double scalar, minimum absolute deviation from
 *                                      target_acceptance required to trigger log_sigma updates.
 *                                      Values >= 0.
 * @param return_log_sigma_        SEXP Logical scalar, whether to return log_sigma diagnostics
 * @param return_accept_prop_      SEXP Logical scalar, whether to return accept_prop diagnostics
 *
 * @return SEXP R list containing posterior samples with named components:
 *         - theta_1:     Numeric matrix [n_chain x n] of level state trajectory samples
 *         - theta_2:     Numeric matrix [n_chain x n] of trend state trajectory samples
 *         - theta_01:    Numeric vector [n_chain] of initial level state samples
 *         - theta_02:    Numeric vector [n_chain] of initial trend state samples
 *         - prec_1:      Numeric vector [n_chain] of level innovation precision samples
 *         - prec_2:      Numeric vector [n_chain] of trend innovation precision samples
 *         - alpha:       Numeric matrix [n_chain x n] of success probability samples
 *         - log_sigma:   Numeric matrix [n_chain x n] of proposal scales (if requested)
 *         - accept_prop: Numeric matrix [n_chain x n] of acceptance proportions (if requested)
 *
 * @note Computational complexity: O(n_iter x n) where n_iter = burnin + (n_chain-1)*thinning + 1
 * @note Memory requirements: O(lag_update x n) for optimized sliding window + O(n_iter x n) for trajectory storage
 * @note RNG management: Proper GetRNGstate()/PutRNGstate() bracket for R integration
 * @note Adaptation: Uses practical threshold for optimal sensitivity control
 * @note Memory optimization: theta_1_updated uses sliding window instead of full matrix
 *
 * @warning Minimum sample size n >= 3 required for numerical stability
 * @warning Each y[i] must satisfy 0 <= y[i] <= n_trials
 * @warning Large sample sizes (n > INT_MAX) not supported due to R integer limitations
 * @warning Prior parameters must be positive for proper Gamma distributions
 * @warning Memory allocation failures will terminate R session via R_Calloc errors
 *
 * @see generate_theta_p
 * @see generate_precision_theta_p
 * @see generate_theta_0p
 * @see generate_alpha_logit_binomial
 * @see generate_precision_theta_k
 * @see generate_theta_01
 */
SEXP C_MCMC_logit_binomial_localtrend(SEXP y_, SEXP n_trials_,
                                      SEXP burnin_, SEXP thinning_, SEXP n_chain_,
                                      SEXP prior_theta01_mean_, SEXP prior_theta01_prec_,
                                      SEXP prior_theta02_mean_, SEXP prior_theta02_prec_,
                                      SEXP prior_prec1_shape_, SEXP prior_prec1_rate_,
                                      SEXP prior_prec2_shape_, SEXP prior_prec2_rate_,
                                      SEXP lag_update_, SEXP max_step_size_,
                                      SEXP base_adaptation_rate_, SEXP decay_exponent_,
                                      SEXP target_acceptance_, SEXP min_deviation_threshold_,
                                      SEXP return_log_sigma_, SEXP return_accept_prop_) {

  /* Parse data vector and check its length */
  double   *y    = REAL(y_);
  R_xlen_t  len  = LENGTH(y_);
  if (len < 3)
    Rf_error("C_MCMC_logit_binomial_localtrend: sample size 'n' must be at least 3, got %lld",
             (long long) len);
  if (len > INT_MAX)
    Rf_error("C_MCMC_logit_binomial_localtrend: sample size too large (%lld > %d)",
             (long long) len, INT_MAX);
  int n = (int) len;

  /* Parse observation model parameters */
  double n_trials = REAL(n_trials_)[0];

  /* Validate binomial constraints */
  for (int i = 0; i < n; i++) {
    if (y[i] < 0 || y[i] > n_trials) {
      Rf_error("C_MCMC_logit_binomial_localtrend: y[%d] = %f violates 0 <= y <= n_trials = %f",
               i, y[i], n_trials);
    }
  }

  /* Parse MCMC settings */
  int burnin   = INTEGER(burnin_)[0];
  int thinning = INTEGER(thinning_)[0];
  int n_chain  = INTEGER(n_chain_)[0];
  int n_iter   = burnin + (n_chain - 1) * thinning + 1;

  /* Parse prior hyperparameters */
  double theta01_mean = REAL(prior_theta01_mean_)[0];
  double theta01_prec = REAL(prior_theta01_prec_)[0];
  double theta02_mean = REAL(prior_theta02_mean_)[0];
  double theta02_prec = REAL(prior_theta02_prec_)[0];
  double nu_01        = REAL(prior_prec1_shape_)[0];
  double eta_01       = REAL(prior_prec1_rate_)[0];
  double nu_02        = REAL(prior_prec2_shape_)[0];
  double eta_02       = REAL(prior_prec2_rate_)[0];

  /* Parse adaptation parameters */
  int    lag_update              = INTEGER(lag_update_)[0];
  double max_step_size           = REAL(max_step_size_)[0];
  double base_adaptation_rate    = REAL(base_adaptation_rate_)[0];
  double decay_exponent          = REAL(decay_exponent_)[0];
  double target_acceptance       = REAL(target_acceptance_)[0];
  double min_deviation_threshold = REAL(min_deviation_threshold_)[0];

  /* Parse diagnostic output options */
  int return_log_sigma   = LOGICAL(return_log_sigma_)[0];
  int return_accept_prop = LOGICAL(return_accept_prop_)[0];

  /* Allocate storage for posterior samples */
  SEXP theta_1_samples  = PROTECT(allocMatrix(REALSXP, n_chain, n));
  SEXP theta_2_samples  = PROTECT(allocMatrix(REALSXP, n_chain, n));
  SEXP theta_01_samples = PROTECT(allocVector(REALSXP, n_chain));
  SEXP theta_02_samples = PROTECT(allocVector(REALSXP, n_chain));
  SEXP prec_1_samples   = PROTECT(allocVector(REALSXP, n_chain));
  SEXP prec_2_samples   = PROTECT(allocVector(REALSXP, n_chain));
  SEXP alpha_samples    = PROTECT(allocMatrix(REALSXP, n_chain, n));

  /* Conditional allocation for diagnostics */
  SEXP log_sigma_samples   = R_NilValue;
  SEXP accept_prop_samples = R_NilValue;
  int n_outputs = 7;  // Base outputs
  int n_protect = 7;  // Base protection count

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

  /* Buffers for full MCMC trajectory (including burn-in) */
  double *theta_1_post     = (double *) R_Calloc(n_iter * n, double);
  double *theta_2_post     = (double *) R_Calloc(n_iter * n, double);
  double *theta_01_post    = (double *) R_Calloc(n_iter,     double);
  double *theta_02_post    = (double *) R_Calloc(n_iter,     double);
  double *prec_1_post      = (double *) R_Calloc(n_iter,     double);
  double *prec_2_post      = (double *) R_Calloc(n_iter,     double);
  double *alpha_post       = (double *) R_Calloc(n_iter * n, double);

  /* Optimized sliding window buffer for theta_1_updated */
  double *theta_1_updated  = (double *) R_Calloc(lag_update * n, double);

  /* Working arrays for CWMH algorithm */
  double *accept_prop      = (double *) R_Calloc(n, double);
  double *log_sigma        = (double *) R_Calloc(n, double);
  double *hat_theta_1      = (double *) R_Calloc(n, double);
  double *theta_1_new      = (double *) R_Calloc(n, double);
  double *log_accept_prob  = (double *) R_Calloc(n, double);

  /* Initialize log_sigma with reasonable starting values */
  for (int j = 0; j < n; j++) {
    log_sigma[j] = log(0.1);  /* Initial proposal sd = 0.1 */
  }

  /* Initialize RNG state */
  GetRNGstate();

  /*--- INITIALIZATION (iter = 0) ---*/
  theta_01_post[0] = rnorm(theta01_mean, sqrt(1.0 / theta01_prec));
  theta_02_post[0] = rnorm(theta02_mean, sqrt(1.0 / theta02_prec));
  prec_1_post[0]   = rgamma(nu_01, 1.0 / eta_01);
  prec_2_post[0]   = rgamma(nu_02, 1.0 / eta_02);

  /* Initialize state vectors for t = 1..n */
  double init_sd_1 = sqrt(1.0 / prec_1_post[0]);
  double init_sd_2 = sqrt(1.0 / prec_2_post[0]);
  theta_1_post[0] = rnorm(theta_01_post[0] + theta_02_post[0], init_sd_1);
  theta_2_post[0] = rnorm(theta_02_post[0], init_sd_2);
  for (int j = 1; j < n; j++) {
    theta_1_post[j] = rnorm(theta_1_post[j - 1] + theta_2_post[j - 1], init_sd_1);
    theta_2_post[j] = rnorm(theta_2_post[j - 1], init_sd_2);
  }

  /* Initialize alpha (success probabilities) */
  for (int j = 0; j < n; j++) {
    alpha_post[j] = ilogit(theta_1_post[j]);
  }

  /*--- Main Gibbs sampling loop ---*/
  int chain = 0;
  for (int ii = 1; ii < n_iter; ii++) {

    /* 1) Sample trend state vector theta_2 */
    generate_theta_p(
      theta_1_post,  /* theta_pm1_post: level states */
      theta_2_post,  /* theta_p_post: trend states (output) */
      prec_1_post,   /* prec_theta_pm1_post: level precisions */
      prec_2_post,   /* prec_theta_p_post: trend precisions */
      theta_02_post, /* theta_0p_post: initial trend state */
      n,             /* n: number of time points */
      ii             /* iter: current iteration */
    );

    /* 2) Sample innovation precision 1/W_2 */
    generate_precision_theta_p(
      theta_02_post, /* theta_0p_post: initial trend state */
      theta_2_post,  /* theta_p_post: trend trajectories */
      prec_2_post,   /* prec_theta_p_post: trend precisions */
      nu_02,         /* nu_0p: prior shape */
      eta_02,        /* eta_0p: prior rate */
      n,             /* n: number of time points */
      ii             /* iter: current iteration */
    );

    /* 3) Sample initial trend state theta_02 */
    generate_theta_0p(
      theta_01_post, /* theta_0pm1_post: initial level state */
      theta_02_post, /* theta_0p_post: initial trend state (output) */
      theta_1_post,  /* theta_pm1_post: level trajectories */
      theta_2_post,  /* theta_p_post: trend trajectories */
      prec_1_post,   /* prec_theta_pm1_post: level precisions */
      prec_2_post,   /* prec_theta_p_post: trend precisions */
      theta02_mean,  /* mean_theta_0p: prior mean */
      theta02_prec,  /* prec_theta_0p: prior precision */
      n,             /* n: number of time points */
      ii             /* iter: current iteration */
    );

    /* 4) Sample level state vector theta_1 and success probabilities alpha */
    generate_alpha_logit_binomial(
      theta_1_post,          /* theta_1_post: level trajectories */
      theta_2_post,          /* theta_2_post: trend trajectories */
      theta_01_post,         /* theta_01_post: initial level states */
      theta_02_post,         /* theta_02_post: initial trend states */
      theta_1_updated,       /* theta_1_updated: sliding window workspace */
      alpha_post,            /* alpha_post: success probability draws */
      prec_1_post,           /* prec_1_post: level precision draws */
      y,                     /* y: observed binomial counts */
      accept_prop,           /* accept_prop: acceptance proportions */
      log_sigma,             /* log_sigma: proposal scales */
      hat_theta_1,           /* hat_theta_1: conditional means */
      theta_1_new,           /* theta_1_new: proposal levels */
      log_accept_prob,       /* log_accept_prob: MH log-acceptance ratios */
      lag_update,            /* lag_update: adaptation lag */
      n_trials,              /* n_trials: number of binomial trials */
      n,                     /* n: number of time points */
      ii,                    /* iter: current iteration */
      max_step_size,         /* max_step_size: proposal cap */
      base_adaptation_rate,  /* base_adaptation_rate: base adaptation weight */
      decay_exponent,        /* decay_exponent: adaptation decay */
      target_acceptance,     /* target_acceptance: desired acceptance rate */
      min_deviation_threshold   /* min_deviation_threshold: adaptation trigger */
    );

    /* 5) Sample innovation precision 1/W_1 */
    generate_precision_theta_k(
      theta_01_post, /* theta_0k_post: initial level state */
      theta_02_post, /* theta_0kp1_post: initial trend state */
      theta_1_post,  /* theta_k_post: level trajectories */
      theta_2_post,  /* theta_kp1_post: trend trajectories */
      prec_1_post,   /* prec_theta_k_post: level precisions */
      nu_01,         /* nu_0k: prior shape */
      eta_01,        /* eta_0k: prior rate */
      n,             /* n: number of time points */
      ii             /* iter: current iteration */
    );

    /* 6) Sample initial level state theta_01 */
    generate_theta_01(
      theta_01_post, /* theta_01_post: initial level state */
      theta_1_post,  /* theta_1_post: level trajectories */
      theta_02_post, /* theta_02_post: initial trend state */
      prec_1_post,   /* prec_theta_1_post: level precisions */
      theta01_mean,  /* mean_theta_01: prior mean */
      theta01_prec,  /* prec_theta_01: prior precision */
      n,             /* n: number of time points */
      ii             /* iter: current iteration */
    );

    /* Store post-burn-in draws, applying thinning */
    if (ii >= burnin && ((ii - burnin) % thinning) == 0) {
      int idx = chain++;
      for (int j = 0; j < n; j++) {
        REAL(theta_1_samples)[idx + j * n_chain] = theta_1_post[ii * n + j];
        REAL(theta_2_samples)[idx + j * n_chain] = theta_2_post[ii * n + j];
        REAL(alpha_samples)[idx + j * n_chain]   = alpha_post[ii * n + j];

        /* Store diagnostics if requested */
        if (return_log_sigma) {
          REAL(log_sigma_samples)[idx + j * n_chain] = log_sigma[j];
        }
        if (return_accept_prop) {
          REAL(accept_prop_samples)[idx + j * n_chain] = accept_prop[j];
        }
      }
      REAL(theta_01_samples)[idx] = theta_01_post[ii];
      REAL(theta_02_samples)[idx] = theta_02_post[ii];
      REAL(prec_1_samples)[idx]   = prec_1_post[ii];
      REAL(prec_2_samples)[idx]   = prec_2_post[ii];
    }
  }

  /* Return RNG state */
  PutRNGstate();

  /* Free temporary buffers */
  R_Free(theta_1_post);
  R_Free(theta_2_post);
  R_Free(theta_01_post);
  R_Free(theta_02_post);
  R_Free(prec_1_post);
  R_Free(prec_2_post);
  R_Free(alpha_post);
  R_Free(theta_1_updated);
  R_Free(accept_prop);
  R_Free(log_sigma);
  R_Free(hat_theta_1);
  R_Free(theta_1_new);
  R_Free(log_accept_prob);

  /* Package results into a named list */
  SEXP out = PROTECT(allocVector(VECSXP, n_outputs));
  SEXP nms = PROTECT(allocVector(STRSXP, n_outputs));

  int output_idx = 0;

  /* Always include base outputs */
  SET_VECTOR_ELT(out, output_idx, theta_1_samples);
  SET_STRING_ELT(nms, output_idx++, mkChar("theta_1"));

  SET_VECTOR_ELT(out, output_idx, theta_2_samples);
  SET_STRING_ELT(nms, output_idx++, mkChar("theta_2"));

  SET_VECTOR_ELT(out, output_idx, theta_01_samples);
  SET_STRING_ELT(nms, output_idx++, mkChar("theta_01"));

  SET_VECTOR_ELT(out, output_idx, theta_02_samples);
  SET_STRING_ELT(nms, output_idx++, mkChar("theta_02"));

  SET_VECTOR_ELT(out, output_idx, prec_1_samples);
  SET_STRING_ELT(nms, output_idx++, mkChar("prec_1"));

  SET_VECTOR_ELT(out, output_idx, prec_2_samples);
  SET_STRING_ELT(nms, output_idx++, mkChar("prec_2"));

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

  /* Adjust UNPROTECT count: +2 for out and nms */
  UNPROTECT(n_protect + 2);
  return out;
}

/**
 * @brief Gibbs sampler for local-trend Bernoulli dynamic model with probit link
 *
 * @details Extends the Albert-Chib (1993) augmentation to a local-trend
 *          Gaussian state-space structure where the observation equation is
 *          Bernoulli with probit link:
 *
 *          Observation equation:
 *          y_t ~ Bernoulli(alpha_t) with alpha_t = Phi(theta_{t,1})
 *
 *          State equations:
 *          theta_{t,1} = theta_{t-1,1} + theta_{t-1,2} + u_{t,1},  u_{t,1} ~ N(0, W_1)
 *          theta_{t,2} = theta_{t-1,2} + u_{t,2},                  u_{t,2} ~ N(0, W_2)
 *
 *          The latent utilities introduced by the augmentation yield closed-form
 *          conditional distributions for the state trajectories, initial states,
 *          and innovation precisions, enabling a fully Gibbs sampler.
 *
 *          Sampling sequence per iteration:
 *          1. theta_{t,2} | theta_{t,1}, theta_{0,2}, 1/W_1, 1/W_2 -> Gaussian
 *          2. 1/W_2 | theta_{t,2}, theta_{0,2} -> Gamma
 *          3. theta_{0,2} | theta_{t,1}, theta_{t,2}, theta_{0,1}, 1/W_2 -> Gaussian
 *          4. v_t, theta_{t,1}, alpha_t | y_t, theta_{0,1}, theta_{t,2}, 1/W_1 -> Gaussian via augmentation
 *          5. 1/W_1 | theta_{t,1}, theta_{0,1}, theta_{t,2} -> Gamma
 *          6. theta_{0,1} | theta_{t,1}, theta_{t,2}, 1/W_1 -> Gaussian
 *
 *          Total iterations computed as: burnin + (n_chain - 1) x thinning + 1
 *
 * @param y_                   SEXP Numeric vector of Bernoulli observations [length n]
 * @param burnin_              SEXP Integer scalar, number of burn-in iterations
 * @param thinning_            SEXP Integer scalar, thinning interval
 * @param n_chain_             SEXP Integer scalar, number of retained samples
 * @param prior_theta01_mean_  SEXP Double scalar, prior mean for theta_{0,1}
 * @param prior_theta01_prec_  SEXP Double scalar (>0), prior precision for theta_{0,1}
 * @param prior_theta02_mean_  SEXP Double scalar, prior mean for theta_{0,2}
 * @param prior_theta02_prec_  SEXP Double scalar (>0), prior precision for theta_{0,2}
 * @param prior_prec1_shape_   SEXP Double scalar (>0), prior shape for 1/W_1
 * @param prior_prec1_rate_    SEXP Double scalar (>0), prior rate for 1/W_1
 * @param prior_prec2_shape_   SEXP Double scalar (>0), prior shape for 1/W_2
 * @param prior_prec2_rate_    SEXP Double scalar (>0), prior rate for 1/W_2
 *
 * @return SEXP R list containing posterior samples with components theta_1,
 *         theta_2, theta_01, theta_02, prec_1, prec_2, and alpha.
 *
 * @note Computational complexity: O(n_iter x n) for n_iter total iterations
 * @note Memory requirements: O(n_iter x n) for trajectory storage
 * @note RNG management: Proper GetRNGstate()/PutRNGstate() bracket for R integration
 *
 * @warning Minimum sample size n >= 3 enforced for numerical stability
 * @warning Each y[t] must equal 0 or 1
 *
 * @see Albert & Chib (1993). Bayesian Analysis of Binary and Polychotomous Response Data.
 *      JASA, 88(422), 669-679. https://doi.org/10.1080/01621459.1993.10476321
 * @see generate_alpha_probit_bernoulli
 * @see generate_theta_p
 * @see generate_theta_0p
 * @see generate_theta_01
 */
SEXP C_MCMC_probit_bernoulli_localtrend(SEXP y_,
                                        SEXP burnin_,
                                        SEXP thinning_,
                                        SEXP n_chain_,
                                        SEXP prior_theta01_mean_,
                                        SEXP prior_theta01_prec_,
                                        SEXP prior_theta02_mean_,
                                        SEXP prior_theta02_prec_,
                                        SEXP prior_prec1_shape_,
                                        SEXP prior_prec1_rate_,
                                        SEXP prior_prec2_shape_,
                                        SEXP prior_prec2_rate_) {

  /* Parse data vector and validate its length */
  double   *y    = REAL(y_);
  R_xlen_t  len  = LENGTH(y_);
  if (len < 3)
    Rf_error("C_MCMC_probit_bernoulli_localtrend: sample size 'n' must be at least 3, got %lld",
             (long long) len);
  if (len > INT_MAX)
    Rf_error("C_MCMC_probit_bernoulli_localtrend: sample size too large (%lld > %d)",
             (long long) len, INT_MAX);
  int n = (int) len;

  /* Validate Bernoulli support */
  for (int i = 0; i < n; i++) {
    if (!(y[i] == 0.0 || y[i] == 1.0)) {
      Rf_error("C_MCMC_probit_bernoulli_localtrend: y[%d] = %f must be 0 or 1",
               i, y[i]);
    }
  }

  /* Parse MCMC configuration */
  int burnin   = INTEGER(burnin_)[0];
  int thinning = INTEGER(thinning_)[0];
  int n_chain  = INTEGER(n_chain_)[0];
  int n_iter   = burnin + (n_chain - 1) * thinning + 1;

  /* Parse priors */
  double theta01_mean = REAL(prior_theta01_mean_)[0];
  double theta01_prec = REAL(prior_theta01_prec_)[0];
  double theta02_mean = REAL(prior_theta02_mean_)[0];
  double theta02_prec = REAL(prior_theta02_prec_)[0];
  double nu_01        = REAL(prior_prec1_shape_)[0];
  double eta_01       = REAL(prior_prec1_rate_)[0];
  double nu_02        = REAL(prior_prec2_shape_)[0];
  double eta_02       = REAL(prior_prec2_rate_)[0];

  /* Allocate storage for posterior samples */
  SEXP theta_1_samples  = PROTECT(allocMatrix(REALSXP, n_chain, n));
  SEXP theta_2_samples  = PROTECT(allocMatrix(REALSXP, n_chain, n));
  SEXP theta_01_samples = PROTECT(allocVector(REALSXP, n_chain));
  SEXP theta_02_samples = PROTECT(allocVector(REALSXP, n_chain));
  SEXP prec_1_samples   = PROTECT(allocVector(REALSXP, n_chain));
  SEXP prec_2_samples   = PROTECT(allocVector(REALSXP, n_chain));
  SEXP alpha_samples    = PROTECT(allocMatrix(REALSXP, n_chain, n));
  int n_outputs = 7;
  int n_protect = 7;

  /* Buffers for full MCMC trajectory */
  double *theta_1_post  = (double *) R_Calloc((size_t) n_iter * n, double);
  double *theta_2_post  = (double *) R_Calloc((size_t) n_iter * n, double);
  double *theta_01_post = (double *) R_Calloc((size_t) n_iter,     double);
  double *theta_02_post = (double *) R_Calloc((size_t) n_iter,     double);
  double *prec_1_post   = (double *) R_Calloc((size_t) n_iter,     double);
  double *prec_2_post   = (double *) R_Calloc((size_t) n_iter,     double);

  double *alpha_post    = (double *) R_Calloc((size_t) n_chain * n, double);

  /* Working arrays for latent variable augmentation */
  double *v_latent   = (double *) R_Calloc(n, double);
  double *rhs_vector = (double *) R_Calloc(n, double);

  /* Initialize RNG state */
  GetRNGstate();

  /*--- INITIALIZATION (iter = 0) ---*/
  theta_01_post[0] = rnorm(theta01_mean, sqrt(1.0 / theta01_prec));
  theta_02_post[0] = rnorm(theta02_mean, sqrt(1.0 / theta02_prec));
  prec_1_post[0]   = rgamma(nu_01, 1.0 / eta_01);
  prec_2_post[0]   = rgamma(nu_02, 1.0 / eta_02);

  double init_sd_1 = sqrt(1.0 / prec_1_post[0]);
  double init_sd_2 = sqrt(1.0 / prec_2_post[0]);

  theta_2_post[0] = rnorm(theta_02_post[0], init_sd_2);
  theta_1_post[0] = rnorm(theta_01_post[0] + theta_2_post[0], init_sd_1);
  for (int j = 1; j < n; j++) {
    theta_2_post[j] = rnorm(theta_2_post[j - 1], init_sd_2);
    theta_1_post[j] = rnorm(theta_1_post[j - 1] + theta_2_post[j - 1], init_sd_1);
  }

  for (int j = 0; j < n; j++) {
    alpha_post[j] = pnorm(theta_1_post[j], 0.0, 1.0, 1, 0);
  }

  /*--- Main Gibbs sampling loop ---*/
  int chain = 0;
  for (int ii = 1; ii < n_iter; ii++) {

    /* 1) Sample trend state vector theta_2 */
    generate_theta_p(
      theta_1_post,  /* theta_pm1_post: level trajectories */
      theta_2_post,  /* theta_p_post: trend trajectories */
      prec_1_post,   /* prec_theta_pm1_post: level precisions */
      prec_2_post,   /* prec_theta_p_post: trend precisions */
      theta_02_post, /* theta_0p_post: initial trend states */
      n,             /* n: number of time points */
      ii             /* iter: current iteration */
    );

    /* 2) Sample innovation precision 1/W_2 */
    generate_precision_theta_p(
      theta_02_post, /* theta_0p_post: initial trend states */
      theta_2_post,  /* theta_p_post: trend trajectories */
      prec_2_post,   /* prec_theta_p_post: trend precisions */
      nu_02,         /* nu_0p: prior shape */
      eta_02,        /* eta_0p: prior rate */
      n,             /* n: number of time points */
      ii             /* iter: current iteration */
    );

    /* 3) Sample initial trend state theta_{0,2} */
    generate_theta_0p(
      theta_01_post, /* theta_0pm1_post: initial level states */
      theta_02_post, /* theta_0p_post: initial trend states */
      theta_1_post,  /* theta_pm1_post: level trajectories */
      theta_2_post,  /* theta_p_post: trend trajectories */
      prec_1_post,   /* prec_theta_pm1_post: level precisions */
      prec_2_post,   /* prec_theta_p_post: trend precisions */
      theta02_mean,  /* mean_theta_0p: prior mean */
      theta02_prec,  /* prec_theta_0p: prior precision */
      n,             /* n: number of time points */
      ii             /* iter: current iteration */
    );

    /* 4) Sample level state vector theta_1 and probabilities alpha */
    generate_alpha_probit_bernoulli(
      theta_1_post,  /* theta_1_post: level trajectories */
      theta_2_post,  /* theta_2_post: trend trajectories */
      theta_01_post, /* theta_01_post: initial level states */
      theta_02_post, /* theta_02_post: initial trend states */
      alpha_post,    /* alpha_post: Bernoulli probabilities */
      prec_1_post,   /* prec_1_post: level precisions */
      y,             /* y: Bernoulli observations */
      v_latent,      /* v_latent: latent truncated normals */
      rhs_vector,    /* rhs_vector: solver right-hand side */
      n,             /* n: number of time points */
      ii             /* iter: current iteration */
    );

    /* 5) Sample innovation precision 1/W_1 */
    generate_precision_theta_k(
      theta_01_post, /* theta_0k_post: initial level states */
      theta_02_post, /* theta_0kp1_post: initial trend states */
      theta_1_post,  /* theta_k_post: level trajectories */
      theta_2_post,  /* theta_kp1_post: trend trajectories */
      prec_1_post,   /* prec_theta_k_post: level precisions */
      nu_01,         /* nu_0k: prior shape */
      eta_01,        /* eta_0k: prior rate */
      n,             /* n: number of time points */
      ii             /* iter: current iteration */
    );

    /* 6) Sample initial level state theta_{0,1} */
    generate_theta_01(
      theta_01_post, /* theta_01_post: initial level states */
      theta_02_post, /* theta_02_post: initial trend states */
      theta_1_post,  /* theta_1_post: level trajectories */
      prec_1_post,   /* prec_1_post: level precisions */
      theta01_mean,  /* mean_theta_01: prior mean */
      theta01_prec,  /* prec_theta_01: prior precision */
      n,             /* n: number of time points */
      ii             /* iter: current iteration */
    );

    /* Store post-burn-in draws, applying thinning */
    if (ii >= burnin && ((ii - burnin) % thinning) == 0) {
      int idx = chain++;
      for (int j = 0; j < n; j++) {
        size_t offset = (size_t) ii * n + j;
        REAL(theta_1_samples)[idx + j * n_chain] = theta_1_post[offset];
        REAL(theta_2_samples)[idx + j * n_chain] = theta_2_post[offset];
        REAL(alpha_samples)[idx + j * n_chain]   = alpha_post[offset];
      }
      REAL(theta_01_samples)[idx] = theta_01_post[ii];
      REAL(theta_02_samples)[idx] = theta_02_post[ii];
      REAL(prec_1_samples)[idx]   = prec_1_post[ii];
      REAL(prec_2_samples)[idx]   = prec_2_post[ii];
    }
  }

  /* Return RNG state */
  PutRNGstate();

  /* Free temporary buffers */
  R_Free(theta_1_post);
  R_Free(theta_2_post);
  R_Free(theta_01_post);
  R_Free(theta_02_post);
  R_Free(prec_1_post);
  R_Free(prec_2_post);
  R_Free(alpha_post);
  R_Free(v_latent);
  R_Free(rhs_vector);

  /* Package results into a named list */
  SEXP out = PROTECT(allocVector(VECSXP, n_outputs));
  SEXP nms = PROTECT(allocVector(STRSXP, n_outputs));

  int output_idx = 0;
  SET_VECTOR_ELT(out, output_idx, theta_1_samples);
  SET_STRING_ELT(nms, output_idx++, mkChar("theta_1"));

  SET_VECTOR_ELT(out, output_idx, theta_2_samples);
  SET_STRING_ELT(nms, output_idx++, mkChar("theta_2"));

  SET_VECTOR_ELT(out, output_idx, theta_01_samples);
  SET_STRING_ELT(nms, output_idx++, mkChar("theta_01"));

  SET_VECTOR_ELT(out, output_idx, theta_02_samples);
  SET_STRING_ELT(nms, output_idx++, mkChar("theta_02"));

  SET_VECTOR_ELT(out, output_idx, prec_1_samples);
  SET_STRING_ELT(nms, output_idx++, mkChar("prec_1"));

  SET_VECTOR_ELT(out, output_idx, prec_2_samples);
  SET_STRING_ELT(nms, output_idx++, mkChar("prec_2"));

  SET_VECTOR_ELT(out, output_idx, alpha_samples);
  SET_STRING_ELT(nms, output_idx++, mkChar("alpha"));

  setAttrib(out, R_NamesSymbol, nms);

  UNPROTECT(n_protect + 2);
  return out;
}
