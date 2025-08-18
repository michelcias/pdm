/**
 * @file mcmc_binomial_localtrend.c
 * @brief Implementation of MCMC sampling for local-trend binomial dynamic models
 * @details Provides a complete Gibbs sampler for Bayesian estimation of binomial
 *          dynamic models with logit link and local-trend structure, utilizing
 *          component-wise Metropolis-Hastings for non-linear state sampling.
 * @author Michel H. Montoril
 * @date 2025-08-18
 * @version 1.0
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
 * @brief Gibbs sampler for local-trend binomial dynamic model with logit link
 *
 * @details Implements a complete Gibbs MCMC algorithm for the local-trend binomial model:
 *
 *          Observation equation:
 *          y_t ~ Binomial(n_trials, alpha_t)
 *          where alpha_t = logit^(-1)(theta_{1,t})
 *
 *          State equations:
 *          theta_{1,t} = theta_{1,t-1} + theta_{2,t-1} + u_{1,t},  u_{1,t} ~ N(0, W_1)
 *          theta_{2,t} = theta_{2,t-1} + u_{2,t},                  u_{2,t} ~ N(0, W_2)
 *
 *          The algorithm employs component-wise Metropolis-Hastings for the non-linear
 *          observation model, with adaptive proposal tuning based on acceptance rates.
 *          Innovation precisions are sampled from conjugate Gamma posteriors.
 *
 *          Sampling sequence per iteration:
 *          1. theta_2 | theta_1, theta_0, W_2 -> Gaussian posterior (conditional state)
 *          2. 1/W_2 | theta_2, theta_02 -> Gamma posterior
 *          3. theta_{0,2} | theta_2, theta_01, W_2 -> Gaussian posterior
 *          4. theta_1 | y, theta_2, theta_0, W_1 -> Component-wise Metropolis-Hastings
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
 * @param y_                    SEXP Numeric vector of observed binomial counts [length n]
 * @param n_trials_             SEXP Double scalar, number of trials per observation
 * @param burnin_               SEXP Integer scalar, number of burn-in iterations
 * @param thinning_             SEXP Integer scalar, thinning interval
 * @param n_chain_              SEXP Integer scalar, target number of retained samples
 * @param prior_theta01_mean_   SEXP Double scalar, prior mean mu_01 for initial state theta_{0,1}
 * @param prior_theta01_prec_   SEXP Double scalar, prior precision tau_01 for initial state theta_{0,1}
 * @param prior_theta02_mean_   SEXP Double scalar, prior mean mu_02 for initial state theta_{0,2}
 * @param prior_theta02_prec_   SEXP Double scalar, prior precision tau_02 for initial state theta_{0,2}
 * @param prior_prec1_shape_    SEXP Double scalar, shape parameter nu_1 for Gamma prior on 1/W_1
 * @param prior_prec1_rate_     SEXP Double scalar, rate parameter eta_1 for Gamma prior on 1/W_1
 * @param prior_prec2_shape_    SEXP Double scalar, shape parameter nu_2 for Gamma prior on 1/W_2
 * @param prior_prec2_rate_     SEXP Double scalar, rate parameter eta_2 for Gamma prior on 1/W_2
 * @param lag_update_           SEXP Integer scalar, adaptation frequency (iterations)
 * @param max_step_size_        SEXP Double scalar, maximum proposal step size
 * @param base_adaptation_rate_ SEXP Double scalar, base adaptation rate
 * @param decay_exponent_       SEXP Double scalar, adaptation decay exponent
 * @param target_acceptance_    SEXP Double scalar, target acceptance rate
 * @param return_log_sigma_     SEXP Logical scalar, whether to return log_sigma diagnostics
 * @param return_accrate_       SEXP Logical scalar, whether to return accrate diagnostics
 *
 * @return SEXP R list containing posterior samples with named components:
 *         - theta_1: Numeric matrix [n_chain x n] of level state trajectory samples
 *         - theta_2: Numeric matrix [n_chain x n] of trend state trajectory samples
 *         - theta_01: Numeric vector [n_chain] of initial level state samples
 *         - theta_02: Numeric vector [n_chain] of initial trend state samples
 *         - prec_1: Numeric vector [n_chain] of level innovation precision samples
 *         - prec_2: Numeric vector [n_chain] of trend innovation precision samples
 *         - alpha: Numeric matrix [n_chain x n] of success probability samples
 *         - log_sigma: Numeric matrix [n_chain x n] of proposal scales (if requested)
 *         - accrate: Numeric matrix [n_chain x n] of acceptance rates (if requested)
 *
 * @note Computational complexity: O(n_iter x n) where n_iter = burnin + (n_chain-1)*thinning + 1
 * @note Memory allocation: Requires O(n_iter x n) temporary storage for full MCMC trajectory
 * @note Numerical stability: Uses R's built-in random number generators with proper state management
 * @note Thread safety: Not thread-safe due to shared RNG state; use GetRNGstate()/PutRNGstate()
 * @note Adaptation: Uses diminishing adaptation with sliding window acceptance rates
 *
 * @warning Minimum sample size n >= 3 required for numerical stability
 * @warning Each y[i] must satisfy 0 <= y[i] <= n_trials
 * @warning Large sample sizes (n > INT_MAX) not supported due to R integer limitations
 * @warning Prior parameters must be positive for proper Gamma distributions
 * @warning No convergence diagnostics implemented; user must assess chain convergence
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
                                      SEXP target_acceptance_,
                                      SEXP return_log_sigma_, SEXP return_accrate_) {

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
  int    lag_update           = INTEGER(lag_update_)[0];
  double max_step_size        = REAL(max_step_size_)[0];
  double base_adaptation_rate = REAL(base_adaptation_rate_)[0];
  double decay_exponent       = REAL(decay_exponent_)[0];
  double target_acceptance    = REAL(target_acceptance_)[0];

  /* Parse diagnostic output options */
  int return_log_sigma = LOGICAL(return_log_sigma_)[0];
  int return_accrate   = LOGICAL(return_accrate_)[0];

  /* Allocate storage for posterior samples */
  SEXP theta_1_samples  = PROTECT(allocMatrix(REALSXP, n_chain, n));
  SEXP theta_2_samples  = PROTECT(allocMatrix(REALSXP, n_chain, n));
  SEXP theta_01_samples = PROTECT(allocVector(REALSXP, n_chain));
  SEXP theta_02_samples = PROTECT(allocVector(REALSXP, n_chain));
  SEXP prec_1_samples   = PROTECT(allocVector(REALSXP, n_chain));
  SEXP prec_2_samples   = PROTECT(allocVector(REALSXP, n_chain));
  SEXP alpha_samples    = PROTECT(allocMatrix(REALSXP, n_chain, n));

  /* Conditional allocation for diagnostics */
  SEXP log_sigma_samples = R_NilValue;
  SEXP accrate_samples   = R_NilValue;
  int n_outputs = 7;  // Base outputs
  int n_protect = 7;  // Base protection count

  if (return_log_sigma) {
    log_sigma_samples = PROTECT(allocMatrix(REALSXP, n_chain, n));
    n_outputs++;
    n_protect++;
  }
  if (return_accrate) {
    accrate_samples = PROTECT(allocMatrix(REALSXP, n_chain, n));
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
  double *theta_1_updated  = (double *) R_Calloc(n_iter * n, double);

  /* Working arrays for CWMH algorithm */
  double *accrate          = (double *) R_Calloc(n, double);
  double *log_sigma        = (double *) R_Calloc(n, double);
  double *hat_theta_1      = (double *) R_Calloc(n, double);
  double *theta_1_new      = (double *) R_Calloc(n, double);
  double *log_accept_prob  = (double *) R_Calloc(n, double);
  int    *updated          = (int *)    R_Calloc(n, int);

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
      theta_1_post,      // theta_pm1_post
      theta_2_post,      // theta_p_post (output)
      prec_1_post,       // prec_theta_pm1_post
      prec_2_post,       // prec_theta_p_post
      theta_02_post,     // theta_0p_post
      n,                 // n
      ii                 // iter
    );

    /* 2) Sample innovation precision 1/W_2 */
    generate_precision_theta_p(
      theta_02_post,     // theta_0p_post
      theta_2_post,      // theta_p_post
      prec_2_post,       // prec_theta_p_post (output)
      nu_02,             // nu_0p (prior shape)
      eta_02,            // eta_0p (prior rate)
      n,                 // n
      ii                 // iter
    );

    /* 3) Sample initial trend state theta_02 */
    generate_theta_0p(
      theta_01_post,     // theta_0pm1_post (initial level)
      theta_02_post,     // theta_0p_post (output)
      theta_1_post,      // theta_pm1_post
      theta_2_post,      // theta_p_post
      prec_1_post,       // prec_theta_pm1_post
      prec_2_post,       // prec_theta_p_post
      theta02_mean,      // mean_theta_0p (prior mean)
      theta02_prec,      // prec_theta_0p (prior precision)
      n,                 // n
      ii                 // iter
    );

    /* 4) Sample level state vector theta_1 and success probabilities alpha */
    generate_alpha_logit_binomial(
      theta_1_post,
      theta_2_post,
      theta_01_post,
      theta_02_post,
      theta_1_updated,
      alpha_post,
      prec_1_post,
      y,
      accrate,
      log_sigma,
      hat_theta_1,
      theta_1_new,
      log_accept_prob,
      updated,
      lag_update,
      n_trials,
      n,
      ii,
      max_step_size,
      base_adaptation_rate,
      decay_exponent,
      target_acceptance
    );

    /* 5) Sample innovation precision 1/W_1 */
    generate_precision_theta_k(
      theta_01_post,     // theta_0k_post (initial level)
      theta_02_post,     // theta_0kp1_post (initial trend)
      theta_1_post,      // theta_k_post (level states)
      theta_2_post,      // theta_kp1_post (trend states)
      prec_1_post,       // prec_theta_k_post (output)
      nu_01,             // nu_0k (prior shape)
      eta_01,            // eta_0k (prior rate)
      n,                 // n
      ii                 // iter
    );

    /* 6) Sample initial level state theta_01 */
    generate_theta_01(
      theta_01_post,     // theta_01_post (output)
      theta_02_post,     // theta_02_post (initial trend)
      theta_1_post,      // theta_1_post (level states)
      prec_1_post,       // prec_theta_1_post (level precision)
      theta01_mean,      // mean_theta_01 (prior mean)
      theta01_prec,      // prec_theta_01 (prior precision)
      n,                 // n
      ii                 // iter
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
        if (return_accrate) {
          REAL(accrate_samples)[idx + j * n_chain] = accrate[j];
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
  R_Free(accrate);
  R_Free(log_sigma);
  R_Free(hat_theta_1);
  R_Free(theta_1_new);
  R_Free(log_accept_prob);
  R_Free(updated);

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
  if (return_accrate) {
    SET_VECTOR_ELT(out, output_idx, accrate_samples);
    SET_STRING_ELT(nms, output_idx++, mkChar("accrate"));
  }

  setAttrib(out, R_NamesSymbol, nms);

  /* Adjust UNPROTECT count: +2 for out and nms */
  UNPROTECT(n_protect + 2);
  return out;
}