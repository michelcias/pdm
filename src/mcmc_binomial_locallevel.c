/**
 * @file mcmc_binomial_locallevel.c
 * @brief Implementation of MCMC sampling for local-level binomial dynamic models
 * @details Provides a complete Gibbs sampler for Bayesian estimation of binomial
 *          dynamic models with logit link and local-level structure, utilizing
 *          component-wise Metropolis-Hastings for non-linear state sampling.
 * @author Michel H. Montoril
 * @date 2025-08-12
 * @version 1.0
 */

#include <R.h>
#include <Rinternals.h>
#include <Rmath.h>
#include "conditional_precision.h"
#include "conditional_theta0.h"
#include "generate_alpha_binomial.h"
#include "utils.h"
#include "mcmc_binomial_locallevel.h"

/**
 * @brief Gibbs sampler for local-level binomial dynamic model with logit link
 *
 * @details Implements a complete Gibbs MCMC algorithm for the local-level binomial model:
 *
 *          Observation equation:
 *          y_t ~ Binomial(n_trials, alpha_t)
 *          where alpha_t = logit^(-1)(theta_{1,t})
 *
 *          State equation:
 *          theta_{1,t} = theta_{1,t-1} + u_{1,t},  u_{1,t} ~ N(0, W_1)
 *
 *          The algorithm employs component-wise Metropolis-Hastings for the non-linear
 *          observation model, with adaptive proposal tuning based on acceptance proportions.
 *          Innovation precision is sampled from conjugate Gamma posterior.
 *
 *          Sampling sequence per iteration:
 *          1. theta_1 | y, theta_0, W_1 -> Component-wise Metropolis-Hastings
 *          2. 1/W_1 | theta_1, theta_0 -> Gamma posterior
 *          3. theta_{0,1} | theta_1, W_1 -> Normal posterior
 *
 *          Total iterations computed as: burnin + (n_chain - 1) x thinning + 1
 *
 * @param y_                    SEXP Numeric vector of observed binomial counts [length n]
 * @param n_trials_             SEXP Double scalar, number of trials per observation
 * @param burnin_               SEXP Integer scalar, number of burn-in iterations
 * @param thinning_             SEXP Integer scalar, thinning interval
 * @param n_chain_              SEXP Integer scalar, target number of retained samples
 * @param prior_theta01_mean_   SEXP Double scalar, prior mean for initial state theta_{0,1}
 * @param prior_theta01_prec_   SEXP Double scalar, prior precision for initial state
 * @param prior_prec1_shape_    SEXP Double scalar, shape parameter for Gamma prior on 1/W_1
 * @param prior_prec1_rate_     SEXP Double scalar, rate parameter for Gamma prior on 1/W_1
 * @param lag_update_           SEXP Integer scalar, adaptation frequency (iterations)
 * @param max_step_size_        SEXP Double scalar, maximum proposal step size
 * @param base_adaptation_rate_ SEXP Double scalar, base adaptation rate
 * @param decay_exponent_       SEXP Double scalar, adaptation decay exponent
 * @param target_acceptance_    SEXP Double scalar, target acceptance proportion
 * @param return_log_sigma_     SEXP Logical scalar, whether to return log_sigma diagnostics
 * @param return_accept_prop_   SEXP Logical scalar, whether to return accept_prop diagnostics
 *
 * @return SEXP R list containing posterior samples with named components:
 *         - theta_1: Numeric matrix [n_chain x n] of state trajectory samples
 *         - theta_01: Numeric vector [n_chain] of initial state samples
 *         - prec_1: Numeric vector [n_chain] of innovation precision samples
 *         - alpha: Numeric matrix [n_chain x n] of success probability samples
 *         - log_sigma: Numeric matrix [n_chain x n] of proposal scales (if requested)
 *         - accept_prop: Numeric matrix [n_chain x n] of acceptance proportions (if requested)
 *
 * @note Computational complexity: O(n_iter x n) for n_iter total iterations
 * @note Memory requirements: O(n_iter x n) for trajectory and adaptation storage
 * @note RNG management: Proper GetRNGstate()/PutRNGstate() bracket for R integration
 * @note Adaptation: Uses diminishing adaptation with sliding window acceptance proportions
 *
 * @warning Minimum sample size n >= 3 enforced for numerical stability
 * @warning Each y[i] must satisfy 0 <= y[i] <= n_trials
 * @warning Memory allocation failures will terminate R session via R_Calloc errors
 *
 * @see generate_alpha_logit_binomial_locallevel
 * @see generate_precision_theta_p
 * @see generate_theta_01_locallevel
 */
SEXP C_MCMC_logit_binomial_locallevel(SEXP y_, SEXP n_trials_,
                                      SEXP burnin_, SEXP thinning_, SEXP n_chain_,
                                      SEXP prior_theta01_mean_, SEXP prior_theta01_prec_,
                                      SEXP prior_prec1_shape_, SEXP prior_prec1_rate_,
                                      SEXP lag_update_, SEXP max_step_size_,
                                      SEXP base_adaptation_rate_, SEXP decay_exponent_,
                                      SEXP target_acceptance_,
                                      SEXP return_log_sigma_, SEXP return_accept_prop_) {

  /* Parse data vector and check its length */
  double   *y    = REAL(y_);
  R_xlen_t  len  = LENGTH(y_);
  if (len < 3)
    Rf_error("C_MCMC_logit_binomial_locallevel: sample size 'n' must be at least 3, got %lld",
             (long long) len);
  if (len > INT_MAX)
    Rf_error("C_MCMC_logit_binomial_locallevel: sample size too large (%lld > %d)",
             (long long) len, INT_MAX);
  int n = (int) len;

  /* Parse observation model parameters */
  double n_trials = REAL(n_trials_)[0];

  /* Validate binomial constraints */
  for (int i = 0; i < n; i++) {
    if (y[i] < 0 || y[i] > n_trials) {
      Rf_error("C_MCMC_logit_binomial_locallevel: y[%d] = %f violates 0 <= y <= n_trials = %f",
               i, y[i], n_trials);
    }
  }

  /* Parse MCMC settings */
  int burnin   = INTEGER(burnin_)[0];
  int thinning = INTEGER(thinning_)[0];
  int n_chain  = INTEGER(n_chain_)[0];
  int n_iter   = burnin + (n_chain - 1) * thinning + 1;

  /* Parse priors */
  double mean_theta01 = REAL(prior_theta01_mean_)[0];
  double prec_theta01 = REAL(prior_theta01_prec_)[0];
  double nu_01        = REAL(prior_prec1_shape_)[0];
  double eta_01       = REAL(prior_prec1_rate_)[0];

  /* Parse adaptation parameters */
  int    lag_update           = INTEGER(lag_update_)[0];
  double max_step_size        = REAL(max_step_size_)[0];
  double base_adaptation_rate = REAL(base_adaptation_rate_)[0];
  double decay_exponent       = REAL(decay_exponent_)[0];
  double target_acceptance    = REAL(target_acceptance_)[0];

  /* Parse diagnostic output options */
  int return_log_sigma = LOGICAL(return_log_sigma_)[0];
  int return_accept_prop = LOGICAL(return_accept_prop_)[0];

  /* Allocate storage for posterior samples */
  SEXP theta_1_samples  = PROTECT(allocMatrix(REALSXP, n_chain, n));
  SEXP theta_01_samples = PROTECT(allocVector(REALSXP, n_chain));
  SEXP prec_1_samples   = PROTECT(allocVector(REALSXP, n_chain));
  SEXP alpha_samples    = PROTECT(allocMatrix(REALSXP, n_chain, n));

  /* Conditional allocation for diagnostics */
  SEXP log_sigma_samples = R_NilValue;
  SEXP accept_prop_samples = R_NilValue;
  int n_outputs = 4;  // Base outputs
  int n_protect = 4;  // Base protection count

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
  double *theta_01_post    = (double *) R_Calloc(n_iter,     double);
  double *prec_1_post      = (double *) R_Calloc(n_iter,     double);
  double *alpha_post       = (double *) R_Calloc(n_iter * n, double);
  double *theta_1_updated  = (double *) R_Calloc(n_iter * n, double);

  /* Working arrays for CWMH algorithm */
  double *accept_prop = (double *) R_Calloc(n, double);
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
  theta_01_post[0] = rnorm(mean_theta01, sqrt(1.0 / prec_theta01));
  prec_1_post[0]   = rgamma(nu_01, 1.0 / eta_01);
  double init_sd   = sqrt(1.0 / prec_1_post[0]);

  /* Initialize theta_1 trajectory */
  theta_1_post[0] = rnorm(theta_01_post[0], init_sd);
  for (int j = 1; j < n; j++) {
    theta_1_post[j] = rnorm(theta_1_post[j - 1], init_sd);
  }

  /* Initialize alpha (success probabilities) */
  for (int j = 0; j < n; j++) {
    alpha_post[j] = ilogit(theta_1_post[j]);
  }

  /*--- Main Gibbs sampling loop ---*/
  int chain = 0;
  for (int ii = 1; ii < n_iter; ii++) {

    /* 1) Sample state vector theta_1 and success probabilities alpha */
    generate_alpha_logit_binomial_locallevel(
      theta_1_post,
      theta_01_post,
      theta_1_updated,
      alpha_post,
      prec_1_post,
      y,
      accept_prop,
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

    /* 2) Sample innovation precision 1/W_1 */
    generate_precision_theta_p(
      theta_01_post,    /* theta_0p */
      theta_1_post,     /* theta_p */
      prec_1_post,      /* output precision 1/W_p */
      nu_01,
      eta_01,
      n,
      ii
    );

    /* 3) Sample initial state theta_01 */
    generate_theta_01_locallevel(
      theta_01_post,
      theta_1_post,
      prec_1_post,
      mean_theta01,
      prec_theta01,
      n,
      ii
    );

    /* Store post-burn-in draws, applying thinning */
    if (ii >= burnin && ((ii - burnin) % thinning) == 0) {
      int idx = chain++;
      for (int j = 0; j < n; j++) {
        REAL(theta_1_samples)[idx + j * n_chain] = theta_1_post[ii * n + j];
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
      REAL(prec_1_samples)[idx]   = prec_1_post[ii];
    }
  }

  /* Return RNG state */
  PutRNGstate();

  /* Free temporary buffers */
  R_Free(theta_1_post);
  R_Free(theta_01_post);
  R_Free(prec_1_post);
  R_Free(alpha_post);
  R_Free(theta_1_updated);
  R_Free(accept_prop);
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

  SET_VECTOR_ELT(out, output_idx, theta_01_samples);
  SET_STRING_ELT(nms, output_idx++, mkChar("theta_01"));

  SET_VECTOR_ELT(out, output_idx, prec_1_samples);
  SET_STRING_ELT(nms, output_idx++, mkChar("prec_1"));

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
