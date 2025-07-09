#include <R.h>
#include <Rinternals.h>
#include <Rmath.h>
#include "conditional_state.h"
#include "conditional_precision.h"
#include "conditional_theta0.h"
#include "mcmc_localtrend.h"

/**
 * C_MCMC_localtrend: Gibbs sampler for a local-trend dynamic model (p = 2).
 *
 * This function runs a Gibbs MCMC for the polynomial dynamic model with a local trend structure,
 * sampling parameters in the following order:
 *   1) state vector      — generate_theta_p (theta_2)
 *   2) innovation prec.  — generate_precision_theta_p (1/W_2)
 *   3) initial state     — generate_theta_0p (theta_02)
 *   4) state vector      — generate_theta_1 (theta_1)
 *   5) innovation prec.  — generate_precision_theta_k (1/W_1)
 *   6) initial state     — generate_theta_01 (theta_01)
 *   7) data precision    — generate_precision_data (1/V)
 *
 * The model is:
 *   y_t        = theta_{t1} + e_t,                         e_t ~ N(0, V)
 *   theta_{t1} = theta_{(t-1)1} + theta_{(t-1)2} + u_{t1}, u_{t1} ~ N(0, W_1)
 *   theta_{t2} = theta_{(t-1)2} + u_{t2},                  u_{t2} ~ N(0, W_2)
 *
 * Burn‐in and thinning are applied so that exactly n_chain posterior draws are returned.
 *
 * @param y                    Numeric vector of observations (length = n).
 * @param burnin               Integer, number of burn‐in iterations.
 * @param thinning             Integer, thinning interval.
 * @param n_chain              Integer, number of retained posterior samples.
 * @param prior_theta01_mean   Double, prior mean for theta_01.
 * @param prior_theta01_prec   Double, prior precision (1/variance) for theta_01.
 * @param prior_theta02_mean   Double, prior mean for theta_02.
 * @param prior_theta02_prec   Double, prior precision (1/variance) for theta_02.
 * @param prior_prec1_shape    Double, shape parameter of Gamma prior for 1/W_1.
 * @param prior_prec1_rate     Double, rate  parameter of Gamma prior for 1/W_1.
 * @param prior_prec2_shape    Double, shape parameter of Gamma prior for 1/W_2.
 * @param prior_prec2_rate     Double, rate  parameter of Gamma prior for 1/W_2.
 * @param prior_prec_y_shape   Double, shape parameter of Gamma prior for 1/V.
 * @param prior_prec_y_rate    Double, rate  parameter of Gamma prior for 1/V.
 *
 * @return An R list with components:
 *   $theta_1  — numeric matrix [n_chain × n] of state samples for level
 *   $theta_2  — numeric matrix [n_chain × n] of state samples for trend
 *   $theta_01 — numeric vector [length = n_chain] of initial state samples (level)
 *   $theta_02 — numeric vector [length = n_chain] of initial state samples (trend)
 *   $prec_1   — numeric vector [length = n_chain] of innovation precisions (level)
 *   $prec_2   — numeric vector [length = n_chain] of innovation precisions (trend)
 *   $prec_y   — numeric vector [length = n_chain] of data precisions
 */
SEXP C_MCMC_localtrend(SEXP y_, SEXP burnin_, SEXP thinning_, SEXP n_chain_,
                       SEXP prior_theta01_mean_, SEXP prior_theta01_prec_,
                       SEXP prior_theta02_mean_, SEXP prior_theta02_prec_,
                       SEXP prior_prec1_shape_, SEXP prior_prec1_rate_,
                       SEXP prior_prec2_shape_, SEXP prior_prec2_rate_,
                       SEXP prior_prec_y_shape_, SEXP prior_prec_y_rate_) {
  /* Parse data vector and check its length */
  double   *y    = REAL(y_);
  R_xlen_t  len  = LENGTH(y_);
  if (len < 3)
    Rf_error("C_MCMC_localtrend: sample size 'n' must be at least 3, got %lld",
             (long long) len);
  if (len > INT_MAX)
    Rf_error("C_MCMC_localtrend: sample size too large (%lld > %d)",
             (long long) len, INT_MAX);
  int n = (int) len;

  /* Parse MCMC settings */
  int burnin   = INTEGER(burnin_)[0];
  int thinning = INTEGER(thinning_)[0];
  int n_chain  = INTEGER(n_chain_)[0];
  /* Use alternative formula to run just enough iters */
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
  double nu_y         = REAL(prior_prec_y_shape_)[0];
  double eta_y        = REAL(prior_prec_y_rate_)[0];

  /* Allocate storage for posterior samples */
  SEXP theta_1_samples  = PROTECT(allocMatrix(REALSXP, n_chain, n));
  SEXP theta_2_samples  = PROTECT(allocMatrix(REALSXP, n_chain, n));
  SEXP theta_01_samples = PROTECT(allocVector(REALSXP, n_chain));
  SEXP theta_02_samples = PROTECT(allocVector(REALSXP, n_chain));
  SEXP prec_1_samples   = PROTECT(allocVector(REALSXP, n_chain));
  SEXP prec_2_samples   = PROTECT(allocVector(REALSXP, n_chain));
  SEXP prec_y_samples   = PROTECT(allocVector(REALSXP, n_chain));

  /* Buffers for full MCMC trajectory (including burn‐in) */
  double *theta_1_post  = (double *) R_Calloc(n_iter * n, double);
  double *theta_2_post  = (double *) R_Calloc(n_iter * n, double);
  double *theta_01_post = (double *) R_Calloc(n_iter,     double);
  double *theta_02_post = (double *) R_Calloc(n_iter,     double);
  double *prec_1_post   = (double *) R_Calloc(n_iter,     double);
  double *prec_2_post   = (double *) R_Calloc(n_iter,     double);
  double *prec_y_post   = (double *) R_Calloc(n_iter,     double);

  /* Initialize RNG state */
  GetRNGstate();

  /*--- INITIALIZATION (iter = 0) ---*/
  theta_01_post[0] = rnorm(theta01_mean, sqrt(1.0 / theta01_prec));
  theta_02_post[0] = rnorm(theta02_mean, sqrt(1.0 / theta02_prec));
  prec_1_post[0]   = rgamma(nu_01, 1.0 / eta_01);
  prec_2_post[0]   = rgamma(nu_02, 1.0 / eta_02);
  prec_y_post[0]   = rgamma(nu_y,  1.0 / eta_y);

  /* Initialize state vectors for t = 1..n */
  double init_sd_1 = sqrt(1.0 / prec_1_post[0]);
  double init_sd_2 = sqrt(1.0 / prec_2_post[0]);
  theta_1_post[0] = rnorm(theta_01_post[0], init_sd_1);
  theta_2_post[0] = rnorm(theta_02_post[0], init_sd_2);
  for (int j = 1; j < n; j++) {
    theta_1_post[j] = rnorm(theta_1_post[j - 1] + theta_2_post[j - 1], init_sd_1);
    theta_2_post[j] = rnorm(theta_2_post[j - 1], init_sd_2);
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

    /* 4) Sample level state vector theta_1 */
    generate_theta_1(
      y,                 // data (observed values)
      theta_1_post,      // theta_1_post (output)
      theta_2_post,      // theta_2_post (trend states)
      prec_y_post,       // prec_data_post (data precision)
      prec_1_post,       // prec_theta_1_post (level precision)
      theta_01_post,     // theta_01_post (initial level)
      theta_02_post,     // theta_02_post (initial trend)
      n,                 // n
      ii                 // iter
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

    /* 7) Sample data precision 1/V */
    generate_precision_data(
      y,                 // y (observed data)
      theta_1_post,      // theta_1_post (level states)
      prec_y_post,       // prec_y_post (output)
      nu_y,              // nu_y (prior shape)
      eta_y,             // eta_y (prior rate)
      n,                 // n
      ii                 // iter
    );

    /* Store post‐burn‐in draws, applying thinning */
    if (ii >= burnin && ((ii - burnin) % thinning) == 0) {
      int idx = chain++;
      for (int j = 0; j < n; j++) {
        REAL(theta_1_samples)[idx + j * n_chain] = theta_1_post[ii * n + j];
        REAL(theta_2_samples)[idx + j * n_chain] = theta_2_post[ii * n + j];
      }
      REAL(theta_01_samples)[idx] = theta_01_post[ii];
      REAL(theta_02_samples)[idx] = theta_02_post[ii];
      REAL(prec_1_samples)[idx]   = prec_1_post[ii];
      REAL(prec_2_samples)[idx]   = prec_2_post[ii];
      REAL(prec_y_samples)[idx]   = prec_y_post[ii];
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
  R_Free(prec_y_post);

  /* Package results into a named list */
  SEXP out = PROTECT(allocVector(VECSXP, 7));
  SET_VECTOR_ELT(out, 0, theta_1_samples);
  SET_VECTOR_ELT(out, 1, theta_2_samples);
  SET_VECTOR_ELT(out, 2, theta_01_samples);
  SET_VECTOR_ELT(out, 3, theta_02_samples);
  SET_VECTOR_ELT(out, 4, prec_1_samples);
  SET_VECTOR_ELT(out, 5, prec_2_samples);
  SET_VECTOR_ELT(out, 6, prec_y_samples);

  SEXP nms = PROTECT(allocVector(STRSXP, 7));
  SET_STRING_ELT(nms, 0, mkChar("theta_1"));
  SET_STRING_ELT(nms, 1, mkChar("theta_2"));
  SET_STRING_ELT(nms, 2, mkChar("theta_01"));
  SET_STRING_ELT(nms, 3, mkChar("theta_02"));
  SET_STRING_ELT(nms, 4, mkChar("prec_1"));
  SET_STRING_ELT(nms, 5, mkChar("prec_2"));
  SET_STRING_ELT(nms, 6, mkChar("prec_y"));
  setAttrib(out, R_NamesSymbol, nms);

  UNPROTECT(9);
  return out;
}
