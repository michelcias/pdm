/**
 * @file mcmc_normal_localtrend.c
 * @brief MCMC sampling for Gaussian local-trend dynamic models
 * @author Michel H. Montoril
 * @date 2025-10-11
 * @version 1.0
 *
 * @details This file implements a complete Gibbs sampler for Bayesian estimation of
 *          local-trend polynomial dynamic models with Gaussian observation equations.
 *          The implementation features:
 *          - Memory-efficient O(n) temporary storage using current/previous buffers
 *          - Scalar parameter passing to conditional samplers
 *          - Direct vector operations without iteration indexing
 *          - Efficient state trajectory sampling via forward-backward recursions
 *          - Conjugate posterior updates for all parameters
 *
 *          **Model specification:**
 *          Observation equation: y_t = theta_{t,1} + e_t,  e_t ~ N(0, V)
 *          State equations:
 *            theta_{t,1} = theta_{t-1,1} + theta_{t-1,2} + u_{t,1},  u_{t,1} ~ N(0, W_1)
 *            theta_{t,2} = theta_{t-1,2} + u_{t,2},                  u_{t,2} ~ N(0, W_2)
 *
 *          **Gibbs sampling sequence per iteration:**
 *          1. theta_2 | theta_1, theta_{0,1}, theta_{0,2}, W_1, W_2
 *          2. 1/W_2 | theta_2, theta_{0,2}
 *          3. theta_{0,2} | theta_1, theta_2, theta_{0,1}, W_1, W_2
 *          4. theta_1 | y, theta_2, theta_{0,1}, theta_{0,2}, W_1, V
 *          5. 1/W_1 | theta_1, theta_2, theta_{0,1}, theta_{0,2}
 *          6. theta_{0,1} | theta_1, theta_{0,2}, W_1
 *          7. 1/V | y, theta_1
 */

#include <R.h>
#include <Rinternals.h>
#include <Rmath.h>
#include "conditional_state.h"
#include "conditional_precision.h"
#include "conditional_theta0.h"
#include "mcmc_normal_localtrend.h"

/**
 * @brief Gibbs sampler for local-trend dynamic model with Gaussian observations
 *
 * @details Implements a complete Gibbs MCMC algorithm for the local-trend dynamic model:
 *
 *          **Observation equation:**
 *          y_t = theta_{t,1} + e_t,     e_t ~ N(0, V)
 *
 *          **State equations:**
 *          theta_{t,1} = theta_{t-1,1} + theta_{t-1,2} + u_{t,1},  u_{t,1} ~ N(0, W_1)
 *          theta_{t,2} = theta_{t-1,2} + u_{t,2},                  u_{t,2} ~ N(0, W_2)
 *
 *          The algorithm employs the Markovian structure of polynomial dynamic models
 *          to enable efficient block sampling of state vectors through forward-backward
 *          recursions implemented via tridiagonal precision matrix methods.
 *
 *          **Sampling sequence per iteration:**
 *          1. theta_2 | theta_1, theta_{0,1}, theta_{0,2}, W_1, W_2 → MVN (tridiagonal)
 *          2. 1/W_2 | theta_2, theta_{0,2} → Gamma posterior
 *          3. theta_{0,2} | theta_1, theta_2, theta_{0,1}, W_1, W_2 → Normal posterior
 *          4. theta_1 | y, theta_2, theta_{0,1}, theta_{0,2}, W_1, V → MVN (tridiagonal)
 *          5. 1/W_1 | theta_1, theta_2, theta_{0,1}, theta_{0,2} → Gamma posterior
 *          6. theta_{0,1} | theta_1, theta_{0,2}, W_1 → Normal posterior
 *          7. 1/V | y, theta_1 → Gamma posterior
 *
 *          **Memory efficiency:**
 *          Uses current/previous iteration buffers instead of full trajectory storage,
 *          requiring only O(n) temporary memory regardless of chain length.
 *
 *          **Iteration count:**
 *          Total iterations = burnin + (n_chain - 1) × thinning + 1
 *
 * @param y_                    SEXP Numeric vector of observed time series data [length n].
 *                              Contains observations y_1, ..., y_n.
 * @param burnin_               SEXP Integer scalar, number of burn-in iterations to discard
 *                              for chain convergence. Typical values: 1000-10000.
 * @param thinning_             SEXP Integer scalar, thinning interval to reduce autocorrelation
 *                              in retained samples. Typical values: 1-10.
 * @param n_chain_              SEXP Integer scalar, target number of retained posterior samples.
 *                              Final output will contain exactly n_chain samples.
 * @param prior_theta01_mean_   SEXP Double scalar, prior mean mu_{0,1} for initial level theta_{0,1}.
 *                              Typical value: 0 (vague prior).
 * @param prior_theta01_prec_   SEXP Double scalar, prior precision tau_{0,1} = 1/sigma_{0,1}^2
 *                              for initial level. Typical value: 0.001 (vague prior).
 * @param prior_theta02_mean_   SEXP Double scalar, prior mean mu_{0,2} for initial trend theta_{0,2}.
 *                              Typical value: 0 (vague prior).
 * @param prior_theta02_prec_   SEXP Double scalar, prior precision tau_{0,2} = 1/sigma_{0,2}^2
 *                              for initial trend. Typical value: 0.001 (vague prior).
 * @param prior_prec1_shape_    SEXP Double scalar, shape parameter nu_1 for Gamma(nu_1, eta_1)
 *                              prior on 1/W_1. Typical value: 0.001 (vague prior).
 * @param prior_prec1_rate_     SEXP Double scalar, rate parameter eta_1 for Gamma(nu_1, eta_1)
 *                              prior on 1/W_1. Typical value: 0.001 (vague prior).
 * @param prior_prec2_shape_    SEXP Double scalar, shape parameter nu_2 for Gamma(nu_2, eta_2)
 *                              prior on 1/W_2. Typical value: 0.001 (vague prior).
 * @param prior_prec2_rate_     SEXP Double scalar, rate parameter eta_2 for Gamma(nu_2, eta_2)
 *                              prior on 1/W_2. Typical value: 0.001 (vague prior).
 * @param prior_prec_y_shape_   SEXP Double scalar, shape parameter nu_y for Gamma(nu_y, eta_y)
 *                              prior on 1/V. Typical value: 0.001 (vague prior).
 * @param prior_prec_y_rate_    SEXP Double scalar, rate parameter eta_y for Gamma(nu_y, eta_y)
 *                              prior on 1/V. Typical value: 0.001 (vague prior).
 *
 * @return SEXP R list containing posterior samples with named components:
 *         - theta_1: Numeric matrix [n_chain × n] of level state trajectory samples
 *         - theta_2: Numeric matrix [n_chain × n] of trend state trajectory samples
 *         - theta_01: Numeric vector [n_chain] of initial level state theta_{0,1} samples
 *         - theta_02: Numeric vector [n_chain] of initial trend state theta_{0,2} samples
 *         - prec_1: Numeric vector [n_chain] of level innovation precision 1/W_1 samples
 *         - prec_2: Numeric vector [n_chain] of trend innovation precision 1/W_2 samples
 *         - prec_y: Numeric vector [n_chain] of observation precision 1/V samples
 *
 * @note Computational complexity: O(n_iter × n) for n_iter total iterations.
 * @note Memory requirements: O(n) temporary storage for efficient buffer management.
 * @note RNG management: Proper GetRNGstate()/PutRNGstate() bracket for R integration.
 * @note Initialization: Uses prior-based random initialization for all parameters.
 *
 * @warning Minimum sample size n >= 3 enforced for numerical stability of recursions.
 * @warning Integer overflow protection: n <= INT_MAX due to R's integer limitations.
 * @warning Memory allocation failures will terminate R session via R_Calloc errors.
 * @warning No input validation for prior hyperparameters; negative values may cause crashes.
 *
 * @see generate_theta_p
 * @see generate_precision_theta_p
 * @see generate_theta_0p
 * @see generate_theta_1
 * @see generate_precision_theta_k
 * @see generate_theta_01
 * @see generate_precision_data
 */
SEXP C_MCMC_normal_localtrend(SEXP y_,
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
                              SEXP prior_prec2_rate_,
                              SEXP prior_prec_y_shape_,
                              SEXP prior_prec_y_rate_) {

  /* ========== Parse Data Vector and Validate Length ========== */
  double   *y   = REAL(y_);
  R_xlen_t  len = LENGTH(y_);

  /* Enforce minimum sample size for numerical stability */
  if (len < 3) {
    Rf_error("C_MCMC_normal_localtrend: sample size 'n' must be at least 3, got %lld",
             (long long) len);
  }

  /* Check for integer overflow (R limitation) */
  if (len > INT_MAX) {
    Rf_error("C_MCMC_normal_localtrend: sample size too large (%lld > %d)",
             (long long) len, INT_MAX);
  }

  int n = (int) len;

  /* ========== Parse MCMC Control Parameters ========== */
  int burnin   = INTEGER(burnin_)[0];     /* Burn-in iterations to discard */
  int thinning = INTEGER(thinning_)[0];   /* Thinning interval for autocorrelation reduction */
  int n_chain  = INTEGER(n_chain_)[0];    /* Number of retained samples */

  /* Compute total iterations needed to produce n_chain thinned samples after burn-in */
  int n_iter = burnin + (n_chain - 1) * thinning + 1;

  /* ========== Parse Prior Hyperparameters ========== */
  double mean_theta01 = REAL(prior_theta01_mean_)[0]; /* Prior mean for theta_{0,1} */
  double prec_theta01 = REAL(prior_theta01_prec_)[0]; /* Prior precision for theta_{0,1} */
  double mean_theta02 = REAL(prior_theta02_mean_)[0]; /* Prior mean for theta_{0,2} */
  double prec_theta02 = REAL(prior_theta02_prec_)[0]; /* Prior precision for theta_{0,2} */
  double nu_01        = REAL(prior_prec1_shape_)[0];  /* Gamma shape for 1/W_1 */
  double eta_01       = REAL(prior_prec1_rate_)[0];   /* Gamma rate for 1/W_1 */
  double nu_02        = REAL(prior_prec2_shape_)[0];  /* Gamma shape for 1/W_2 */
  double eta_02       = REAL(prior_prec2_rate_)[0];   /* Gamma rate for 1/W_2 */
  double nu_y         = REAL(prior_prec_y_shape_)[0]; /* Gamma shape for 1/V */
  double eta_y        = REAL(prior_prec_y_rate_)[0];  /* Gamma rate for 1/V */

  /* ========== Allocate Output Storage (Retained Samples Only) ========== */
  SEXP theta_1_samples  = PROTECT(allocMatrix(REALSXP, n_chain, n));
  SEXP theta_2_samples  = PROTECT(allocMatrix(REALSXP, n_chain, n));
  SEXP theta_01_samples = PROTECT(allocVector(REALSXP, n_chain));
  SEXP theta_02_samples = PROTECT(allocVector(REALSXP, n_chain));
  SEXP prec_1_samples   = PROTECT(allocVector(REALSXP, n_chain));
  SEXP prec_2_samples   = PROTECT(allocVector(REALSXP, n_chain));
  SEXP prec_y_samples   = PROTECT(allocVector(REALSXP, n_chain));

  /* ========== Allocate Temporary Buffers (Memory-Efficient O(n) Storage) ========== */
  /* Uses current/previous iteration buffers for efficient memory management.
   * Only requires O(n) space regardless of total chain length. */
  double *theta_1_current  = (double *) R_Calloc(n, double);
  double *theta_1_previous = (double *) R_Calloc(n, double);
  double *theta_2_current  = (double *) R_Calloc(n, double);
  double *theta_2_previous = (double *) R_Calloc(n, double);

  /* Scalar parameters for current and previous iterations */
  double theta_01_current, theta_01_previous;
  double theta_02_current, theta_02_previous;
  double prec_1_current,   prec_1_previous;
  double prec_2_current,   prec_2_previous;
  double prec_y_current,   prec_y_previous;

  /* ========== Initialize RNG State ========== */
  GetRNGstate();

  /* ========== Initialize Parameters (Iteration 0) ========== */
  /* Draw initial values from priors to start the Markov chain */
  theta_01_previous = rnorm(mean_theta01, sqrt(1.0 / prec_theta01));
  theta_02_previous = rnorm(mean_theta02, sqrt(1.0 / prec_theta02));
  prec_1_previous   = rgamma(nu_01, 1.0 / eta_01);
  prec_2_previous   = rgamma(nu_02, 1.0 / eta_02);
  prec_y_previous   = rgamma(nu_y, 1.0 / eta_y);

  /* Initialize theta_2 trajectory via random walk from initial trend */
  double init_sd_2 = sqrt(1.0 / prec_2_previous);
  theta_2_previous[0] = rnorm(theta_02_previous, init_sd_2);
  for (int j = 1; j < n; j++) {
    theta_2_previous[j] = rnorm(theta_2_previous[j - 1], init_sd_2);
  }

  /* Initialize theta_1 trajectory via random walk with trend from initial level */
  double init_sd_1 = sqrt(1.0 / prec_1_previous);
  theta_1_previous[0] = rnorm(theta_01_previous + theta_02_previous, init_sd_1);
  for (int j = 1; j < n; j++) {
    theta_1_previous[j] = rnorm(theta_1_previous[j - 1] + theta_2_previous[j - 1], init_sd_1);
  }

  /* ========== Main Gibbs Sampling Loop ========== */
  /* Uses current/previous buffers for memory efficiency.
   * Only retained samples are copied to output (post burn-in, thinned). */
  int chain_idx = 0;

  for (int ii = 1; ii < n_iter; ii++) {

    /* ===== Step 1: Sample Trend State Vector theta_2 ===== */
    /* Draw theta_2 | theta_1, previous parameters from multivariate Normal
     * with tridiagonal precision matrix (O(n) via Cholesky).
     * Uses theta_1 from previous iteration for computing differences. */
    generate_theta_p(
      theta_1_previous,   /* theta_{p-1}: level from previous iteration [n] */
      theta_2_current,    /* output: current iteration theta_2 [n] */
      prec_1_previous,    /* scalar: level precision from previous iteration */
      prec_2_previous,    /* scalar: trend precision from previous iteration */
      theta_02_previous,  /* scalar: initial trend from previous iteration */
      n                   /* sample size */
    );

    /* ===== Step 2: Sample Trend Innovation Precision 1/W_2 ===== */
    /* Draw 1/W_2 | theta_2_current, theta_{0,2}_previous from Gamma posterior.
     * Uses current theta_2 (just sampled) and previous theta_{0,2}. */
    prec_2_current = generate_precision_theta_p(
      theta_02_previous,  /* scalar: initial trend from previous iteration */
      theta_2_current,    /* vector: current theta_2 [n] */
      nu_02,              /* prior shape */
      eta_02,             /* prior rate */
      n                   /* sample size */
    );

    /* ===== Step 3: Sample Initial Trend State theta_{0,2} ===== */
    /* Draw theta_{0,2} | theta_1_previous, theta_2_current, theta_{0,1}_previous,
     * prec_1_previous, prec_2_current from Normal posterior.
     * Uses information from both level and trend components. */
    theta_02_current = generate_theta_0p(
      theta_1_previous,   /* theta_{p-1}: level from previous iteration [n] */
      theta_2_current,    /* theta_p: current trend [n] */
      theta_01_previous,  /* theta_{0,p-1}: initial level from previous iteration */
      prec_1_previous,    /* prec_{p-1}: level precision from previous iteration */
      prec_2_current,     /* prec_p: current trend precision */
      mean_theta02,       /* prior mean */
      prec_theta02,       /* prior precision */
      n                   /* sample size */
    );

    /* ===== Step 4: Sample Level State Vector theta_1 ===== */
    /* Draw theta_1 | y, theta_2_current, theta_{0,1}_previous, theta_{0,2}_current,
     * prec_1_previous, prec_y_previous from multivariate Normal with tridiagonal precision.
     * Uses current theta_2 to account for trend contribution to level evolution. */
    generate_theta_1(
      y,                  /* data: observed series [n] */
      theta_1_current,    /* output: current iteration theta_1 [n] */
      theta_2_current,    /* theta_2: current trend [n] */
      prec_y_previous,    /* scalar: data precision from previous iteration */
      prec_1_previous,    /* scalar: level precision from previous iteration */
      theta_01_previous,  /* scalar: initial level from previous iteration */
      theta_02_current,   /* scalar: current initial trend */
      n                   /* sample size */
    );

    /* ===== Step 5: Sample Level Innovation Precision 1/W_1 ===== */
    /* Draw 1/W_1 | theta_{0,1}_previous, theta_{0,2}_current, theta_1_current,
     * theta_2_current from Gamma posterior.
     * Uses both level and trend information to compute innovations. */
    prec_1_current = generate_precision_theta_k(
      theta_01_previous,  /* scalar: initial level from previous iteration */
      theta_02_current,   /* scalar: current initial trend */
      theta_1_current,    /* vector: current level [n] */
      theta_2_current,    /* vector: current trend [n] */
      nu_01,              /* prior shape */
      eta_01,             /* prior rate */
      n                   /* sample size */
    );

    /* ===== Step 6: Sample Initial Level State theta_{0,1} ===== */
    /* Draw theta_{0,1} | theta_1_current, theta_{0,2}_current, prec_1_current
     * from Normal posterior.
     * Uses current level and trend information. */
    theta_01_current = generate_theta_01(
      theta_1_current,    /* vector: current level [n] */
      theta_02_current,   /* scalar: current initial trend */
      prec_1_current,     /* scalar: current level precision */
      mean_theta01,       /* prior mean */
      prec_theta01,       /* prior precision */
      n                   /* sample size */
    );

    /* ===== Step 7: Sample Observation Precision 1/V ===== */
    /* Draw 1/V | y, theta_1_current from Gamma posterior.
     * Uses current level to compute observation residuals. */
    prec_y_current = generate_precision_data(
      y,                  /* data: observed series [n] */
      theta_1_current,    /* vector: current level [n] */
      nu_y,               /* prior shape */
      eta_y,              /* prior rate */
      n                   /* sample size */
    );

    /* ===== Store Post-Burn-in Samples with Thinning ===== */
    /* Only copy to output matrices for retained iterations.
     * This avoids storing the entire burn-in and thinned-out samples. */
    if (ii >= burnin && ((ii - burnin) % thinning) == 0) {
      int idx = chain_idx++;

      /* Copy current theta_1 and theta_2 vectors to output matrices (column-major storage) */
      for (int j = 0; j < n; j++) {
        REAL(theta_1_samples)[idx + j * n_chain] = theta_1_current[j];
        REAL(theta_2_samples)[idx + j * n_chain] = theta_2_current[j];
      }

      /* Copy scalar parameters to output vectors */
      REAL(theta_01_samples)[idx] = theta_01_current;
      REAL(theta_02_samples)[idx] = theta_02_current;
      REAL(prec_1_samples)[idx]   = prec_1_current;
      REAL(prec_2_samples)[idx]   = prec_2_current;
      REAL(prec_y_samples)[idx]   = prec_y_current;
    }

    /* ===== Update Previous Values for Next Iteration ===== */
    /* Efficient element-wise copy for theta_1 and theta_2 trajectories */
    for (int j = 0; j < n; j++) {
      theta_1_previous[j] = theta_1_current[j];
      theta_2_previous[j] = theta_2_current[j];
    }
    theta_01_previous = theta_01_current;
    theta_02_previous = theta_02_current;
    prec_1_previous   = prec_1_current;
    prec_2_previous   = prec_2_current;
    prec_y_previous   = prec_y_current;
  }

  /* ========== Restore RNG State ========== */
  PutRNGstate();

  /* ========== Free Temporary Buffers ========== */
  R_Free(theta_1_current);
  R_Free(theta_1_previous);
  R_Free(theta_2_current);
  R_Free(theta_2_previous);

  /* ========== Package Results into Named List ========== */
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
