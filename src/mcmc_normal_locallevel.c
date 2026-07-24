/**
 * @file mcmc_normal_locallevel.c
 * @brief MCMC sampling for Gaussian local-level dynamic models
 * @author Michel H. Montoril
 * @date 2025-10-11
 * @version 1.1
 *
 * @details This file implements a complete Gibbs sampler for Bayesian estimation of
 *          local-level polynomial dynamic models with Gaussian observation equations.
 *          The implementation features:
 *          - Memory-efficient O(n) temporary storage using current/previous buffers
 *          - Scalar parameter passing to conditional samplers
 *          - Direct vector operations without iteration indexing
 *          - Efficient state trajectory sampling via forward-backward recursions
 *          - Conjugate posterior updates for all parameters
 *
 *          **Model specification:**
 *          Observation equation: y_t = theta_{t,1} + e_t,  e_t ~ N(0, V)
 *          State equation: theta_{t,1} = theta_{t-1,1} + u_{t,1},  u_{t,1} ~ N(0, W_1)
 *
 *          **Gibbs sampling sequence per iteration:**
 *          1. theta_1 | y, theta_{0,1}, W_1, V
 *          2. 1/W_1 | theta_1, theta_{0,1}
 *          3. theta_{0,1} | theta_1, W_1
 *          4. 1/V | y, theta_1
 */

#include <R.h>
#include <Rinternals.h>
#include <Rmath.h>
#include "conditional_state.h"
#include "conditional_precision.h"
#include "conditional_theta0.h"
#include "utils.h"          /* rgamma_positive */
#include "prec_prior_dispatch.h" /* prec_prior_t, step_prec_*, pdm_init_prec_prior */
#include "mcmc_progress_bar.h"
#include "mcmc_normal_locallevel.h"

/**
 * @brief Gibbs sampler for local-level dynamic model with Gaussian observations
 *
 * @details Implements a complete Gibbs MCMC algorithm for the local-level dynamic model:
 *
 *          **Observation equation:**
 *          y_t = theta_{t,1} + e_t,     e_t ~ N(0, V)
 *
 *          **State equation:**
 *          theta_{t,1} = theta_{t-1,1} + u_{t,1},  u_{t,1} ~ N(0, W_1)
 *
 *          The algorithm employs the Markovian structure of polynomial dynamic models
 *          to enable efficient block sampling of the state vector through forward-backward
 *          recursions implemented via tridiagonal precision matrix methods.
 *
 *          **Sampling sequence per iteration:**
 *          1. theta_1 | y, theta_{0,1}, W_1, V → Multivariate Normal (tridiagonal precision)
 *          2. 1/W_1 | theta_1, theta_{0,1} → Gamma posterior
 *          3. theta_{0,1} | theta_1, W_1 → Normal posterior
 *          4. 1/V | y, theta_1 → Gamma posterior
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
 * @param prior_theta01_mean_   SEXP Double scalar, prior mean mu_0 for initial state theta_{0,1}.
 *                              Typical value: 0 (vague prior).
 * @param prior_theta01_prec_   SEXP Double scalar, prior precision tau_0 = 1/sigma_0^2 for
 *                              initial state. Typical value: 0.001 (vague prior).
 * @param prior_prec1_type_     SEXP Integer scalar selecting the prior on 1/W_1:
 *                              0 = Gamma on the precision, 1 = Half-t on the
 *                              innovation standard deviation sqrt(W_1).
 * @param prior_prec1_shape_    SEXP Double scalar, Gamma shape nu_1 (used when
 *                              prior_prec1_type_ == 0). Typical value: 0.001.
 * @param prior_prec1_rate_     SEXP Double scalar, Gamma rate eta_1 (used when
 *                              prior_prec1_type_ == 0). Typical value: 0.001.
 * @param prior_prec1_scale_    SEXP Double scalar, Half-t scale A_1 > 0 (used when
 *                              prior_prec1_type_ == 1).
 * @param prior_prec1_df_       SEXP Double scalar, Half-t degrees of freedom nu_1 > 0
 *                              (used when prior_prec1_type_ == 1; 1 = Half-Cauchy).
 * @param prior_prec_y_type_    SEXP Integer scalar selecting the prior on 1/V
 *                              (0 = Gamma on the precision, 1 = Half-t on sqrt(V)).
 * @param prior_prec_y_shape_   SEXP Double scalar, Gamma shape nu_y (used when
 *                              prior_prec_y_type_ == 0). Typical value: 0.001.
 * @param prior_prec_y_rate_    SEXP Double scalar, Gamma rate eta_y (used when
 *                              prior_prec_y_type_ == 0). Typical value: 0.001.
 * @param prior_prec_y_scale_   SEXP Double scalar, Half-t scale A_V > 0 (used when
 *                              prior_prec_y_type_ == 1).
 * @param prior_prec_y_df_      SEXP Double scalar, Half-t degrees of freedom nu_y > 0
 *                              (used when prior_prec_y_type_ == 1; 1 = Half-Cauchy).
 * @param verbose_              SEXP Logical scalar controlling progress bar display
 *                              (0 = disabled, non-zero = enabled).
 * @param bar_width_            SEXP Integer scalar defining progress bar width in
 *                              characters. Recommended range: 10-120.
 *
 * @return SEXP R list containing posterior samples with named components:
 *         - theta_1: Numeric matrix [n_chain × n] of complete state trajectory samples
 *         - theta_01: Numeric vector [n_chain] of initial state theta_{0,1} samples
 *         - prec_theta1: Numeric vector [n_chain] of innovation precision 1/W_1 samples
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
 * @see generate_theta_1_locallevel
 * @see generate_precision_theta_p
 * @see generate_theta_01_locallevel
 * @see generate_precision_data
 */
SEXP C_MCMC_normal_locallevel(SEXP y_,
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
                              SEXP prior_prec_y_type_,
                              SEXP prior_prec_y_shape_,
                              SEXP prior_prec_y_rate_,
                              SEXP prior_prec_y_scale_,
                              SEXP prior_prec_y_df_,
                              SEXP verbose_,
                              SEXP bar_width_) {

  /* ========== Parse Data Vector and Validate Length ========== */
  double   *y   = REAL(y_);
  R_xlen_t  len = LENGTH(y_);

  /* Enforce minimum sample size for numerical stability */
  if (len < 3) {
    Rf_error("C_MCMC_normal_locallevel: sample size 'n' must be at least 3, got %lld",
             (long long) len);
  }

  /* Check for integer overflow (R limitation) */
  if (len > INT_MAX) {
    Rf_error("C_MCMC_normal_locallevel: sample size too large (%lld > %d)",
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

  /* Innovation precision 1/W_1 prior. The R wrapper resolves the "halfcauchy"
   * alias to Half-t with df = 1 and passes finite placeholders for the unused
   * fields, so no field is ever NA regardless of the selected kind. */
  int          prec1_kind = asInteger(prior_prec1_type_);
  prec_prior_t prior_W1   = {
    .shape    = REAL(prior_prec1_shape_)[0],  /* Gamma shape nu_1 */
    .rate     = REAL(prior_prec1_rate_)[0],   /* Gamma rate  eta_1 */
    .df       = REAL(prior_prec1_df_)[0],     /* Half-t df   nu_1 */
    .hc_scale = REAL(prior_prec1_scale_)[0]   /* Half-t scale A_1 */
  };

  /* Observation precision 1/V prior. */
  int          precy_kind = asInteger(prior_prec_y_type_);
  prec_prior_t prior_V    = {
    .shape    = REAL(prior_prec_y_shape_)[0],
    .rate     = REAL(prior_prec_y_rate_)[0],
    .df       = REAL(prior_prec_y_df_)[0],
    .hc_scale = REAL(prior_prec_y_scale_)[0]
  };

  /* ===== Resolve the prior dispatch ONCE (outside the Gibbs loop) ===== */
  prec_thetap_step_t update_prec_W1 =
    (prec1_kind == PDM_PREC_PRIOR_HALFT) ? step_prec_thetap_halft
                                         : step_prec_thetap_gamma;
  prec_data_step_t   update_prec_V  =
    (precy_kind == PDM_PREC_PRIOR_HALFT) ? step_prec_data_halft
                                         : step_prec_data_gamma;

  /* ========== Parse Progress Bar Parameters ========== */
  int verbose   = asLogical(verbose_);
  int bar_width = asInteger(bar_width_);

  /* ===== Initiate Progress Bar ===== */
  ProgressBar pb = progress_bar_init(n_iter, bar_width, burnin, thinning, verbose);
  progress_bar_start(&pb);

  /* ========== Allocate Output Storage (Retained Samples Only) ========== */
  SEXP theta_1_samples     = PROTECT(allocMatrix(REALSXP, n_chain, n));
  SEXP theta_01_samples    = PROTECT(allocVector(REALSXP, n_chain));
  SEXP prec_theta1_samples = PROTECT(allocVector(REALSXP, n_chain));
  SEXP prec_y_samples      = PROTECT(allocVector(REALSXP, n_chain));

  /* ========== Allocate Temporary Buffers (Memory-Efficient O(n) Storage) ========== */
  /* Uses current/previous iteration buffers for efficient memory management.
   * Only requires O(n) space regardless of total chain length.
   * Example: n=500 requires only ~8 KB instead of storing full trajectories. */
  double *theta_1_current  = (double *) R_Calloc(n, double);
  double *theta_1_previous = (double *) R_Calloc(n, double);

  /* Scalar parameters for current and previous iterations */
  double theta_01_current, theta_01_previous;
  double prec_theta1_current,   prec_theta1_previous;
  double prec_y_current,   prec_y_previous;

  /* Half-t auxiliary variables b = 1/a. Each is refreshed in place right after
   * its precision draw when the corresponding prior is Half-t; under the Gamma
   * prior it stays 0 and is never read. A single variable suffices (no
   * previous/current split): it is used only to form the next precision rate. */
  double aux_W1 = 0.0, aux_V = 0.0;

  /* ========== Initialize RNG State ========== */
  GetRNGstate();

  /* ========== Initialize Parameters (Iteration 0) ========== */
  /* Draw initial values from the priors to start the Markov chain. For a Half-t
   * prior the starting precision is drawn through the scale-mixture
   * representation: b ~ Gamma(1/2, scale = A^2), then W^{-1} ~ Gamma(df/2, df*b).
   * These one-time branches are outside the Gibbs loop. */
  theta_01_previous = rnorm(mean_theta01, sqrt(1.0 / prec_theta01));

  prec_theta1_previous = pdm_init_prec_prior(prec1_kind, &prior_W1, &aux_W1);
  prec_y_previous      = pdm_init_prec_prior(precy_kind, &prior_V,  &aux_V);

  /* Initialize theta_1 trajectory with neutral starting values */
  for (int j = 0; j < n; j++) {
    theta_1_previous[j] = 0.0;
  }

  /* ========== Main Gibbs Sampling Loop ========== */
  /* Uses current/previous buffers for memory efficiency.
   * Only retained samples are copied to output (post burn-in, thinned). */
  int chain_idx = 0;

  for (int ii = 1; ii < n_iter; ii++) {

    /* ===== Step 1: Sample State Vector theta_1 ===== */
    /* Draw theta_1 | y, previous parameters from multivariate Normal
     * with tridiagonal precision matrix (O(n) via Cholesky). */
    generate_theta_1_locallevel(
      y,                  /* data: observed series [n] */
      theta_1_current,    /* output: current iteration theta_1 [n] */
      prec_y_previous,    /* scalar: data precision from previous iteration */
      prec_theta1_previous,    /* scalar: innovation precision from previous iteration */
      theta_01_previous,  /* scalar: initial state from previous iteration */
      n                   /* sample size */
    );

    /* ===== Step 2: Sample Innovation Precision 1/W_1 ===== */
    /* Draw 1/W_1 | theta_1_current, theta_{0,1}_previous through the prior chosen
     * before the loop (Gamma posterior, or Half-t via its scale-mixture step,
     * which also refreshes aux_W1). Uses current theta_1 and previous theta_{0,1}. */
    prec_theta1_current = update_prec_W1(
      theta_01_previous,  /* scalar: initial state from previous iteration */
      theta_1_current,    /* vector: current theta_1 [n] */
      n,                  /* sample size */
      &prior_W1,          /* prior hyperparameters for the resolved kind */
      &aux_W1             /* Half-t auxiliary (updated in place; unused if Gamma) */
    );

    /* ===== Step 3: Sample Initial State theta_{0,1} ===== */
    /* Draw theta_{0,1} | theta_1_current, prec_theta1_current from Normal posterior.
     * Uses current theta_1 and current prec_theta1 (both just sampled). */
    theta_01_current = generate_theta_01_locallevel(
      theta_1_current,    /* vector: current theta_1 [n] */
      prec_theta1_current,     /* scalar: current innovation precision */
      mean_theta01,       /* prior mean */
      prec_theta01,       /* prior precision */
      n                   /* sample size */
    );

    /* ===== Step 4: Sample Observation Precision 1/V ===== */
    /* Draw 1/V | y, theta_1_current through the prior chosen before the loop
     * (Gamma posterior, or Half-t via its scale-mixture step, which also
     * refreshes aux_V). Uses current theta_1 to compute observation residuals. */
    prec_y_current = update_prec_V(
      y,                  /* data: observed series [n] */
      theta_1_current,    /* vector: current theta_1 [n] */
      n,                  /* sample size */
      &prior_V,           /* prior hyperparameters for the resolved kind */
      &aux_V              /* Half-t auxiliary (updated in place; unused if Gamma) */
    );

    /* ===== Store Post-Burn-in Samples with Thinning ===== */
    /* Only copy to output matrix for retained iterations.
     * This avoids storing the entire burn-in and thinned-out samples. */
    if (ii >= burnin && ((ii - burnin) % thinning) == 0) {
      int idx = chain_idx++;

      /* Copy current theta_1 vector to output matrix (column-major storage) */
      for (int j = 0; j < n; j++) {
        REAL(theta_1_samples)[idx + j * n_chain] = theta_1_current[j];
      }

      /* Copy scalar parameters to output vectors */
      REAL(theta_01_samples)[idx] = theta_01_current;
      REAL(prec_theta1_samples)[idx] = prec_theta1_current;
      REAL(prec_y_samples)[idx]   = prec_y_current;
    }

    /* ===== Update Progress Bar ===== */
    if (ii % pb.update_step == 0 || ii == n_iter - 1) {
      progress_bar_update(&pb, ii);
    }

    /* ===== Update Previous Values for Next Iteration ===== */
    /* Efficient element-wise copy for theta_1 trajectory */
    for (int j = 0; j < n; j++) {
      theta_1_previous[j] = theta_1_current[j];
    }
    theta_01_previous = theta_01_current;
    prec_theta1_previous = prec_theta1_current;
    prec_y_previous   = prec_y_current;
  }

  /* ========== Finalize Progress Bar ========== */
  progress_bar_finish(&pb, n_chain);

  /* ========== Restore RNG State ========== */
  PutRNGstate();

  /* ========== Free Temporary Buffers ========== */
  R_Free(theta_1_current);
  R_Free(theta_1_previous);

  /* ========== Package Results into Named List ========== */
  SEXP out = PROTECT(allocVector(VECSXP, 4));
  SET_VECTOR_ELT(out, 0, theta_1_samples);
  SET_VECTOR_ELT(out, 1, theta_01_samples);
  SET_VECTOR_ELT(out, 2, prec_theta1_samples);
  SET_VECTOR_ELT(out, 3, prec_y_samples);

  SEXP nms = PROTECT(allocVector(STRSXP, 4));
  SET_STRING_ELT(nms, 0, mkChar("theta_1"));
  SET_STRING_ELT(nms, 1, mkChar("theta_01"));
  SET_STRING_ELT(nms, 2, mkChar("prec_theta1"));
  SET_STRING_ELT(nms, 3, mkChar("prec_y"));
  setAttrib(out, R_NamesSymbol, nms);

  UNPROTECT(6);
  return out;
}
