/**
 * @file mcmc_normal_localacceleration.c
 * @brief MCMC sampling for Gaussian local-acceleration dynamic models
 * @author Michel H. Montoril
 * @date 2026-07-25
 * @version 1.3
 *
 * @details This file implements a complete Gibbs sampler for Bayesian estimation of
 *          local-acceleration polynomial dynamic models with Gaussian observation equations.
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
 *            theta_{t,2} = theta_{t-1,2} + theta_{t-1,3} + u_{t,2},  u_{t,2} ~ N(0, W_2)
 *            theta_{t,3} = theta_{t-1,3} + u_{t,3},                  u_{t,3} ~ N(0, W_3)
 *
 *          **Gibbs sampling sequence per iteration:**
 *          1. theta_3 | theta_2, theta_{0,2}, theta_{0,3}, W_2, W_3
 *          2. 1/W_3 | theta_3, theta_{0,3}
 *          3. theta_{0,3} | theta_2, theta_3, theta_{0,2}, W_2, W_3
 *          4. theta_2 | theta_1, theta_3, theta_{0,1}, theta_{0,2}, theta_{0,3}, W_1, W_2
 *          5. 1/W_2 | theta_2, theta_3, theta_{0,2}, theta_{0,3}
 *          6. theta_{0,2} | theta_1, theta_2, theta_{0,1}, theta_{0,3}, W_1, W_2
 *          7. theta_1 | y, theta_2, theta_{0,1}, theta_{0,2}, W_1, V
 *          8. 1/W_1 | theta_1, theta_2, theta_{0,1}, theta_{0,2}
 *          9. theta_{0,1} | theta_1, theta_{0,2}, W_1
 *          10. 1/V | y, theta_1
 */

#include <R.h>
#include <Rinternals.h>
#include <Rmath.h>
#include "conditional_state.h"
#include "conditional_precision.h"
#include "conditional_theta0.h"
#include "prec_prior_dispatch.h" /* prec_prior_t, step_prec_* */
#include "mcmc_normal_localacceleration.h"
#include "mcmc_progress_bar.h"

/**
 * @brief Gibbs sampler for local-acceleration dynamic model with Gaussian observations
 *
 * @details Implements a complete Gibbs MCMC algorithm for the local-acceleration dynamic model:
 *
 *          **Observation equation:**
 *          y_t = theta_{t,1} + e_t,     e_t ~ N(0, V)
 *
 *          **State equations:**
 *          theta_{t,1} = theta_{t-1,1} + theta_{t-1,2} + u_{t,1},  u_{t,1} ~ N(0, W_1)
 *          theta_{t,2} = theta_{t-1,2} + theta_{t-1,3} + u_{t,2},  u_{t,2} ~ N(0, W_2)
 *          theta_{t,3} = theta_{t-1,3} + u_{t,3},                  u_{t,3} ~ N(0, W_3)
 *
 *          The algorithm employs the Markovian structure of polynomial dynamic models
 *          to enable efficient block sampling of state vectors through forward-backward
 *          recursions implemented via tridiagonal precision matrix methods.
 *
 *          **Sampling sequence per iteration:**
 *          1. theta_3 | theta_2, theta_{0,2}, theta_{0,3}, W_2, W_3 → MVN (tridiagonal)
 *          2. 1/W_3 | theta_3, theta_{0,3} → Gamma posterior
 *          3. theta_{0,3} | theta_2, theta_3, theta_{0,2}, W_2, W_3 → Normal posterior
 *          4. theta_2 | theta_1, theta_3, theta_{0,1}, theta_{0,2}, theta_{0,3}, W_1, W_2 → MVN
 *          5. 1/W_2 | theta_2, theta_3, theta_{0,2}, theta_{0,3} → Gamma posterior
 *          6. theta_{0,2} | theta_1, theta_2, theta_{0,1}, theta_{0,3}, W_1, W_2 → Normal
 *          7. theta_1 | y, theta_2, theta_{0,1}, theta_{0,2}, W_1, V → MVN (tridiagonal)
 *          8. 1/W_1 | theta_1, theta_2, theta_{0,1}, theta_{0,2} → Gamma posterior
 *          9. theta_{0,1} | theta_1, theta_{0,2}, W_1 → Normal posterior
 *          10. 1/V | y, theta_1 → Gamma posterior
 *
 *          **Memory efficiency:**
 *          Uses current/previous iteration buffers instead of full trajectory storage,
 *          requiring only O(n) temporary memory regardless of chain length.
 *
 *          **Iteration count:**
 *          Total iterations = burnin + (n_draws - 1) × thinning + 1
 *
 * @param y_                    SEXP Numeric vector of observed time series data [length n].
 *                              Contains observations y_1, ..., y_n.
 * @param burnin_               SEXP Integer scalar, number of burn-in iterations to discard
 *                              for chain convergence. Typical values: 1000-10000.
 * @param thinning_             SEXP Integer scalar, thinning interval to reduce autocorrelation
 *                              in retained samples. Typical values: 1-10.
 * @param n_draws_              SEXP Integer scalar, target number of retained posterior samples.
 *                              Final output will contain exactly n_draws samples.
 * @param prior_theta01_mean_   SEXP Double scalar, prior mean mu_{0,1} for initial level theta_{0,1}.
 *                              Typical value: 0 (vague prior).
 * @param prior_theta01_prec_   SEXP Double scalar, prior precision tau_{0,1} = 1/sigma_{0,1}^2
 *                              for initial level. Typical value: 0.001 (vague prior).
 * @param prior_theta02_mean_   SEXP Double scalar, prior mean mu_{0,2} for initial trend theta_{0,2}.
 *                              Typical value: 0 (vague prior).
 * @param prior_theta02_prec_   SEXP Double scalar, prior precision tau_{0,2} = 1/sigma_{0,2}^2
 *                              for initial trend. Typical value: 0.001 (vague prior).
 * @param prior_theta03_mean_   SEXP Double scalar, prior mean mu_{0,3} for initial acceleration theta_{0,3}.
 *                              Typical value: 0 (vague prior).
 * @param prior_theta03_prec_   SEXP Double scalar, prior precision tau_{0,3} = 1/sigma_{0,3}^2
 *                              for initial acceleration. Typical value: 0.001 (vague prior).
 * @param prior_prec1_type_     SEXP Integer scalar selecting the prior on 1/W_1
 *                              (0 = Gamma on the precision, 1 = Half-t on sqrt(W_1)).
 * @param prior_prec1_shape_    SEXP Double scalar, Gamma shape nu_1 (Gamma kind). Typical: 0.001.
 * @param prior_prec1_rate_     SEXP Double scalar, Gamma rate eta_1 (Gamma kind). Typical: 0.001.
 * @param prior_prec1_scale_    SEXP Double scalar, Half-t scale A_1 > 0 (Half-t kind).
 * @param prior_prec1_df_       SEXP Double scalar, Half-t df nu_1 > 0 (Half-t kind; 1 = Half-Cauchy).
 * @param prior_prec2_type_     SEXP Integer scalar selecting the prior on 1/W_2 (0 = Gamma, 1 = Half-t).
 * @param prior_prec2_shape_    SEXP Double scalar, Gamma shape nu_2 (Gamma kind). Typical: 0.001.
 * @param prior_prec2_rate_     SEXP Double scalar, Gamma rate eta_2 (Gamma kind). Typical: 0.001.
 * @param prior_prec2_scale_    SEXP Double scalar, Half-t scale A_2 > 0 (Half-t kind).
 * @param prior_prec2_df_       SEXP Double scalar, Half-t df nu_2 > 0 (Half-t kind; 1 = Half-Cauchy).
 * @param prior_prec3_type_     SEXP Integer scalar selecting the prior on 1/W_3 (0 = Gamma, 1 = Half-t).
 * @param prior_prec3_shape_    SEXP Double scalar, Gamma shape nu_3 (Gamma kind). Typical: 0.001.
 * @param prior_prec3_rate_     SEXP Double scalar, Gamma rate eta_3 (Gamma kind). Typical: 0.001.
 * @param prior_prec3_scale_    SEXP Double scalar, Half-t scale A_3 > 0 (Half-t kind).
 * @param prior_prec3_df_       SEXP Double scalar, Half-t df nu_3 > 0 (Half-t kind; 1 = Half-Cauchy).
 * @param prior_prec_y_type_    SEXP Integer scalar selecting the prior on 1/V (0 = Gamma, 1 = Half-t).
 * @param prior_prec_y_shape_   SEXP Double scalar, Gamma shape nu_y (Gamma kind). Typical: 0.001.
 * @param prior_prec_y_rate_    SEXP Double scalar, Gamma rate eta_y (Gamma kind). Typical: 0.001.
 * @param prior_prec_y_scale_   SEXP Double scalar, Half-t scale A_V > 0 (Half-t kind).
 * @param prior_prec_y_df_      SEXP Double scalar, Half-t df nu_y > 0 (Half-t kind; 1 = Half-Cauchy).
 * @param init_                 SEXP Double vector [11] of starting values, resolved in R by
 *                              resolve_init(): theta_{0,1} to theta_{0,3}, then 1/W_1 to
 *                              1/W_3 and 1/V, then the Half-t auxiliary of each precision
 *                              in the same order (0 under a Gamma prior, never read).
 * @param verbose_              SEXP Logical scalar controlling progress bar display
 *                              (0 = disabled, non-zero = enabled).
 * @param bar_width_            SEXP Integer scalar defining progress bar width in
 *                              characters. Recommended range: 10-120.
 *
 * @return SEXP R list containing posterior samples with named components:
 *         - theta_1: Numeric matrix [n_draws × n] of level state trajectory samples
 *         - theta_2: Numeric matrix [n_draws × n] of trend state trajectory samples
 *         - theta_3: Numeric matrix [n_draws × n] of acceleration state trajectory samples
 *         - theta_01: Numeric vector [n_draws] of initial level state theta_{0,1} samples
 *         - theta_02: Numeric vector [n_draws] of initial trend state theta_{0,2} samples
 *         - theta_03: Numeric vector [n_draws] of initial acceleration state theta_{0,3} samples
 *         - prec_theta1: Numeric vector [n_draws] of level innovation precision 1/W_1 samples
 *         - prec_theta2: Numeric vector [n_draws] of trend innovation precision 1/W_2 samples
 *         - prec_theta3: Numeric vector [n_draws] of acceleration innovation precision 1/W_3 samples
 *         - prec_y: Numeric vector [n_draws] of observation precision 1/V samples
 *
 * @note Computational complexity: O(n_iter × n) for n_iter total iterations.
 * @note Memory requirements: O(n) temporary storage for efficient buffer management.
 * @note RNG management: Proper GetRNGstate()/PutRNGstate() bracket for R integration.
 * @note Initialization: Starting values are decided in R and read from init_; this
 *       function draws none of them itself.
 *
 * @warning Minimum sample size n >= 3 enforced for numerical stability of recursions.
 * @warning Integer overflow protection: n <= INT_MAX due to R's integer limitations.
 * @warning Memory allocation failures will terminate R session via R_Calloc errors.
 * @warning No input validation for prior hyperparameters; negative values may cause crashes.
 * @warning No input validation for init_; it is assumed to have the documented
 *          length and to hold finite values, both guaranteed by resolve_init().
 *
 * @see generate_theta_p
 * @see generate_precision_theta_p
 * @see generate_theta_0p
 * @see generate_theta_k
 * @see generate_precision_theta_k
 * @see generate_theta_0k
 * @see generate_theta_1
 * @see generate_theta_01
 * @see generate_precision_data
 */
SEXP C_MCMC_normal_localacceleration(SEXP y_,
                                     SEXP burnin_,
                                     SEXP thinning_,
                                     SEXP n_draws_,
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
                                     SEXP prior_prec_y_type_,
                                     SEXP prior_prec_y_shape_,
                                     SEXP prior_prec_y_rate_,
                                     SEXP prior_prec_y_scale_,
                                     SEXP prior_prec_y_df_,
                                     SEXP init_,
                                     SEXP verbose_,
                                     SEXP bar_width_) {

  /* ========== Parse Data Vector and Validate Length ========== */
  double   *y   = REAL(y_);
  R_xlen_t  len = LENGTH(y_);

  /* Enforce minimum sample size for numerical stability */
  if (len < 3) {
    Rf_error("C_MCMC_normal_localacceleration: sample size 'n' must be at least 3, got %lld",
             (long long) len);
  }

  /* Check for integer overflow (R limitation) */
  if (len > INT_MAX) {
    Rf_error("C_MCMC_normal_localacceleration: sample size too large (%lld > %d)",
             (long long) len, INT_MAX);
  }

  int n = (int) len;

  /* ========== Parse MCMC Control Parameters ========== */
  int burnin   = INTEGER(burnin_)[0];     /* Burn-in iterations to discard */
  int thinning = INTEGER(thinning_)[0];   /* Thinning interval for autocorrelation reduction */
  int n_draws  = INTEGER(n_draws_)[0];    /* Number of retained samples */

  /* Compute total iterations needed to produce n_draws thinned samples after burn-in */
  int n_iter = burnin + (n_draws - 1) * thinning + 1;

  /* ========== Parse Prior Hyperparameters ========== */
  double mean_theta01 = REAL(prior_theta01_mean_)[0]; /* Prior mean for theta_{0,1} */
  double prec_theta01 = REAL(prior_theta01_prec_)[0]; /* Prior precision for theta_{0,1} */
  double mean_theta02 = REAL(prior_theta02_mean_)[0]; /* Prior mean for theta_{0,2} */
  double prec_theta02 = REAL(prior_theta02_prec_)[0]; /* Prior precision for theta_{0,2} */
  double mean_theta03 = REAL(prior_theta03_mean_)[0]; /* Prior mean for theta_{0,3} */
  double prec_theta03 = REAL(prior_theta03_prec_)[0]; /* Prior precision for theta_{0,3} */

  /* Precision priors. The R wrapper resolves the "halfcauchy" alias to Half-t
   * with df = 1 and passes finite placeholders for the unused fields, so no
   * field is ever NA regardless of the selected kind (0 = Gamma, 1 = Half-t). */
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
  int          precy_kind = asInteger(prior_prec_y_type_);  /* prior on 1/V */
  prec_prior_t prior_V    = {
    .shape    = REAL(prior_prec_y_shape_)[0],
    .rate     = REAL(prior_prec_y_rate_)[0],
    .df       = REAL(prior_prec_y_df_)[0],
    .hc_scale = REAL(prior_prec_y_scale_)[0]
  };

  /* ===== Resolve the prior dispatch ONCE (outside the Gibbs loop) =====
   * W_1 and W_2 are intermediate components (theta_k sampler), W_3 the terminal
   * random-walk component (theta_p sampler), V the observation precision. */
  prec_thetak_step_t update_prec_W1 =
    (prec1_kind == PDM_PREC_PRIOR_HALFT) ? step_prec_thetak_halft
                                         : step_prec_thetak_gamma;
  prec_thetak_step_t update_prec_W2 =
    (prec2_kind == PDM_PREC_PRIOR_HALFT) ? step_prec_thetak_halft
                                         : step_prec_thetak_gamma;
  prec_thetap_step_t update_prec_W3 =
    (prec3_kind == PDM_PREC_PRIOR_HALFT) ? step_prec_thetap_halft
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
  SEXP theta_1_samples     = PROTECT(allocMatrix(REALSXP, n_draws, n));
  SEXP theta_2_samples     = PROTECT(allocMatrix(REALSXP, n_draws, n));
  SEXP theta_3_samples     = PROTECT(allocMatrix(REALSXP, n_draws, n));
  SEXP theta_01_samples    = PROTECT(allocVector(REALSXP, n_draws));
  SEXP theta_02_samples    = PROTECT(allocVector(REALSXP, n_draws));
  SEXP theta_03_samples    = PROTECT(allocVector(REALSXP, n_draws));
  SEXP prec_theta1_samples = PROTECT(allocVector(REALSXP, n_draws));
  SEXP prec_theta2_samples = PROTECT(allocVector(REALSXP, n_draws));
  SEXP prec_theta3_samples = PROTECT(allocVector(REALSXP, n_draws));
  SEXP prec_y_samples      = PROTECT(allocVector(REALSXP, n_draws));

  /* ========== Allocate Temporary Buffers (Memory-Efficient O(n) Storage) ========== */
  /* Uses current/previous iteration buffers for efficient memory management.
   * Only requires O(n) space regardless of total chain length. */
  double *theta_1_current  = (double *) R_Calloc(n, double);
  double *theta_1_previous = (double *) R_Calloc(n, double);
  double *theta_2_current  = (double *) R_Calloc(n, double);
  double *theta_2_previous = (double *) R_Calloc(n, double);
  double *theta_3_current  = (double *) R_Calloc(n, double);
  double *theta_3_previous = (double *) R_Calloc(n, double);

  /* Scalar parameters for current and previous iterations */
  double theta_01_current, theta_01_previous;
  double theta_02_current, theta_02_previous;
  double theta_03_current, theta_03_previous;
  double prec_theta1_current,   prec_theta1_previous;
  double prec_theta2_current,   prec_theta2_previous;
  double prec_theta3_current,   prec_theta3_previous;
  double prec_y_current,   prec_y_previous;

  /* Half-t auxiliaries b = 1/a, one per precision. Refreshed in place after the
   * matching precision draw when its prior is Half-t; hold 0 (never read)
   * under the Gamma prior. */
  double aux_W1, aux_W2, aux_W3, aux_V;

  /* ========== Initialize RNG State ========== */
  GetRNGstate();

  /* ========== Initialize Parameters (Iteration 0) ========== */
  /* Starting values arrive already decided from R (see resolve_init() in
   * R/init_values.R): whatever the caller pinned through `init`, and a draw
   * from the corresponding prior for everything else. `init_` is laid out as
   * the initial states in order, then the precisions, then one auxiliary per
   * precision in the same order. */
  const double *init = REAL(init_);

  theta_01_previous    = init[0];
  theta_02_previous    = init[1];
  theta_03_previous    = init[2];
  prec_theta1_previous = init[3];
  prec_theta2_previous = init[4];
  prec_theta3_previous = init[5];
  prec_y_previous      = init[6];
  aux_W1               = init[7];
  aux_W2               = init[8];
  aux_W3               = init[9];
  aux_V                = init[10];

  /* Initialize theta_1, theta_2, and theta_3 trajectories with neutral starting values */
  for (int j = 0; j < n; j++) {
    theta_1_previous[j] = 0.0;
    theta_2_previous[j] = 0.0;
    theta_3_previous[j] = 0.0;
  }

  /* ========== Main Gibbs Sampling Loop ========== */
  /* Uses current/previous buffers for memory efficiency.
   * Only retained samples are copied to output (post burn-in, thinned). */
  int chain_idx = 0;

  for (int ii = 1; ii < n_iter; ii++) {

    /* ===== Step 1: Sample Acceleration State Vector theta_3 ===== */
    /* Draw theta_3 | theta_2, previous parameters from multivariate Normal
     * with tridiagonal precision matrix (O(n) via Cholesky).
     * Uses theta_2 from previous iteration for computing differences. */
    generate_theta_p(
      theta_2_previous,   /* theta_{p-1}: trend from previous iteration [n] */
      theta_3_current,    /* output: current iteration theta_3 [n] */
      prec_theta2_previous,    /* scalar: trend precision from previous iteration */
      prec_theta3_previous,    /* scalar: acceleration precision from previous iteration */
      theta_03_previous,  /* scalar: initial acceleration from previous iteration */
      n                   /* sample size */
    );

    /* ===== Step 2: Sample Acceleration Innovation Precision 1/W_3 ===== */
    /* Draw 1/W_3 | theta_3_current, theta_{0,3}_previous through the prior chosen
     * before the loop (Gamma posterior, or Half-t via its scale-mixture step,
     * which also refreshes aux_W3). Uses current theta_3 and previous theta_{0,3}. */
    prec_theta3_current = update_prec_W3(
      theta_03_previous,  /* scalar: initial acceleration from previous iteration */
      theta_3_current,    /* vector: current theta_3 [n] */
      n,                  /* sample size */
      &prior_W3,          /* prior hyperparameters for the resolved kind */
      &aux_W3             /* Half-t auxiliary (updated in place; unused if Gamma) */
    );

    /* ===== Step 3: Sample Initial Acceleration State theta_{0,3} ===== */
    /* Draw theta_{0,3} | theta_2_previous, theta_3_current, theta_{0,2}_previous,
     * prec_theta2_previous, prec_theta3_current from Normal posterior.
     * Uses information from both trend and acceleration components. */
    theta_03_current = generate_theta_0p(
      theta_2_previous,   /* theta_{p-1}: trend from previous iteration [n] */
      theta_3_current,    /* theta_p: current acceleration [n] */
      theta_02_previous,  /* theta_{0,p-1}: initial trend from previous iteration */
      prec_theta2_previous,    /* prec_{p-1}: trend precision from previous iteration */
      prec_theta3_current,     /* prec_p: current acceleration precision */
      mean_theta03,       /* prior mean */
      prec_theta03,       /* prior precision */
      n                   /* sample size */
    );

    /* ===== Step 4: Sample Trend State Vector theta_2 ===== */
    /* Draw theta_2 | theta_1_previous, theta_3_current, theta_{0,1}_previous,
     * theta_{0,2}_previous, theta_{0,3}_current, prec_theta1_previous, prec_theta2_previous
     * from multivariate Normal with tridiagonal precision.
     * Uses current theta_3 to account for acceleration contribution to trend evolution. */
    generate_theta_k(
      theta_1_previous,   /* theta_{k-1}: level from previous iteration [n] */
      theta_2_current,    /* output: current iteration theta_2 [n] */
      theta_3_current,    /* theta_{k+1}: current acceleration [n] */
      prec_theta1_previous,    /* scalar: level precision from previous iteration */
      prec_theta2_previous,    /* scalar: trend precision from previous iteration */
      theta_02_previous,  /* scalar: initial trend from previous iteration */
      theta_03_current,   /* scalar: current initial acceleration */
      n                   /* sample size */
    );

    /* ===== Step 5: Sample Trend Innovation Precision 1/W_2 ===== */
    /* Draw 1/W_2 | theta_{0,2}_previous, theta_{0,3}_current, theta_2_current,
     * theta_3_current through the prior chosen before the loop (Gamma posterior,
     * or Half-t via its scale-mixture step, which also refreshes aux_W2).
     * Uses both trend and acceleration information to compute innovations. */
    prec_theta2_current = update_prec_W2(
      theta_02_previous,  /* scalar: initial trend from previous iteration */
      theta_03_current,   /* scalar: current initial acceleration */
      theta_2_current,    /* vector: current trend [n] */
      theta_3_current,    /* vector: current acceleration [n] */
      n,                  /* sample size */
      &prior_W2,          /* prior hyperparameters for the resolved kind */
      &aux_W2             /* Half-t auxiliary (updated in place; unused if Gamma) */
    );

    /* ===== Step 6: Sample Initial Trend State theta_{0,2} ===== */
    /* Draw theta_{0,2} | theta_1_previous, theta_2_current, theta_{0,1}_previous,
     * theta_{0,3}_current, prec_theta1_previous, prec_theta2_current from Normal posterior.
     * Uses information from level, trend, and acceleration components. */
    theta_02_current = generate_theta_0k(
      theta_1_previous,   /* theta_{k-1}: level from previous iteration [n] */
      theta_2_current,    /* theta_k: current trend [n] */
      theta_01_previous,  /* theta_{0,k-1}: initial level from previous iteration */
      theta_03_current,   /* theta_{0,k+1}: current initial acceleration */
      prec_theta1_previous,    /* prec_{k-1}: level precision from previous iteration */
      prec_theta2_current,     /* prec_k: current trend precision */
      mean_theta02,       /* prior mean */
      prec_theta02,       /* prior precision */
      n                   /* sample size */
    );

    /* ===== Step 7: Sample Level State Vector theta_1 ===== */
    /* Draw theta_1 | y, theta_2_current, theta_{0,1}_previous, theta_{0,2}_current,
     * prec_theta1_previous, prec_y_previous from multivariate Normal with tridiagonal precision.
     * Uses current theta_2 to account for trend contribution to level evolution. */
    generate_theta_1(
      y,                  /* data: observed series [n] */
      theta_1_current,    /* output: current iteration theta_1 [n] */
      theta_2_current,    /* theta_2: current trend [n] */
      prec_y_previous,    /* scalar: data precision from previous iteration */
      prec_theta1_previous,    /* scalar: level precision from previous iteration */
      theta_01_previous,  /* scalar: initial level from previous iteration */
      theta_02_current,   /* scalar: current initial trend */
      n                   /* sample size */
    );

    /* ===== Step 8: Sample Level Innovation Precision 1/W_1 ===== */
    /* Draw 1/W_1 | theta_{0,1}_previous, theta_{0,2}_current, theta_1_current,
     * theta_2_current through the prior chosen before the loop (Gamma posterior,
     * or Half-t via its scale-mixture step, which also refreshes aux_W1).
     * Uses both level and trend information to compute innovations. */
    prec_theta1_current = update_prec_W1(
      theta_01_previous,  /* scalar: initial level from previous iteration */
      theta_02_current,   /* scalar: current initial trend */
      theta_1_current,    /* vector: current level [n] */
      theta_2_current,    /* vector: current trend [n] */
      n,                  /* sample size */
      &prior_W1,          /* prior hyperparameters for the resolved kind */
      &aux_W1             /* Half-t auxiliary (updated in place; unused if Gamma) */
    );

    /* ===== Step 9: Sample Initial Level State theta_{0,1} ===== */
    /* Draw theta_{0,1} | theta_1_current, theta_{0,2}_current, prec_theta1_current
     * from Normal posterior.
     * Uses current level and trend information. */
    theta_01_current = generate_theta_01(
      theta_1_current,    /* vector: current level [n] */
      theta_02_current,   /* scalar: current initial trend */
      prec_theta1_current,     /* scalar: current level precision */
      mean_theta01,       /* prior mean */
      prec_theta01,       /* prior precision */
      n                   /* sample size */
    );

    /* ===== Step 10: Sample Observation Precision 1/V ===== */
    /* Draw 1/V | y, theta_1_current through the prior chosen before the loop
     * (Gamma posterior, or Half-t via its scale-mixture step, which also
     * refreshes aux_V). Uses current level to compute observation residuals. */
    prec_y_current = update_prec_V(
      y,                  /* data: observed series [n] */
      theta_1_current,    /* vector: current level [n] */
      n,                  /* sample size */
      &prior_V,           /* prior hyperparameters for the resolved kind */
      &aux_V              /* Half-t auxiliary (updated in place; unused if Gamma) */
    );

    /* ===== Store Post-Burn-in Samples with Thinning ===== */
    /* Only copy to output matrices for retained iterations.
     * This avoids storing the entire burn-in and thinned-out samples. */
    if (ii >= burnin && ((ii - burnin) % thinning) == 0) {
      int idx = chain_idx++;

      /* Copy current theta_1, theta_2, and theta_3 vectors to output matrices (column-major) */
      for (int j = 0; j < n; j++) {
        REAL(theta_1_samples)[idx + j * n_draws] = theta_1_current[j];
        REAL(theta_2_samples)[idx + j * n_draws] = theta_2_current[j];
        REAL(theta_3_samples)[idx + j * n_draws] = theta_3_current[j];
      }

      /* Copy scalar parameters to output vectors */
      REAL(theta_01_samples)[idx] = theta_01_current;
      REAL(theta_02_samples)[idx] = theta_02_current;
      REAL(theta_03_samples)[idx] = theta_03_current;
      REAL(prec_theta1_samples)[idx] = prec_theta1_current;
      REAL(prec_theta2_samples)[idx] = prec_theta2_current;
      REAL(prec_theta3_samples)[idx] = prec_theta3_current;
      REAL(prec_y_samples)[idx]   = prec_y_current;
    }

    /* ===== Update Progress Bar ===== */
    if (ii % pb.update_step == 0 || ii == n_iter - 1) {
      progress_bar_update(&pb, ii);
    }

    /* ===== Update Previous Values for Next Iteration ===== */
    /* Efficient element-wise copy for theta_1, theta_2, and theta_3 trajectories */
    for (int j = 0; j < n; j++) {
      theta_1_previous[j] = theta_1_current[j];
      theta_2_previous[j] = theta_2_current[j];
      theta_3_previous[j] = theta_3_current[j];
    }
    theta_01_previous = theta_01_current;
    theta_02_previous = theta_02_current;
    theta_03_previous = theta_03_current;
    prec_theta1_previous = prec_theta1_current;
    prec_theta2_previous = prec_theta2_current;
    prec_theta3_previous = prec_theta3_current;
    prec_y_previous   = prec_y_current;
  }

  /* ========== Finalize Progress Bar ========== */
  progress_bar_finish(&pb, n_draws);

  /* ========== Restore RNG State ========== */
  PutRNGstate();

  /* ========== Free Temporary Buffers ========== */
  R_Free(theta_1_current);
  R_Free(theta_1_previous);
  R_Free(theta_2_current);
  R_Free(theta_2_previous);
  R_Free(theta_3_current);
  R_Free(theta_3_previous);

  /* ========== Package Results into Named List ========== */
  SEXP out = PROTECT(allocVector(VECSXP, 10));
  SET_VECTOR_ELT(out, 0, theta_1_samples);
  SET_VECTOR_ELT(out, 1, theta_2_samples);
  SET_VECTOR_ELT(out, 2, theta_3_samples);
  SET_VECTOR_ELT(out, 3, theta_01_samples);
  SET_VECTOR_ELT(out, 4, theta_02_samples);
  SET_VECTOR_ELT(out, 5, theta_03_samples);
  SET_VECTOR_ELT(out, 6, prec_theta1_samples);
  SET_VECTOR_ELT(out, 7, prec_theta2_samples);
  SET_VECTOR_ELT(out, 8, prec_theta3_samples);
  SET_VECTOR_ELT(out, 9, prec_y_samples);

  SEXP nms = PROTECT(allocVector(STRSXP, 10));
  SET_STRING_ELT(nms, 0, mkChar("theta_1"));
  SET_STRING_ELT(nms, 1, mkChar("theta_2"));
  SET_STRING_ELT(nms, 2, mkChar("theta_3"));
  SET_STRING_ELT(nms, 3, mkChar("theta_01"));
  SET_STRING_ELT(nms, 4, mkChar("theta_02"));
  SET_STRING_ELT(nms, 5, mkChar("theta_03"));
  SET_STRING_ELT(nms, 6, mkChar("prec_theta1"));
  SET_STRING_ELT(nms, 7, mkChar("prec_theta2"));
  SET_STRING_ELT(nms, 8, mkChar("prec_theta3"));
  SET_STRING_ELT(nms, 9, mkChar("prec_y"));
  setAttrib(out, R_NamesSymbol, nms);

  UNPROTECT(12);
  return out;
}
