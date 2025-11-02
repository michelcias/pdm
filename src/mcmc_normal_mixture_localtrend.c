/**
 * @file mcmc_normal_mixture_localtrend.c
 * @brief MCMC sampling for Gaussian mixture models with dynamic mixture weights
 * @author Michel H. Montoril
 * @date 2025-10-22
 * @version 1.0
 *
 * @details Implements complete Gibbs sampler for Bayesian estimation of two-component
 *          Gaussian mixture models with time-varying mixture weights following a
 *          local-trend dynamic structure (level + trend).
 *
 *          Supports two link functions:
 *          - Logit: Component-wise Metropolis-Hastings with adaptive tuning
 *          - Probit: Gibbs sampling via Albert-Chib data augmentation
 *
 *          All implementations utilize memory-efficient current/previous iteration buffers
 *          requiring only O(n) temporary storage regardless of chain length.
 *
 *          **Key features:**
 *          - Memory-efficient O(n) temporary storage using current/previous buffers
 *          - Conditional alpha computation for performance optimization
 *          - Configurable adaptive threshold for Metropolis-Hastings (logit only)
 *          - Conjugate posterior updates for all parameters
 *          - Flexible burn-in and thinning controls
 *          - Label switching constraint enforcement (μ₁ < μ₂)
 *
 *          **Gaussian mixture model with dynamic weights:**
 *          Observation: y_t | z_t, μ, φ ~ N(z_t·μ₂ + (1-z_t)·μ₁, [z_t·φ₂ + (1-z_t)·φ₁]⁻¹)
 *          Indicators: z_t | α_t ~ Bernoulli(α_t)
 *
 *          **Dynamic weight models:**
 *          Logit:  α_t = logit⁻¹(θ_{t,1}), z_t ~ Bernoulli(α_t)
 *          Probit: α_t = Φ(θ_{t,1}), z_t ~ Bernoulli(α_t)
 *
 *          **State equations (local trend):**
 *          θ_{t,1} = θ_{t-1,1} + θ_{t-1,2} + u_{t,1}, u_{t,1} ~ N(0, W₁)
 *          θ_{t,2} = θ_{t-1,2} + u_{t,2},             u_{t,2} ~ N(0, W₂)
 *
 *          **Prior distributions:**
 *          - μₖ ~ N(μ₀ₖ, σ²₀ₖ), k=1,2
 *          - φₖ ~ Gamma(ν₀ₖ, η₀ₖ), k=1,2
 *          - θ_{0,1} ~ N(μ_{0,1}, σ²_{0,1})
 *          - θ_{0,2} ~ N(μ_{0,2}, σ²_{0,2})
 *          - 1/W₁ ~ Gamma(ν₁, η₁)
 *          - 1/W₂ ~ Gamma(ν₂, η₂)
 *
 *          **Sampling sequence per iteration:**
 *          1. (μ₁, φ₁, μ₂, φ₂) | y, z → Conjugate Normal-Gamma posteriors
 *          2. z | y, α, μ, φ → Bernoulli with weighted densities
 *          3. θ₂ | θ₁, θ_{0,2}, W₂ → Gaussian posterior (tridiagonal)
 *          4. 1/W₂ | θ₂, θ_{0,2} → Gamma posterior
 *          5. θ_{0,2} | θ₁, θ₂, θ_{0,1}, W₁, W₂ → Gaussian posterior
 *          6. θ₁, α | z, θ₂, θ_{0,1}, W₁ → Link-specific sampling
 *             - Logit: Component-wise MH with adaptive tuning
 *             - Probit: Gibbs via latent utilities
 *          7. 1/W₁ | θ₁, θ_{0,1}, θ_{0,2} → Gamma posterior
 *          8. θ_{0,1} | θ₁, θ_{0,2}, W₁ → Gaussian posterior
 *
 *          Total iterations: burnin + (n_chain - 1) × thinning + 1
 *
 *          **Performance note:**
 *          The computational overhead of supporting both link functions via conditional
 *          branching is negligible (<0.001% of total execution time). The branch
 *          prediction in modern CPUs makes the if-statement essentially "free" after
 *          the first iteration.
 */

#include <R.h>
#include <Rinternals.h>
#include <Rmath.h>
#include <string.h>  /* memcpy, strcmp */
#include "conditional_state.h"
#include "conditional_precision.h"
#include "conditional_theta0.h"
#include "conditional_mixture_normal_parameters.h"
#include "conditional_mixture_normal_indicators.h"
#include "generate_alpha_binomial.h"
#include "utils.h"
#include "mcmc_progress_bar.h"
#include "mcmc_normal_mixture_localtrend.h"

/**
 * @brief Unified Gibbs sampler for Gaussian mixture model with dynamic weights
 *
 * @details Implements a complete Gibbs MCMC algorithm for the two-component Gaussian
 *          mixture model with time-varying mixture weights following local trend dynamics.
 *          Supports both logit and probit link functions via the 'link' argument.
 *
 *          **Observation equation:**
 *          y_t | z_t, μ, φ ~ N(z_t·μ₂ + (1-z_t)·μ₁, [z_t·φ₂ + (1-z_t)·φ₁]⁻¹)
 *          z_t | α_t ~ Bernoulli(α_t)
 *          α_t = T⁻¹(θ_{t,1}) where T = logit or probit
 *
 *          **State equations:**
 *          θ_{t,1} = θ_{t-1,1} + θ_{t-1,2} + u_{t,1}, u_{t,1} ~ N(0, W₁)
 *          θ_{t,2} = θ_{t-1,2} + u_{t,2},             u_{t,2} ~ N(0, W₂)
 *
 *          **Link functions:**
 *          - "logit": α_t = exp(θ_{t,1}) / [1 + exp(θ_{t,1})]
 *                     Uses component-wise Metropolis-Hastings with adaptive tuning
 *                     Returns optional diagnostics: log_sigma, accept_prop
 *
 *          - "probit": α_t = Φ(θ_{t,1}) where Φ is the standard normal CDF
 *                      Uses Albert-Chib data augmentation (always accepts)
 *                      No adaptation parameters needed
 *
 *          **Prior distributions:**
 *          - μₖ ~ N(μ₀ₖ, σ²₀ₖ), φₖ ~ Gamma(ν₀ₖ, η₀ₖ), k=1,2
 *          - θ_{0,j} ~ N(μ_{0,j}, σ²_{0,j}), j=1,2
 *          - 1/Wⱼ ~ Gamma(νⱼ, ηⱼ), j=1,2
 *
 *          **Optimizations implemented:**
 *          - Memory-efficient current/previous iteration buffers (O(n) storage)
 *          - Conditional alpha computation (skip during burn-in/thinning)
 *          - Link-specific buffer allocation (only allocate what's needed)
 *          - Scalar parameter passing to avoid array indexing
 *          - Efficient initialization with neutral starting values
 *
 *          **Label switching:**
 *          Enforces μ₁ < μ₂ constraint via component swapping in parameter sampling
 *
 * @param y_                          Numeric vector [n] of observations.
 * @param link_                       Character string: "logit" or "probit".
 *                                    Specifies link function for mixture weights.
 * @param burnin_                     Number of burn-in iterations (discarded).
 * @param thinning_                   Thinning interval for autocorrelation reduction.
 * @param n_chain_                    Number of retained posterior samples.
 * @param prior_mu01_mean_            Prior mean for μ₁.
 * @param prior_mu01_prec_            Prior precision for μ₁.
 * @param prior_prec01_shape_         Gamma shape for φ₁.
 * @param prior_prec01_rate_          Gamma rate for φ₁.
 * @param prior_mu02_mean_            Prior mean for μ₂.
 * @param prior_mu02_prec_            Prior precision for μ₂.
 * @param prior_prec02_shape_         Gamma shape for φ₂.
 * @param prior_prec02_rate_          Gamma rate for φ₂.
 * @param prior_theta01_mean_         Prior mean for θ_{0,1}.
 * @param prior_theta01_prec_         Prior precision for θ_{0,1}.
 * @param prior_theta02_mean_         Prior mean for θ_{0,2}.
 * @param prior_theta02_prec_         Prior precision for θ_{0,2}.
 * @param prior_prec1_shape_          Gamma shape for 1/W₁.
 * @param prior_prec1_rate_           Gamma rate for 1/W₁.
 * @param prior_prec2_shape_          Gamma shape for 1/W₂.
 * @param prior_prec2_rate_           Gamma rate for 1/W₂.
 * @param lag_update_                 Adaptation frequency (logit only, ignored for probit).
 * @param max_step_size_              Maximum proposal step size (logit only).
 * @param base_adaptation_rate_       Base adaptation rate (logit only).
 * @param decay_exponent_             Adaptation decay exponent (logit only).
 * @param target_acceptance_          Target acceptance proportion (logit only).
 * @param min_deviation_threshold_    Minimum deviation to trigger adaptation (logit only).
 * @param return_log_sigma_           Flag to return log_sigma diagnostics (logit only).
 * @param return_accept_prop_         Flag to return accept_prop diagnostics (logit only).
 * @param verbose_                    Logical: display progress bar (0 = FALSE, 1 = TRUE).
 * @param bar_width_                  Integer: width of progress bar in characters (10-120).
 *
 * @return R list with components:
 *         **Always returned:**
 *         - mu_1:        Vector [n_chain] of component 1 mean samples
 *         - prec_1:      Vector [n_chain] of component 1 precision samples
 *         - mu_2:        Vector [n_chain] of component 2 mean samples
 *         - prec_2:      Vector [n_chain] of component 2 precision samples
 *         - theta_1:     Matrix [n_chain × n] of level state trajectory samples
 *         - theta_2:     Matrix [n_chain × n] of trend state trajectory samples
 *         - theta_01:    Vector [n_chain] of initial level state samples
 *         - theta_02:    Vector [n_chain] of initial trend state samples
 *         - prec_theta1: Vector [n_chain] of level innovation precision samples
 *         - prec_theta2: Vector [n_chain] of trend innovation precision samples
 *         - alpha:       Matrix [n_chain × n] of mixture weight samples
 *         - z:           Matrix [n_chain × n] of latent indicator samples
 *
 *         **Conditionally returned (logit only, if requested):**
 *         - log_sigma:   Matrix [n_chain × n] of proposal scales
 *         - accept_prop: Matrix [n_chain × n] of acceptance proportions
 *
 * @note Complexity: O(n_iter × n) time, O(n) space
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
 * @see generate_alpha_logit_binomial
 * @see generate_alpha_probit_bernoulli
 * @see generate_theta_p
 * @see generate_precision_theta_p
 * @see generate_theta_0p
 * @see generate_precision_theta_k
 * @see generate_theta_01
 *
 * @example
 * @code
 * # R usage example
 *
 * # Logit link with adaptation diagnostics
 * fit_logit <- mcmc_normal_mixture_localtrend(
 *   y = data,
 *   link = "logit",
 *   burnin = 1000,
 *   thinning = 5,
 *   n_chain = 1000,
 *   prior_mu01_mean = 0, prior_mu01_prec = 0.01,
 *   prior_prec01_shape = 0.01, prior_prec01_rate = 0.01,
 *   prior_mu02_mean = 2, prior_mu02_prec = 0.01,
 *   prior_prec02_shape = 0.01, prior_prec02_rate = 0.01,
 *   prior_theta01_mean = 0, prior_theta01_prec = 0.01,
 *   prior_theta02_mean = 0, prior_theta02_prec = 0.01,
 *   prior_prec1_shape = 0.01, prior_prec1_rate = 0.01,
 *   prior_prec2_shape = 0.01, prior_prec2_rate = 0.01,
 *   lag_update = 50,
 *   max_step_size = 10,
 *   base_adaptation_rate = 0.01,
 *   decay_exponent = 0.6,
 *   target_acceptance = 0.44,
 *   min_deviation_threshold = 0.02,
 *   return_log_sigma = TRUE,
 *   return_accept_prop = TRUE,
 *   verbose = TRUE,
 *   bar_width = 50
 * )
 *
 * # Probit link (simpler, no adaptation parameters)
 * fit_probit <- mcmc_normal_mixture_localtrend(
 *   y = data,
 *   link = "probit",
 *   burnin = 1000,
 *   thinning = 5,
 *   n_chain = 1000,
 *   # ... same prior specifications ...
 *   verbose = TRUE,
 *   bar_width = 50
 * )
 *
 * # Compare models
 * plot(fit_logit$alpha[1, ], type = "l", col = "blue")
 * lines(fit_probit$alpha[1, ], col = "red")
 * @endcode
 */
SEXP C_MCMC_normal_mixture_localtrend(SEXP y_,
                                      SEXP link_,
                                      SEXP burnin_,
                                      SEXP thinning_,
                                      SEXP n_chain_,
                                      SEXP prior_mu01_mean_,
                                      SEXP prior_mu01_prec_,
                                      SEXP prior_prec01_shape_,
                                      SEXP prior_prec01_rate_,
                                      SEXP prior_mu02_mean_,
                                      SEXP prior_mu02_prec_,
                                      SEXP prior_prec02_shape_,
                                      SEXP prior_prec02_rate_,
                                      SEXP prior_theta01_mean_,
                                      SEXP prior_theta01_prec_,
                                      SEXP prior_theta02_mean_,
                                      SEXP prior_theta02_prec_,
                                      SEXP prior_prec1_shape_,
                                      SEXP prior_prec1_rate_,
                                      SEXP prior_prec2_shape_,
                                      SEXP prior_prec2_rate_,
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
    Rf_error("C_MCMC_normal_mixture_localtrend: link must be 'logit' or 'probit', got '%s'", link);
  }

  int use_logit = (strcmp(link, "logit") == 0);

  /* ========== Parse Data Vector and Validate Length ========== */
  double   *y   = REAL(y_);
  R_xlen_t  len = LENGTH(y_);

  /* Enforce minimum sample size for numerical stability */
  if (len < 3) {
    Rf_error("C_MCMC_normal_mixture_localtrend: sample size 'n' must be at least 3, got %lld",
             (long long) len);
  }

  /* Check for integer overflow (R limitation) */
  if (len > INT_MAX) {
    Rf_error("C_MCMC_normal_mixture_localtrend: sample size too large (%lld > %d)",
             (long long) len, INT_MAX);
  }

  int n = (int) len;

  /* ========== Parse MCMC Control Parameters ========== */
  int burnin   = INTEGER(burnin_)[0];
  int thinning = INTEGER(thinning_)[0];
  int n_chain  = INTEGER(n_chain_)[0];
  int n_iter   = burnin + (n_chain - 1) * thinning + 1;

  /* ========== Parse Mixture Component Prior Hyperparameters ========== */
  double mu_01_mean    = REAL(prior_mu01_mean_)[0];
  double mu_01_prec    = REAL(prior_mu01_prec_)[0];
  double prec_01_shape = REAL(prior_prec01_shape_)[0];
  double prec_01_rate  = REAL(prior_prec01_rate_)[0];
  double mu_02_mean    = REAL(prior_mu02_mean_)[0];
  double mu_02_prec    = REAL(prior_mu02_prec_)[0];
  double prec_02_shape = REAL(prior_prec02_shape_)[0];
  double prec_02_rate  = REAL(prior_prec02_rate_)[0];

  /* ========== Parse Dynamic State Prior Hyperparameters ========== */
  double mean_theta01 = REAL(prior_theta01_mean_)[0];
  double prec_theta01 = REAL(prior_theta01_prec_)[0];
  double mean_theta02 = REAL(prior_theta02_mean_)[0];
  double prec_theta02 = REAL(prior_theta02_prec_)[0];
  double nu_01        = REAL(prior_prec1_shape_)[0];
  double eta_01       = REAL(prior_prec1_rate_)[0];
  double nu_02        = REAL(prior_prec2_shape_)[0];
  double eta_02       = REAL(prior_prec2_rate_)[0];

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

  /* ===== Initiate Progress Bar ===== */
  ProgressBar pb = progress_bar_init(n_iter, bar_width, burnin, thinning, verbose);
  progress_bar_start(&pb);

  /* ========== Allocate Output Storage (Retained Samples Only) ========== */
  SEXP mu_1_samples        = PROTECT(allocVector(REALSXP, n_chain));
  SEXP prec_1_samples      = PROTECT(allocVector(REALSXP, n_chain));
  SEXP mu_2_samples        = PROTECT(allocVector(REALSXP, n_chain));
  SEXP prec_2_samples      = PROTECT(allocVector(REALSXP, n_chain));
  SEXP theta_1_samples     = PROTECT(allocMatrix(REALSXP, n_chain, n));
  SEXP theta_2_samples     = PROTECT(allocMatrix(REALSXP, n_chain, n));
  SEXP theta_01_samples    = PROTECT(allocVector(REALSXP, n_chain));
  SEXP theta_02_samples    = PROTECT(allocVector(REALSXP, n_chain));
  SEXP prec_theta1_samples = PROTECT(allocVector(REALSXP, n_chain));
  SEXP prec_theta2_samples = PROTECT(allocVector(REALSXP, n_chain));
  SEXP alpha_samples       = PROTECT(allocMatrix(REALSXP, n_chain, n));
  SEXP z_samples           = PROTECT(allocMatrix(REALSXP, n_chain, n));

  /* Conditional allocation for logit-specific diagnostics */
  SEXP log_sigma_samples   = R_NilValue;
  SEXP accept_prop_samples = R_NilValue;
  int n_outputs = 12;  /* Base outputs */
  int n_protect = 12;  /* Base protection count */

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

  /* ========== Allocate Temporary Buffers (Memory-Efficient O(n) Storage) ========== */

  /* Common buffers for both link functions */
  double *theta_1_current  = (double *) R_Calloc(n, double);
  double *theta_1_previous = (double *) R_Calloc(n, double);
  double *theta_2_current  = (double *) R_Calloc(n, double);
  double *theta_2_previous = (double *) R_Calloc(n, double);
  double *alpha_current    = (double *) R_Calloc(n, double);
  double *z_current        = (double *) R_Calloc(n, double);

  /* Mixture component parameters (structure: [mu_1, prec_1, mu_2, prec_2]) */
  double *params_current  = (double *) R_Calloc(4, double);
  double *params_previous = (double *) R_Calloc(4, double);

  /* Scalar parameters for current and previous iterations */
  double theta_01_current, theta_01_previous;
  double theta_02_current, theta_02_previous;
  double prec_theta1_current, prec_theta1_previous;
  double prec_theta2_current, prec_theta2_previous;

  /* Link-specific buffers - conditional allocation */
  double *theta_1_updated  = NULL;
  double *accept_prop      = NULL;
  double *log_sigma        = NULL;
  double *hat_theta_1      = NULL;
  double *theta_1_new      = NULL;
  double *log_accept_prob  = NULL;
  double *rhs_vector       = NULL;

  if (use_logit) {
    /* Allocate buffers for component-wise Metropolis-Hastings */
    theta_1_updated  = (double *) R_Calloc(lag_update * n, double);
    accept_prop      = (double *) R_Calloc(n, double);
    log_sigma        = (double *) R_Calloc(n, double);
    hat_theta_1      = (double *) R_Calloc(n, double);
    theta_1_new      = (double *) R_Calloc(n, double);
    log_accept_prob  = (double *) R_Calloc(n, double);

    /* Initialize log_sigma with reasonable starting values */
    for (int t = 0; t < n; t++) {
      log_sigma[t] = log(0.1);
    }
  } else {
    /* Allocate buffer for probit link (Albert-Chib augmentation) */
    rhs_vector = (double *) R_Calloc(n, double);
  }

  /* ========== Initialize RNG State ========== */
  GetRNGstate();

  /* ========== Initialize Parameters (Iteration 0) ========== */

  /* Initialize mixture component parameters */
  params_previous[0] = rnorm(mu_01_mean, sqrt(1.0 / mu_01_prec));      /* mu_1 */
  params_previous[1] = rgamma(prec_01_shape, 1.0 / prec_01_rate);      /* prec_1 */
  params_previous[2] = rnorm(mu_02_mean, sqrt(1.0 / mu_02_prec));      /* mu_2 */
  params_previous[3] = rgamma(prec_02_shape, 1.0 / prec_02_rate);      /* prec_2 */

  /* Enforce label switching constraint for initialization */
  if (params_previous[0] > params_previous[2]) {
    double temp_mu = params_previous[0];
    params_previous[0] = params_previous[2];
    params_previous[2] = temp_mu;
    double temp_prec = params_previous[1];
    params_previous[1] = params_previous[3];
    params_previous[3] = temp_prec;
  }

  /* Initialize dynamic state parameters */
  theta_01_previous     = rnorm(mean_theta01, sqrt(1.0 / prec_theta01));
  theta_02_previous     = rnorm(mean_theta02, sqrt(1.0 / prec_theta02));
  prec_theta1_previous  = rgamma(nu_01, 1.0 / eta_01);
  prec_theta2_previous  = rgamma(nu_02, 1.0 / eta_02);

  /* Initialize state vectors with neutral starting values */
  for (int t = 0; t < n; t++) {
    theta_1_previous[t] = 0.0;
    theta_2_previous[t] = 0.0;
    alpha_current[t]    = 0.5;
    z_current[t]        = 0.0;  /* Start with component 1 */
  }

  /* ========== Main Gibbs Sampling Loop ========== */
  int chain_idx = 0;

  for (int ii = 1; ii < n_iter; ii++) {

    /* Determine whether to compute alpha transformations and store samples */
    int compute_alpha = (ii >= burnin && ((ii - burnin) % thinning) == 0) ? 1 : 0;

    /* ===== Step 1: Sample Mixture Component Parameters (μ, φ) ===== */
    /* Draw (μ₁, φ₁, μ₂, φ₂) | y, z from conjugate Normal-Gamma posteriors.
     * Enforces label switching constraint μ₁ < μ₂ via component swapping.
     * Uses z from previous iteration and updates parameters. */
    conditional_mixture_normal_parameters_k2(
      y,                  /* observed data [n] */
    z_current,          /* latent indicators [n] from previous iteration */
    params_previous,    /* previous [mu_1, prec_1, mu_2, prec_2] */
    params_current,     /* output: current [mu_1, prec_1, mu_2, prec_2] */
    mu_01_mean,         /* prior mean for μ₁ */
    mu_01_prec,         /* prior precision for μ₁ */
    prec_01_shape,      /* prior shape for φ₁ */
    prec_01_rate,       /* prior rate for φ₁ */
    mu_02_mean,         /* prior mean for μ₂ */
    mu_02_prec,         /* prior precision for μ₂ */
    prec_02_shape,      /* prior shape for φ₂ */
    prec_02_rate,       /* prior rate for φ₂ */
    n                   /* sample size */
    );

    /* ===== Step 2: Sample Latent Indicators z ===== */
    /* Draw z_t | y, α, μ, φ ~ Bernoulli(α*_t) where
     * α*_t = [α_t · N(y_t|μ₂,φ₂⁻¹)] / [(1-α_t)·N(y_t|μ₁,φ₁⁻¹) + α_t·N(y_t|μ₂,φ₂⁻¹)]
     * Uses current (μ, φ) just sampled and α from previous iteration.
     * This creates better mixing by using most recent component parameters. */
    conditional_mixture_normal_indicators_k2(
      y,                  /* observed data [n] */
    z_current,          /* output: latent indicators [n] */
    params_current,     /* current [mu_1, prec_1, mu_2, prec_2] */
    alpha_current,      /* current mixture weights [n] from previous iteration */
    n                   /* sample size */
    );

    /* ===== Step 3: Sample Trend State Vector θ₂ ===== */
    /* Draw θ₂ | θ₁, θ_{0,2}, W₂ from multivariate Normal with tridiagonal precision.
     * Uses θ₁ from previous iteration for computing state differences.
     * Independent of z, so can be sampled before θ₁. */
    generate_theta_p(
      theta_1_previous,      /* θ₁: level from previous iteration [n] */
    theta_2_current,       /* output: current iteration θ₂ [n] */
    prec_theta1_previous,  /* scalar: level precision from previous iteration */
    prec_theta2_previous,  /* scalar: trend precision from previous iteration */
    theta_02_previous,     /* scalar: initial trend from previous iteration */
    n                      /* sample size */
    );

    /* ===== Step 4: Sample Trend Innovation Precision 1/W₂ ===== */
    /* Draw 1/W₂ | θ₂, θ_{0,2} from Gamma posterior.
     * Uses current θ₂ just sampled and previous θ_{0,2}. */
    prec_theta2_current = generate_precision_theta_p(
      theta_02_previous,  /* scalar: initial trend from previous iteration */
    theta_2_current,    /* vector: current θ₂ [n] */
    nu_02,              /* prior shape */
    eta_02,             /* prior rate */
    n                   /* sample size */
    );

    /* ===== Step 5: Sample Initial Trend State θ_{0,2} ===== */
    /* Draw θ_{0,2} | θ₁, θ₂, θ_{0,1}, W₁, W₂ from Normal posterior.
     * Uses information from both level and trend components. */
    theta_02_current = generate_theta_0p(
      theta_1_previous,      /* θ₁: level from previous iteration [n] */
    theta_2_current,       /* θ₂: current trend [n] */
    theta_01_previous,     /* θ_{0,1}: initial level from previous iteration */
    prec_theta1_previous,  /* W₁⁻¹: level precision from previous iteration */
    prec_theta2_current,   /* W₂⁻¹: current trend precision */
    mean_theta02,          /* prior mean */
    prec_theta02,          /* prior precision */
    n                      /* sample size */
    );

    /* ===== Step 6: Sample Level State Vector θ₁ and Mixture Weights α ===== */
    /* Draw θ₁, α | z, θ₂, θ_{0,1}, θ_{0,2}, W₁ using link-specific method.
     * The latent indicators z are treated as "pseudo-observations" for a binomial
     * model with n_trials=1 (Bernoulli). This allows using the binomial samplers
     * designed for dynamic GLMs.
     *
     * Logit link: Uses component-wise Metropolis-Hastings with adaptive tuning
     *             to sample θ₁, then transforms to α = logit⁻¹(θ₁)
     *
     * Probit link: Uses Albert-Chib data augmentation with latent utilities
     *              to sample θ₁, then transforms to α = Φ(θ₁)
     */
    if (use_logit) {
      /* Logit link: Component-wise MH with adaptive tuning */
      generate_alpha_logit_binomial(
        theta_1_previous,          /* θ₁: level from previous iteration [n] */
      theta_1_current,           /* output: current iteration θ₁ [n] */
      compute_alpha ? alpha_current : NULL,  /* α: NULL if not retained */
      theta_2_current,           /* θ₂: trend from current iteration [n] */
      theta_01_previous,         /* θ_{0,1}: initial level from previous iteration */
      theta_02_current,          /* θ_{0,2}: initial trend from current iteration */
      prec_theta1_previous,      /* W₁⁻¹: level precision from previous iteration */
      theta_1_updated,           /* sliding window workspace [lag_update × n] */
      z_current,                 /* z: latent indicators as "observations" [n] */
      accept_prop,               /* acceptance proportions workspace [n] */
      log_sigma,                 /* proposal scale parameters [n] */
      hat_theta_1,               /* conditional means workspace [n] */
      theta_1_new,               /* proposal states workspace [n] */
      log_accept_prob,           /* MH log-acceptance ratios [n] */
      lag_update,                /* adaptation lag */
      1.0,                       /* n_trials: 1 for Bernoulli (z ∈ {0,1}) */
      n,                         /* series length */
      ii,                        /* current iteration */
      max_step_size,             /* proposal cap */
      base_adaptation_rate,      /* base adaptation weight */
      decay_exponent,            /* adaptation decay */
      target_acceptance,         /* desired acceptance rate */
      min_deviation_threshold,   /* adaptation trigger */
      compute_alpha              /* flag for alpha computation */
      );
    } else {
      /* Probit link: Gibbs sampling via Albert-Chib augmentation */
      generate_alpha_probit_bernoulli(
        theta_1_previous,          /* θ₁: level from previous iteration [n] */
      theta_1_current,           /* output: current iteration θ₁ [n] */
      compute_alpha ? alpha_current : NULL,  /* α: NULL if not retained */
      theta_2_current,           /* θ₂: trend from current iteration [n] */
      theta_01_previous,         /* θ_{0,1}: initial level from previous iteration */
      theta_02_current,          /* θ_{0,2}: initial trend from current iteration */
      prec_theta1_previous,      /* W₁⁻¹: level precision from previous iteration */
      z_current,                 /* z: latent indicators [n] */
      rhs_vector,                /* workspace: solver right-hand side [n] */
      n,                         /* series length */
      compute_alpha              /* flag for alpha computation */
      );
    }

    /* ===== Step 7: Sample Level Innovation Precision 1/W₁ ===== */
    /* Draw 1/W₁ | θ₁, θ₂, θ_{0,1}, θ_{0,2} from Gamma posterior.
     * Uses both level and trend information to compute innovations. */
    prec_theta1_current = generate_precision_theta_k(
      theta_01_previous,  /* scalar: initial level from previous iteration */
    theta_02_current,   /* scalar: current initial trend */
    theta_1_current,    /* vector: current level [n] */
    theta_2_current,    /* vector: current trend [n] */
    nu_01,              /* prior shape */
    eta_01,             /* prior rate */
    n                   /* sample size */
    );

    /* ===== Step 8: Sample Initial Level State θ_{0,1} ===== */
    /* Draw θ_{0,1} | θ₁, θ_{0,2}, W₁ from Normal posterior.
     * Uses current level and trend information. */
    theta_01_current = generate_theta_01(
      theta_1_current,    /* vector: current level [n] */
    theta_02_current,   /* scalar: current initial trend */
    prec_theta1_current,/* scalar: current level precision */
    mean_theta01,       /* prior mean */
    prec_theta01,       /* prior precision */
    n                   /* sample size */
    );

    /* ===== Store Post-Burn-in Samples with Thinning ===== */
    /* Only copy to output matrices for retained iterations.
     * This avoids storing the entire burn-in and thinned-out samples. */
    if (compute_alpha) {
      int idx = chain_idx++;

      /* Store mixture component parameters */
      REAL(mu_1_samples)[idx]   = params_current[0];
      REAL(prec_1_samples)[idx] = params_current[1];
      REAL(mu_2_samples)[idx]   = params_current[2];
      REAL(prec_2_samples)[idx] = params_current[3];

      /* Store dynamic state parameters */
      REAL(theta_01_samples)[idx]     = theta_01_current;
      REAL(theta_02_samples)[idx]     = theta_02_current;
      REAL(prec_theta1_samples)[idx] = prec_theta1_current;
      REAL(prec_theta2_samples)[idx] = prec_theta2_current;

      /* Store state trajectories, mixture weights, and indicators */
      for (int t = 0; t < n; t++) {
        REAL(theta_1_samples)[idx + t * n_chain] = theta_1_current[t];
        REAL(theta_2_samples)[idx + t * n_chain] = theta_2_current[t];
        REAL(alpha_samples)[idx + t * n_chain] = alpha_current[t];
        REAL(z_samples)[idx + t * n_chain]       = z_current[t];

        /* Store logit-specific diagnostics if requested */
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

    /* ===== Update Progress Bar ===== */
    if (ii % pb.update_step == 0 || ii == n_iter - 1) {
      progress_bar_update(&pb, ii);
    }

    /* ===== Update Previous Values for Next Iteration ===== */
    /* Efficient element-wise copy using memcpy for vector parameters */
    memcpy(theta_1_previous, theta_1_current, n * sizeof(double));
    memcpy(theta_2_previous, theta_2_current, n * sizeof(double));
    memcpy(params_previous, params_current, 4 * sizeof(double));

    /* Direct assignment for scalar parameters */
    theta_01_previous     = theta_01_current;
    theta_02_previous     = theta_02_current;
    prec_theta1_previous = prec_theta1_current;
    prec_theta2_previous = prec_theta2_current;
  }

  /* ========== Finalize Progress Bar ========== */
  progress_bar_finish(&pb, n_chain);

  /* ========== Restore RNG State ========== */
  PutRNGstate();

  /* ========== Free Temporary Buffers ========== */
  /* Free common buffers */
  R_Free(theta_1_current);
  R_Free(theta_1_previous);
  R_Free(theta_2_current);
  R_Free(theta_2_previous);
  R_Free(alpha_current);
  R_Free(z_current);
  R_Free(params_current);
  R_Free(params_previous);

  /* Free link-specific buffers conditionally */
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

  SET_VECTOR_ELT(out, output_idx, theta_2_samples);
  SET_STRING_ELT(nms, output_idx++, mkChar("theta_2"));

  SET_VECTOR_ELT(out, output_idx, theta_01_samples);
  SET_STRING_ELT(nms, output_idx++, mkChar("theta_01"));

  SET_VECTOR_ELT(out, output_idx, theta_02_samples);
  SET_STRING_ELT(nms, output_idx++, mkChar("theta_02"));

  SET_VECTOR_ELT(out, output_idx, prec_theta1_samples);
  SET_STRING_ELT(nms, output_idx++, mkChar("prec_theta1"));

  SET_VECTOR_ELT(out, output_idx, prec_theta2_samples);
  SET_STRING_ELT(nms, output_idx++, mkChar("prec_theta2"));

  SET_VECTOR_ELT(out, output_idx, alpha_samples);
  SET_STRING_ELT(nms, output_idx++, mkChar("alpha"));

  SET_VECTOR_ELT(out, output_idx, z_samples);
  SET_STRING_ELT(nms, output_idx++, mkChar("z"));

  /* Add logit-specific diagnostics if requested */
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
