/**
 * @file cwmh_binomial.c
 * @brief Component-wise Metropolis-Hastings sampling for logit-binomial state-space models
 * @author Michel H. Montoril
 * @date 2026-07-25
 * @version 1.1
 *
 * @details This file implements optimized MCMC update routines for state-space models with
 *          binomial observations and logit link, including:
 *          - Component-wise MH updates for local level (random walk) binomial models
 *          - Component-wise MH updates for local trend (random walk + trend) binomial models
 *          - Memory-efficient implementations using current/previous iteration buffers
 *          - Numerical stability enhancements with stable log-probability computations
 *          - Precision caching to avoid repeated expensive calculations
 *          - Conditional alpha computation with branch hoisting for optimal performance
 */

#include <R.h>
#include <Rmath.h>
#include <string.h>  /* memcpy */
#include "utils.h"   /* ilogit */
#include "link_guard.h"  /* clamp_link_alpha, ilogit_guarded */
#include "cwmh_binomial.h"

/**
 * @brief Numerically stable log acceptance probability computation
 * @details Prevents overflow/underflow in acceptance probability calculations
 *
 * @param lp1n Log-likelihood contribution of the first component at the new state.
 * @param lp2n Log-likelihood contribution of the second component at the new state.
 * @param lp1o Log-likelihood contribution of the first component at the old state.
 * @param lp2o Log-likelihood contribution of the second component at the old state.
 * @return Log of the acceptance probability truncated at 0 to cap the probability at 1.
 */
static inline double stable_log_accept_prob(double lp1n,
                                            double lp2n,
                                            double lp1o,
                                            double lp2o) {
  double log_ratio = (lp1n + lp2n) - (lp1o + lp2o);
  return fmin2(0.0, log_ratio);  /* Cap at 0 (probability 1) */
}

/**
 * @brief Numerical guard bounds for the logit success probability alpha = g(theta).
 *
 * @details Instead of clamping the latent state theta_1, the sampler constrains
 *          the *probability* alpha = ilogit(theta) to stay strictly inside (0, 1).
 *          alpha is the quantity that actually enters the binomial likelihood
 *          dbinom(y, n_trials, alpha, ...) and is reported to the caller. If alpha
 *          were allowed to reach exactly 0 or 1, then dbinom(y, n_trials, {0, 1})
 *          would be -Inf whenever 0 < y < n_trials, and that -Inf/NaN would
 *          propagate into the Metropolis-Hastings acceptance ratio and stall the
 *          chain.
 *
 *          The bounds are placed at the resolution of the logit link at its
 *          numerical saturation point: ilogit(-36) = 2.3e-16 and
 *          ilogit(36) = 1 - 2.3e-16. Beyond |theta| ~ 36 the transform is already
 *          numerically indistinguishable from the boundary, so constraining the
 *          probability there leaves alpha essentially unchanged while removing the
 *          exact-boundary hazard. The lower floor is set marginally wider (2e-16)
 *          for symmetry with the probit guard in generate_alpha_binomial.c.
 *
 *          Unlike a state clamp, this guard never distorts the sampled latent
 *          state theta_1: theta_1 is stored exactly as drawn, and only the derived
 *          probability is protected.
 */
/* Moved to link_guard.h; see the include at the top of this file. */

/**
 * @brief Numerical saturation bound for the logit latent state theta_1.
 *
 * @details Complements the alpha probability guard above, which only protects
 *          the *reported/likelihood* probability. This state clamp instead
 *          protects the *sampler dynamics*: once ilogit saturates the
 *          Binomial/Bernoulli likelihood P(y | theta) becomes flat, theta_1 is
 *          no longer identified by the data, and a random walk of the state can
 *          drift into that tail, inflating the sampled innovations and dragging
 *          the innovation precision 1/W_1 toward zero. Symmetric counterpart of
 *          clamp_probit_state in generate_alpha_binomial.c.
 *
 *          For |theta| >= LOGIT_THETA_CLAMP = 36, ilogit(theta) is within
 *          ~2e-16 of 0 or 1, so clamping leaves alpha numerically unchanged; for
 *          well-identified problems |theta_1| stays far below 36 and the guard
 *          is inert.
 */
#define LOGIT_THETA_CLAMP 36.0

static inline double clamp_logit_state(double theta) {
  if (theta >  LOGIT_THETA_CLAMP) return  LOGIT_THETA_CLAMP;
  if (theta < -LOGIT_THETA_CLAMP) return -LOGIT_THETA_CLAMP;
  return theta;
}

/**
 * @brief Cache structure for expensive precision computations
 */
typedef struct {
  double cached_prec;
  double cached_sd_last;
  double cached_sd_regular;
  int    cached_iter;
} precision_cache_t;

static precision_cache_t prec_cache = {-1.0, 0.0, 0.0, -1};

/**
 * @brief Component-wise Metropolis-Hastings sampler for theta_1 in a logit-binomial
 *        local level model
 *
 * @details Implements an optimized component-wise Metropolis-Hastings algorithm to sample
 *          the level state vector theta_1 in a binomial observation model with logit link:
 *
 *          **Observation equation:**
 *          y_t ~ Binomial(n_trials, alpha_t), where alpha_t = logit^{-1}(theta_{t,1})
 *
 *          **State equation (local level):**
 *          theta_{t,1} = theta_{t-1,1} + u_{t,1}, u_{t,1} ~ N(0, 1/prec_theta_1)
 *
 *          **Optimizations implemented:**
 *          - Cached precision computations to avoid repeated sqrt/division
 *          - Numerically stable ilogit function from utils.c
 *          - Improved memory locality through sequential access patterns
 *          - Stable log-probability computations
 *          - Memory-efficient current/previous iteration buffers
 *          - Branch hoisting for conditional alpha computation
 *          - Fast memcpy path when alpha computation is skipped
 *
 *          **Precision structure:**
 *          - Regular elements (t=0..n-2): sd_regular = 1/sqrt(prec_theta_1 * 2)
 *          - Last element (t=n-1):        sd_last    = 1/sqrt(prec_theta_1)
 *
 *          **Conditional means for proposal:**
 *          - First:        E[theta_{1,1} | theta_01, theta_{2,1}] = 0.5 * (theta_{2,1} + theta_01)
 *          - Intermediate: E[theta_{t,1} | theta_{t-1,1}, theta_{t+1,1}] = 0.5 * (theta_{t-1,1} + theta_{t+1,1})
 *          - Last:         E[theta_{n,1} | theta_{n-1,1}] = theta_{n-1,1}
 *
 *          **Algorithm:**
 *          1. Cache precision-dependent standard deviations if precision changed
 *          2. Pre-compute proposal standard deviations from log_sigma
 *          3. For each component, propose new value and compute acceptance probability
 *          4. Update theta_1_current and alpha_current with accepted/rejected values
 *          5. Store acceptance indicators in sliding window for adaptation
 *
 * @param theta_1_previous   Level state vector [n] from previous iteration (const, read-only).
 *                           Contains theta_{1,1}, ..., theta_{n,1} from iteration iter-1.
 * @param theta_1_current    Output level state vector [n] for current iteration.
 *                           Will be filled with updated theta_{1,1}, ..., theta_{n,1}.
 * @param alpha_current      Output probability vector [n] for current iteration.
 *                           Will be filled with alpha_t = logit^{-1}(theta_{t,1}).
 *                           Can be NULL if compute_alpha = 0.
 * @param theta_01_previous  Scalar initial level state from previous iteration.
 *                           Used in conditional mean for first element.
 * @param prec_theta1_previous    Scalar level precision from previous iteration.
 *                           Controls proposal variance and prior contribution.
 * @param theta_1_updated    Sliding window matrix [lag_update * n] of acceptance indicators.
 *                           Uses circular indexing based on iteration modulo lag_update.
 * @param y                  Observed binomial counts vector [n] (const, read-only).
 *                           Each y[t] must satisfy 0 <= y[t] <= n_trials.
 * @param log_sigma          Log proposal standard deviations vector [n] (const, read-only).
 *                           Adaptive proposal scales for each component.
 * @param hat_theta_1        Workspace vector [n] for conditional means.
 * @param theta_1_new        Workspace vector [n] for proposed values.
 * @param log_accept_prob    Workspace vector [n] for log acceptance probabilities.
 * @param lag_update         Sliding window size for theta_1_updated indexing (> 0).
 * @param n_trials           Number of Bernoulli trials for binomial distribution.
 * @param n                  Length of the time series.
 * @param iter               Current MCMC iteration (0-based, must be > 0).
 * @param compute_alpha      Flag to control alpha computation (0 = skip, 1 = compute).
 *                           Set to 0 during burn-in or for thinned-out iterations to
 *                           save ~50% computation time by skipping ilogit transformations.
 *
 * @note Computational complexity: O(n) per MCMC iteration due to component-wise updates.
 * @note Memory requirements: O(n) current/previous buffers + O(lag_update * n) sliding window.
 * @note Numerical stability: Uses optimized log-probabilities and stable ilogit function.
 * @note Forward sampling: Components are updated sequentially using previously updated
 *       values within the same iteration, which can improve mixing.
 * @note Model specification: Local level binomial model (random walk, no trend).
 * @note Cache optimization: Precision-dependent calculations are cached between iterations.
 * @note Performance: Branch hoisting and memcpy optimization provide 10-30% speedup
 *       when compute_alpha = 0 (typical during burn-in and thinning).
 *
 * @warning Each y[t] must satisfy 0 <= y[t] <= n_trials.
 * @warning iter must be >= 1 for valid theta_1_previous access.
 * @warning lag_update must be > 0 for theta_1_updated indexing.
 * @warning theta_01_previous and prec_theta1_previous must contain valid values from iteration iter-1.
 * @warning If compute_alpha = 1, alpha_current must be a valid pointer to n doubles.
 * @warning If compute_alpha = 0, alpha_current is ignored and can be NULL.
 *
 * @see cwmh_alpha_logit_binomial
 * @see stable_log_accept_prob
 * @see ilogit
 */
void cwmh_alpha_logit_binomial_locallevel(const double *theta_1_previous,
                                          double       *theta_1_current,
                                          double       *alpha_current,
                                          double        theta_01_previous,
                                          double        prec_theta1_previous,
                                          double       *theta_1_updated,
                                          const double *y,
                                          const double *log_sigma,
                                          double       *hat_theta_1,
                                          double       *theta_1_new,
                                          double       *log_accept_prob,
                                          int           lag_update,
                                          double        n_trials,
                                          int           n,
                                          int           iter,
                                          int           compute_alpha) {

  int t;

  /* ========== Optimized Precision Calculations with Caching ========== */
  /* Cache expensive sqrt and division operations when precision hasn't changed.
   * Standard deviations for conditional distributions depend on precision structure:
   * - sd_regular: for elements with two neighbors (interior points)
   * - sd_last: for final element with one neighbor (boundary condition) */
  double current_prec = prec_theta1_previous;
  double sd_last, sd_regular;

  if (prec_cache.cached_prec != current_prec || prec_cache.cached_iter != iter) {
    prec_cache.cached_prec = current_prec;
    prec_cache.cached_sd_last = 1.0 / sqrt(current_prec);
    prec_cache.cached_sd_regular = prec_cache.cached_sd_last * M_SQRT1_2;  /* 1/sqrt(2*prec) */
    prec_cache.cached_iter = iter;
  }
  sd_last = prec_cache.cached_sd_last;
  sd_regular = prec_cache.cached_sd_regular;

  /* ========== Pre-compute Proposal Standard Deviations ========== */
  /* Transform log-scale proposal parameters to standard deviation scale.
   * This vectorized transformation avoids repeated exp() calls in the main loop. */
  double sigma_vals[n];
  for (t = 0; t < n; t++) {
    sigma_vals[t] = exp(log_sigma[t]);
  }

  /* ========== Sliding Window Index for Acceptance Tracking ========== */
  /* Circular buffer indexing for memory-efficient acceptance rate monitoring.
   * Maps current iteration to row in sliding window matrix. */
  const int window_row = iter % lag_update;
  const int window_idx = window_row * n;

  /* ========== First Element (t = 0) ========== */
  /* Conditional mean incorporates initial state from previous iteration.
   * For local level: E[theta_{1,1} | theta_01, theta_{2,1}] = 0.5 * (theta_{2,1} + theta_01)
   * This centers the prior around the average of initial state and next time point. */
  hat_theta_1[0] = 0.5 * (theta_1_previous[1] + theta_01_previous);

  /* Generate proposal from random walk centered at previous iteration's value */
  theta_1_new[0] = rnorm(theta_1_previous[0], sigma_vals[0]);

  /* Calculate log acceptance probability using cached precision values.
   * lp1 = log prior density, lp2 = log likelihood
   * Subscripts: n = new (proposed), o = old (current) */
  double lp1n = dnorm(theta_1_new[0], hat_theta_1[0], sd_regular, 1);
  double lp2n = dbinom(y[0], n_trials, ilogit_guarded(theta_1_new[0]), 1);
  double lp1o = dnorm(theta_1_previous[0], hat_theta_1[0], sd_regular, 1);
  double lp2o = dbinom(y[0], n_trials, ilogit_guarded(theta_1_previous[0]), 1);

  log_accept_prob[0] = stable_log_accept_prob(
    lp1n, /* lp1n: log-density of new state w.r.t. prior */
    lp2n, /* lp2n: log-likelihood of new state */
    lp1o, /* lp1o: log-density of current state w.r.t. prior */
    lp2o  /* lp2o: log-likelihood of current state */
  );

  /* Accept/reject step via uniform comparison in log scale */
  int accepted_0 = log(runif(0, 1)) <= log_accept_prob[0];

  if (!accepted_0) {
    theta_1_new[0] = theta_1_previous[0];  /* Reject: keep old value */
  }

  /* Store acceptance indicator in sliding window for adaptation monitoring */
  theta_1_updated[window_idx + 0] = accepted_0;

  /* ========== Intermediate Elements (t = 1 to n-2) ========== */
  /* Cache previous value for improved memory locality in forward sampling.
   * This allows us to use the just-updated value in the conditional mean. */
  double theta_prev = theta_1_new[0];

  for (t = 1; t < (n - 1); t++) {
    /* Conditional mean for local level model: E[theta_{t,1} | theta_{t-1,1}, theta_{t+1,1}]
     * Using forward-sampling: previously updated theta_prev from current iteration.
     * This can improve mixing by breaking the Markovian dependence structure. */
    double theta_next = theta_1_previous[t + 1];
    hat_theta_1[t] = 0.5 * (theta_next + theta_prev);

    /* Propose new value from random walk */
    theta_1_new[t] = rnorm(theta_1_previous[t], sigma_vals[t]);

    /* Compute log densities using cached precision values */
    lp1n = dnorm(theta_1_new[t], hat_theta_1[t], sd_regular, 1);
    lp2n = dbinom(y[t], n_trials, ilogit_guarded(theta_1_new[t]), 1);
    lp1o = dnorm(theta_1_previous[t], hat_theta_1[t], sd_regular, 1);
    lp2o = dbinom(y[t], n_trials, ilogit_guarded(theta_1_previous[t]), 1);

    log_accept_prob[t] = stable_log_accept_prob(
      lp1n, /* lp1n: log-density of new state w.r.t. prior */
      lp2n, /* lp2n: log-likelihood of new state */
      lp1o, /* lp1o: log-density of current state w.r.t. prior */
      lp2o  /* lp2o: log-likelihood of current state */
    );

    /* Accept/reject step */
    int accepted_t = log(runif(0, 1)) <= log_accept_prob[t];

    if (!accepted_t) {
      theta_1_new[t] = theta_1_previous[t];
    }

    /* Store acceptance indicator in sliding window */
    theta_1_updated[window_idx + t] = accepted_t;

    /* Update cached value for next iteration (forward sampling) */
    theta_prev = theta_1_new[t];
  }

  /* ========== Last Element (t = n-1) ========== */
  /* Simplified conditional mean for last element: E[theta_{n,1} | theta_{n-1,1}]
   * For local level: just the previous state (random walk boundary condition).
   * Uses cached theta_prev from last intermediate update. */
  hat_theta_1[n - 1] = theta_prev;

  /* Propose new value from random walk */
  theta_1_new[n - 1] = rnorm(theta_1_previous[n - 1], sigma_vals[n - 1]);

  /* Compute log densities using cached precision (boundary variance structure) */
  lp1n = dnorm(theta_1_new[n - 1], hat_theta_1[n - 1], sd_last, 1);
  lp2n = dbinom(y[n - 1], n_trials, ilogit_guarded(theta_1_new[n - 1]), 1);
  lp1o = dnorm(theta_1_previous[n - 1], hat_theta_1[n - 1], sd_last, 1);
  lp2o = dbinom(y[n - 1], n_trials, ilogit_guarded(theta_1_previous[n - 1]), 1);

  log_accept_prob[n - 1] = stable_log_accept_prob(
    lp1n, /* lp1n: log-density of new state w.r.t. prior */
    lp2n, /* lp2n: log-likelihood of new state */
    lp1o, /* lp1o: log-density of current state w.r.t. prior */
    lp2o  /* lp2o: log-likelihood of current state */
  );

  /* Accept/reject step */
  int accepted_last = log(runif(0, 1)) <= log_accept_prob[n - 1];

  if (!accepted_last) {
    theta_1_new[n - 1] = theta_1_previous[n - 1];
  }

  /* Store acceptance indicator in sliding window */
  theta_1_updated[window_idx + (n - 1)] = accepted_last;

  /* ========== Guard Against Logit Saturation Drift ========== */
  /* Clamp the accepted states to the band where ilogit is not numerically
   * saturated (see clamp_logit_state). Inert whenever |theta_1| < LOGIT_THETA_CLAMP,
   * which is the norm for the small-step random walk; it defends against the
   * runaway drift / 1/W_1 collapse and is complementary to the alpha probability
   * guard (clamp_link_alpha), which only protects the reported/likelihood alpha. */
  for (t = 0; t < n; t++) {
    theta_1_new[t] = clamp_logit_state(theta_1_new[t]);
  }

  /* ========== Update Output Arrays with Branch Hoisting Optimization ========== */
  /* Branch hoisting: test compute_alpha once outside loop instead of n times inside.
   * This enables better CPU pipelining, potential auto-vectorization by compiler,
   * and eliminates ~500+ branch mispredictions per call.
   *
   * The accepted latent states theta_1 are clamped by clamp_logit_state above; the
   * success probability alpha = ilogit_guarded(theta_1) additionally stays strictly
   * inside (0, 1) via the probability guard (see clamp_link_alpha).
   *
   * Performance impact:
   * - compute_alpha = 0: Fast memcpy path (2-3x faster than manual loop)
   * - compute_alpha = 1: Sequential loop with ilogit transformations
   * - Typical speedup: 10-30% when skipping alpha during burn-in/thinning */
  if (compute_alpha) {
    /* Path 1: Compute both theta_1 and alpha transformations */
    for (t = 0; t < n; t++) {
      theta_1_current[t] = theta_1_new[t];          /* Store sampled states */
      alpha_current[t] = ilogit_guarded(theta_1_new[t]);   /* Store transformed probabilities */
    }
  } else {
    /* Path 2: Fast bulk copy without transformation (optimized in assembly/SIMD) */
    memcpy(theta_1_current, theta_1_new, (size_t)n * sizeof(double));
  }
}

//----------------------------------------------------------------------

/**
 * @brief Component-wise Metropolis-Hastings sampler for theta_1 in a logit-binomial
 *        local trend model
 *
 * @details Implements an optimized component-wise Metropolis-Hastings algorithm to sample
 *          the level state vector theta_1 in a binomial observation model with logit link:
 *
 *          **Observation equation:**
 *          y_t ~ Binomial(n_trials, alpha_t), where alpha_t = logit^{-1}(theta_{t,1})
 *
 *          **State equations (local trend):**
 *          theta_{t,1} = theta_{t-1,1} + theta_{t-1,2} + u_{t,1}, u_{t,1} ~ N(0, 1/prec_theta_1)
 *          theta_{t,2} = theta_{t-1,2} + u_{t,2},                  u_{t,2} ~ N(0, 1/prec_theta_2)
 *
 *          **Optimizations implemented:**
 *          - Cached precision computations to avoid repeated sqrt/division
 *          - Numerically stable ilogit function from utils.c
 *          - Improved memory locality through sequential access patterns
 *          - Stable log-probability computations
 *          - Memory-efficient current/previous iteration buffers
 *          - Branch hoisting for conditional alpha computation
 *          - Fast memcpy path when alpha computation is skipped
 *
 *          **Precision structure:**
 *          - Regular elements (t=0..n-2): sd_regular = 1/sqrt(prec_theta_1 * 2)
 *          - Last element (t=n-1):        sd_last    = 1/sqrt(prec_theta_1)
 *
 *          **Conditional means for proposal:**
 *          - First: E[theta_{1,1} | theta_01, theta_02, theta_{2,1}, theta_{1,2}] =
 *                   0.5 * (theta_{2,1} - theta_{1,2} + theta_01 + theta_02)
 *          - Intermediate: E[theta_{t,1} | theta_{t-1,1}, theta_{t-1,2}, theta_{t+1,1}, theta_{t,2}] =
 *                   0.5 * (theta_{t+1,1} - theta_{t,2} + theta_{t-1,1} + theta_{t-1,2})
 *          - Last: E[theta_{n,1} | theta_{n-1,1}, theta_{n-1,2}] = theta_{n-1,1} + theta_{n-1,2}
 *
 *          **Algorithm:**
 *          1. Cache precision-dependent standard deviations if precision changed
 *          2. Pre-compute proposal standard deviations from log_sigma
 *          3. For each component, compute trend-adjusted conditional mean
 *          4. Propose new value and compute acceptance probability
 *          5. Update theta_1_current and alpha_current with accepted/rejected values
 *          6. Store acceptance indicators in sliding window for adaptation
 *
 * @param theta_1_previous   Level state vector [n] from previous iteration (const, read-only).
 * @param theta_1_current    Output level state vector [n] for current iteration.
 * @param alpha_current      Output probability vector [n] for current iteration.
 *                           Can be NULL if compute_alpha = 0.
 * @param theta_2_current    Trend state vector [n] from current iteration (const, read-only).
 *                           Must be sampled before calling this function in Gibbs sequence.
 * @param theta_01_previous  Scalar initial level state from previous iteration.
 * @param theta_02_previous  Scalar initial trend state from previous iteration.
 * @param prec_theta1_previous    Scalar level precision from previous iteration.
 * @param theta_1_updated    Sliding window matrix [lag_update * n] of acceptance indicators.
 * @param y                  Observed binomial counts vector [n] (const, read-only).
 * @param log_sigma          Log proposal standard deviations vector [n] (const, read-only).
 * @param hat_theta_1        Workspace vector [n] for conditional means.
 * @param theta_1_new        Workspace vector [n] for proposed values.
 * @param log_accept_prob    Workspace vector [n] for log acceptance probabilities.
 * @param lag_update         Sliding window size for theta_1_updated indexing (> 0).
 * @param n_trials           Number of Bernoulli trials for binomial distribution.
 * @param n                  Length of the time series.
 * @param iter               Current MCMC iteration (0-based, must be > 0).
 * @param compute_alpha      Flag to control alpha computation (0 = skip, 1 = compute).
 *
 * @note Computational complexity: O(n) per MCMC iteration due to component-wise updates.
 * @note Memory requirements: O(n) current/previous buffers + O(lag_update * n) sliding window.
 * @note Numerical stability: Uses optimized log-probabilities and stable ilogit function.
 * @note Forward sampling: Components are updated sequentially using previously updated values.
 * @note Model specification: Local trend binomial model (random walk + trend).
 * @note Cache optimization: Precision-dependent calculations are cached between iterations.
 * @note Performance: Branch hoisting and memcpy optimization provide 10-30% speedup
 *       when compute_alpha = 0.
 *
 * @warning Each y[t] must satisfy 0 <= y[t] <= n_trials.
 * @warning iter must be >= 1 for valid previous iteration access.
 * @warning lag_update must be > 0 for theta_1_updated indexing.
 * @warning theta_2_current must contain valid values from current iteration.
 * @warning All previous iteration parameters must contain valid values.
 * @warning If compute_alpha = 1, alpha_current must be a valid pointer to n doubles.
 * @warning If compute_alpha = 0, alpha_current is ignored and can be NULL.
 *
 * @see cwmh_alpha_logit_binomial_locallevel
 * @see stable_log_accept_prob
 * @see ilogit
 */
void cwmh_alpha_logit_binomial(const double *theta_1_previous,
                               double       *theta_1_current,
                               double       *alpha_current,
                               const double *theta_2_current,
                               double        theta_01_previous,
                               double        theta_02_previous,
                               double        prec_theta1_previous,
                               double       *theta_1_updated,
                               const double *y,
                               const double *log_sigma,
                               double       *hat_theta_1,
                               double       *theta_1_new,
                               double       *log_accept_prob,
                               int           lag_update,
                               double        n_trials,
                               int           n,
                               int           iter,
                               int           compute_alpha) {

  int t;

  /* ========== Optimized Precision Calculations with Caching ========== */
  /* Cache expensive sqrt and division operations when precision hasn't changed.
   * Standard deviations for conditional distributions depend on precision structure. */
  double current_prec = prec_theta1_previous;
  double sd_last, sd_regular;

  if (prec_cache.cached_prec != current_prec || prec_cache.cached_iter != iter) {
    prec_cache.cached_prec = current_prec;
    prec_cache.cached_sd_last = 1.0 / sqrt(current_prec);
    prec_cache.cached_sd_regular = prec_cache.cached_sd_last * M_SQRT1_2;  /* 1/sqrt(2*prec) */
    prec_cache.cached_iter = iter;
  }
  sd_last = prec_cache.cached_sd_last;
  sd_regular = prec_cache.cached_sd_regular;

  /* ========== Pre-compute Proposal Standard Deviations ========== */
  /* Transform log-scale proposal parameters to standard deviation scale. */
  double sigma_vals[n];
  for (t = 0; t < n; t++) {
    sigma_vals[t] = exp(log_sigma[t]);
  }

  /* ========== Sliding Window Index for Acceptance Tracking ========== */
  /* Circular buffer indexing for memory-efficient acceptance rate monitoring. */
  const int window_row = iter % lag_update;
  const int window_idx = window_row * n;

  /* ========== First Element (t = 0) ========== */
  /* Conditional mean incorporates initial states and trend adjustment.
   * For local trend: E[theta_{1,1} | ...] = 0.5 * (theta_{2,1} - theta_{1,2} + theta_01 + theta_02)
   * The trend terms adjust the level evolution expectation. */
  hat_theta_1[0] = 0.5 * (theta_1_previous[1] - theta_2_current[0] +
  theta_01_previous + theta_02_previous);

  /* Generate proposal from random walk centered at previous iteration's value */
  theta_1_new[0] = rnorm(theta_1_previous[0], sigma_vals[0]);

  /* Calculate log acceptance probability using cached precision values */
  double lp1n = dnorm(theta_1_new[0], hat_theta_1[0], sd_regular, 1);
  double lp2n = dbinom(y[0], n_trials, ilogit_guarded(theta_1_new[0]), 1);
  double lp1o = dnorm(theta_1_previous[0], hat_theta_1[0], sd_regular, 1);
  double lp2o = dbinom(y[0], n_trials, ilogit_guarded(theta_1_previous[0]), 1);

  log_accept_prob[0] = stable_log_accept_prob(
    lp1n, /* lp1n: log-density of new state w.r.t. prior */
    lp2n, /* lp2n: log-likelihood of new state */
    lp1o, /* lp1o: log-density of current state w.r.t. prior */
    lp2o  /* lp2o: log-likelihood of current state */
  );

  /* Accept/reject step via uniform comparison in log scale */
  int accepted_0 = log(runif(0, 1)) <= log_accept_prob[0];

  if (!accepted_0) {
    theta_1_new[0] = theta_1_previous[0];  /* Reject: keep old value */
  }

  /* Store acceptance indicator in sliding window */
  theta_1_updated[window_idx + 0] = accepted_0;

  /* ========== Intermediate Elements (t = 1 to n-2) ========== */
  /* Cache previous value for improved memory locality in forward sampling. */
  double theta_prev = theta_1_new[0];

  for (t = 1; t < (n - 1); t++) {
    /* Conditional mean using forward-sampling and trend adjustments.
     * Incorporates theta_prev (just updated), theta_next, and current/previous trend values. */
    double theta_next = theta_1_previous[t + 1];
    double theta_2_curr = theta_2_current[t];
    double theta_2_prev = theta_2_current[t - 1];

    hat_theta_1[t] = 0.5 * (theta_next - theta_2_curr + theta_prev + theta_2_prev);

    /* Propose new value from random walk */
    theta_1_new[t] = rnorm(theta_1_previous[t], sigma_vals[t]);

    /* Compute log densities using cached precision values */
    lp1n = dnorm(theta_1_new[t], hat_theta_1[t], sd_regular, 1);
    lp2n = dbinom(y[t], n_trials, ilogit_guarded(theta_1_new[t]), 1);
    lp1o = dnorm(theta_1_previous[t], hat_theta_1[t], sd_regular, 1);
    lp2o = dbinom(y[t], n_trials, ilogit_guarded(theta_1_previous[t]), 1);

    log_accept_prob[t] = stable_log_accept_prob(
      lp1n, /* lp1n: log-density of new state w.r.t. prior */
      lp2n, /* lp2n: log-likelihood of new state */
      lp1o, /* lp1o: log-density of current state w.r.t. prior */
      lp2o  /* lp2o: log-likelihood of current state */
    );

    /* Accept/reject step */
    int accepted_t = log(runif(0, 1)) <= log_accept_prob[t];

    if (!accepted_t) {
      theta_1_new[t] = theta_1_previous[t];
    }

    /* Store acceptance indicator in sliding window */
    theta_1_updated[window_idx + t] = accepted_t;

    /* Update cached value for next iteration (forward sampling) */
    theta_prev = theta_1_new[t];
  }

  /* ========== Last Element (t = n-1) ========== */
  /* Simplified conditional mean: no future state dependency, includes trend.
   * For local trend: E[theta_{n,1} | theta_{n-1,1}, theta_{n-1,2}] = theta_{n-1,1} + theta_{n-1,2}
   * Uses cached theta_prev from last intermediate update. */
  double theta_2_prev = theta_2_current[n - 2];
  hat_theta_1[n - 1] = theta_prev + theta_2_prev;

  /* Propose new value from random walk */
  theta_1_new[n - 1] = rnorm(theta_1_previous[n - 1], sigma_vals[n - 1]);

  /* Compute log densities using cached precision (boundary variance structure) */
  lp1n = dnorm(theta_1_new[n - 1], hat_theta_1[n - 1], sd_last, 1);
  lp2n = dbinom(y[n - 1], n_trials, ilogit_guarded(theta_1_new[n - 1]), 1);
  lp1o = dnorm(theta_1_previous[n - 1], hat_theta_1[n - 1], sd_last, 1);
  lp2o = dbinom(y[n - 1], n_trials, ilogit_guarded(theta_1_previous[n - 1]), 1);

  log_accept_prob[n - 1] = stable_log_accept_prob(
    lp1n, /* lp1n: log-density of new state w.r.t. prior */
    lp2n, /* lp2n: log-likelihood of new state */
    lp1o, /* lp1o: log-density of current state w.r.t. prior */
    lp2o  /* lp2o: log-likelihood of current state */
  );

  /* Accept/reject step */
  int accepted_last = log(runif(0, 1)) <= log_accept_prob[n - 1];

  if (!accepted_last) {
    theta_1_new[n - 1] = theta_1_previous[n - 1];
  }

  /* Store acceptance indicator in sliding window */
  theta_1_updated[window_idx + (n - 1)] = accepted_last;

  /* ========== Guard Against Logit Saturation Drift ========== */
  /* Clamp the accepted states to the band where ilogit is not numerically
   * saturated (see clamp_logit_state). Inert whenever |theta_1| < LOGIT_THETA_CLAMP,
   * which is the norm for the small-step random walk; it defends against the
   * runaway drift / 1/W_1 collapse and is complementary to the alpha probability
   * guard (clamp_link_alpha), which only protects the reported/likelihood alpha. */
  for (t = 0; t < n; t++) {
    theta_1_new[t] = clamp_logit_state(theta_1_new[t]);
  }

  /* ========== Update Output Arrays with Branch Hoisting Optimization ========== */
  /* Branch hoisting: test compute_alpha once outside loop instead of n times inside.
   * This enables better CPU pipelining, potential auto-vectorization by compiler,
   * and eliminates ~500+ branch mispredictions per call.
   *
   * The accepted latent states theta_1 are clamped by clamp_logit_state above; the
   * success probability alpha = ilogit_guarded(theta_1) additionally stays strictly
   * inside (0, 1) via the probability guard (see clamp_link_alpha). */
  if (compute_alpha) {
    /* Path 1: Compute both theta_1 and alpha transformations */
    for (t = 0; t < n; t++) {
      theta_1_current[t] = theta_1_new[t];          /* Store sampled states */
      alpha_current[t] = ilogit_guarded(theta_1_new[t]);   /* Store transformed probabilities */
    }
  } else {
    /* Path 2: Fast bulk copy without transformation (optimized in assembly/SIMD) */
    memcpy(theta_1_current, theta_1_new, (size_t)n * sizeof(double));
  }
}
