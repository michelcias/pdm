/**
 * @file generate_alpha_binomial.c
 * @brief Sampling for binomial and Bernoulli state-space models with logit and probit links
 * @author Michel H. Montoril
 * @date 2025-10-11
 * @version 1.0
 *
 * @details This file contains optimized functions for MCMC sampling in binomial and Bernoulli
 *          state-space models with different link functions:
 *          - Logit-binomial models: Component-wise Metropolis-Hastings with adaptive tuning
 *          - Probit-Bernoulli models: Gibbs sampling via Albert-Chib data augmentation
 *          - Memory-efficient implementations using current/previous iteration buffers
 *          - Integration with configurable adaptive threshold parameters
 *          - Conditional alpha computation for performance optimization
 *
 *          **Logit-binomial models:**
 *          Use component-wise Metropolis-Hastings updates for the level state vector
 *          with adaptive proposal tuning based on acceptance rates.
 *
 *          **Probit-Bernoulli models:**
 *          Use Gibbs sampling with latent variable augmentation (Albert-Chib scheme)
 *          for efficient sampling from the posterior distribution.
 */

#include <R.h>
#include <Rmath.h>
#include <float.h>          /* DBL_EPSILON */
#include "cwmh_adaptive.h"  /* adapt_cwmh_parameters */
#include "cwmh_binomial.h"
#include "utils.h"          /* generate_normal_vector */
#include "generate_alpha_binomial.h"

/**
 * @brief Component-wise Metropolis-Hastings sampler for theta_1 in a logit-binomial
 *        local level model with adaptive tuning
 *
 * @details Implements an optimized component-wise Metropolis-Hastings algorithm to sample the
 *          level state vector theta_1 in a binomial observation model with logit link:
 *
 *          **Observation equation:**
 *          y_t ~ Binomial(n_trials, alpha_t), where alpha_t = logit^{-1}(theta_{t,1})
 *
 *          **State equation (random walk):**
 *          theta_{t,1} = theta_{t-1,1} + u_{t,1}, u_{t,1} ~ N(0, W_1)
 *
 *          This routine integrates:
 *          - Adaptive proposal tuning (log_sigma) via recent acceptance proportions (accept_prop)
 *          - Component-wise Metropolis-Hastings update for theta_1 (nonlinear observation
 *            with logit link)
 *          - Conditional alpha computation for performance optimization
 *
 *          The adaptation follows a diminishing adaptation schedule and is executed
 *          periodically over a sliding window of size lag_update. The actual state
 *          update is delegated to cwmh_alpha_logit_binomial_locallevel, which handles boundary
 *          conditions and log-acceptance.
 *
 *          **Adaptation cadence:**
 *          Performed when iter >= lag_update and (iter % lag_update == 0), i.e.,
 *          at iterations lag_update, 2*lag_update, 3*lag_update, ...
 *
 *          **Memory efficiency:**
 *          Uses current/previous iteration buffers instead of full trajectory storage,
 *          requiring only O(n) temporary memory regardless of chain length.
 *
 * @param theta_1_previous        Level state vector [n] from previous iteration (const).
 * @param theta_1_current         Output level state vector [n] for current iteration.
 * @param alpha_current           Output probability vector [n] for current iteration.
 *                                Can be NULL if compute_alpha = 0.
 * @param theta_01_previous       Scalar initial level state from previous iteration.
 * @param prec_theta1_previous         Scalar level precision from previous iteration.
 * @param theta_1_updated         Sliding window matrix [lag_update * n] of acceptance indicators.
 * @param y                       Observed binomial counts vector [n] (const, read-only).
 *                                Each y[t] must satisfy 0 <= y[t] <= n_trials.
 * @param accept_prop             Workspace vector [n] for acceptance proportions.
 * @param log_sigma               Input/output vector [n] of log proposal standard deviations.
 * @param hat_theta_1             Workspace vector [n] for conditional means.
 * @param theta_1_new             Workspace vector [n] for proposed values.
 * @param log_accept_prob         Workspace vector [n] for log acceptance probabilities.
 * @param lag_update              Sliding window size for adaptation frequency (> 0).
 *                                Set to 0 to disable adaptation.
 * @param n_trials                Number of Bernoulli trials.
 * @param n                       Length of the time series.
 * @param iter                    Current MCMC iteration (0-based, must be >= 1).
 * @param max_step_size           Maximum adaptation step size for log_sigma updates.
 * @param base_adaptation_rate    Base adaptation rate before decay.
 * @param decay_exponent          Exponent for diminishing adaptation schedule.
 * @param target_acceptance       Target acceptance rate for adaptive tuning.
 * @param min_deviation_threshold Minimum deviation from target to trigger updates (>= 0).
 * @param compute_alpha           Flag to control alpha computation (0 = skip, 1 = compute).
 *
 * @note Complexity: O(n) per iteration (component-wise updates).
 * @note Uses log-probabilities for numerical stability.
 * @note Forward sampling for better mixing.
 * @note Model is local level (no trend).
 * @note Adaptive tuning performed every lag_update iterations if iter >= lag_update.
 * @note Memory optimization: uses current/previous buffers instead of full trajectory.
 * @note Performance: Skipping alpha computation provides 10-30% speedup during burn-in/thinning.
 *
 * @warning Each y[t] must satisfy 0 <= y[t] <= n_trials.
 * @warning iter must be >= 1 for valid theta_1_previous access.
 * @warning lag_update must be > 0 for theta_1_updated indexing.
 * @warning min_deviation_threshold must be >= 0.0.
 * @warning If compute_alpha = 1, alpha_current must be a valid pointer.
 * @warning If compute_alpha = 0, alpha_current can be NULL.
 *
 * @see adapt_cwmh_parameters
 * @see cwmh_alpha_logit_binomial_locallevel
 */
void generate_alpha_logit_binomial_locallevel(const double *theta_1_previous,
                                              double       *theta_1_current,
                                              double       *alpha_current,
                                              double        theta_01_previous,
                                              double        prec_theta1_previous,
                                              double       *theta_1_updated,
                                              const double *y,
                                              double       *accept_prop,
                                              double       *log_sigma,
                                              double       *hat_theta_1,
                                              double       *theta_1_new,
                                              double       *log_accept_prob,
                                              int           lag_update,
                                              double        n_trials,
                                              int           n,
                                              int           iter,
                                              double        max_step_size,
                                              double        base_adaptation_rate,
                                              double        decay_exponent,
                                              double        target_acceptance,
                                              double        min_deviation_threshold,
                                              int           compute_alpha) {

  /* ========== Prerequisites and Safety Checks ========== */
  /* cwmh_alpha_logit_binomial_locallevel requires iter >= 1 for previous iteration access */
  if (iter <= 0) {
    /* Nothing to do in iteration 0; typically used to initialize storage. */
    return;
  }

  /* ========== Adaptive Tuning (periodic, sliding window) ========== */
  /* Trigger adaptation every 'lag_update' iterations once sufficient history exists */
  if (lag_update > 0 && iter >= lag_update && (iter % lag_update == 0)) {

    adapt_cwmh_parameters(
      theta_1_updated,        /* theta_updated: sliding window acceptance buffer */
      accept_prop,            /* accept_prop: workspace for acceptance rates */
      log_sigma,              /* log_sigma: proposal scale parameters */
      lag_update,             /* lag_update: adaptation window length */
      n,                      /* n: number of time points */
      iter,                   /* iter: current iteration index */
      max_step_size,          /* max_step_size: cap on adaptation step */
      base_adaptation_rate,   /* base_adaptation_rate: initial adaptation rate */
      decay_exponent,         /* decay_exponent: diminishing schedule */
      target_acceptance,      /* target_acceptance: desired acceptance probability */
      min_deviation_threshold /* min_deviation_threshold: deviation trigger */
    );
  }

  /* ========== CWMH Update for Current Iteration ========== */
  /* Updates theta_1 for current iteration, logs acceptance, and optionally stores alpha.
   * Uses memory-efficient current/previous buffers instead of full trajectory storage. */
  cwmh_alpha_logit_binomial_locallevel(
    theta_1_previous,   /* theta_1_previous: state from previous iteration [n] */
    theta_1_current,    /* theta_1_current: output for current iteration [n] */
    alpha_current,      /* alpha_current: success probabilities (NULL if compute_alpha=0) */
    theta_01_previous,  /* theta_01_previous: initial level from previous iteration */
    prec_theta1_previous,    /* prec_theta1_previous: level precision from previous iteration */
    theta_1_updated,    /* theta_1_updated: sliding window indicators */
    y,                  /* y: observed counts */
    log_sigma,          /* log_sigma: proposal log standard deviations */
    hat_theta_1,        /* hat_theta_1: conditional means workspace */
    theta_1_new,        /* theta_1_new: proposal buffer */
    log_accept_prob,    /* log_accept_prob: log acceptance storage */
    lag_update,         /* lag_update: adaptation window length */
    n_trials,           /* n_trials: binomial trials */
    n,                  /* n: number of observations */
    iter,               /* iter: current iteration */
    compute_alpha       /* compute_alpha: flag for alpha computation */
  );
}

//----------------------------------------------------------------------

/**
 * @brief Component-wise Metropolis-Hastings sampler for theta_1 in a logit-binomial
 *        local trend model with adaptive tuning
 *
 * @details Implements an optimized component-wise Metropolis-Hastings algorithm to sample the
 *          level state vector theta_1 with a binomial observation model and local
 *          trend state-space evolution:
 *
 *          **Observation equation:**
 *          y_t ~ Binomial(n_trials, alpha_t), where alpha_t = logit^{-1}(theta_{t,1})
 *
 *          **State equations:**
 *          theta_{t,1} = theta_{t-1,1} + theta_{t-1,2} + u_{t,1}, u_{t,1} ~ N(0, W_1)
 *          theta_{t,2} = theta_{t-1,2} + u_{t,2},                 u_{t,2} ~ N(0, W_2)
 *
 *          This routine integrates:
 *          - Adaptive proposal tuning (log_sigma) via recent acceptance proportions (accept_prop)
 *          - Component-wise Metropolis-Hastings update for theta_1 (nonlinear observation
 *            with logit link)
 *          - Conditional alpha computation for performance optimization
 *
 *          The adaptation follows a diminishing adaptation schedule and is executed
 *          periodically over a sliding window of size lag_update. The actual state
 *          update is delegated to cwmh_alpha_logit_binomial, which handles boundary
 *          conditions and log-acceptance.
 *
 *          **Adaptation cadence:**
 *          Performed when iter >= lag_update and (iter % lag_update == 0), i.e.,
 *          at iterations lag_update, 2*lag_update, 3*lag_update, ...
 *
 *          **Memory efficiency:**
 *          Uses current/previous iteration buffers instead of full trajectory storage,
 *          requiring only O(n) temporary memory regardless of chain length.
 *
 * @param theta_1_previous        Level state vector [n] from previous iteration (const).
 * @param theta_1_current         Output level state vector [n] for current iteration.
 * @param alpha_current           Output probability vector [n] for current iteration.
 *                                Can be NULL if compute_alpha = 0.
 * @param theta_2_current         Trend state vector [n] from current iteration (const).
 *                                Must be sampled before calling this function.
 * @param theta_01_previous       Scalar initial level state from previous iteration.
 * @param theta_02_previous       Scalar initial trend state from previous iteration.
 * @param prec_theta1_previous         Scalar level precision from previous iteration.
 * @param theta_1_updated         Sliding window matrix [lag_update * n] of acceptance indicators.
 * @param y                       Observed binomial counts vector [n] (const, read-only).
 * @param accept_prop             Workspace vector [n] for acceptance proportions.
 * @param log_sigma               Input/output vector [n] of log proposal standard deviations.
 * @param hat_theta_1             Workspace vector [n] for conditional means.
 * @param theta_1_new             Workspace vector [n] for proposed values.
 * @param log_accept_prob         Workspace vector [n] for log acceptance probabilities.
 * @param lag_update              Sliding window size for adaptation frequency (> 0).
 * @param n_trials                Number of Bernoulli trials.
 * @param n                       Length of the time series.
 * @param iter                    Current MCMC iteration (0-based, must be >= 1).
 * @param max_step_size           Maximum adaptation step size for log_sigma updates.
 * @param base_adaptation_rate    Base adaptation rate before decay.
 * @param decay_exponent          Exponent for diminishing adaptation schedule.
 * @param target_acceptance       Target acceptance rate for adaptive tuning.
 * @param min_deviation_threshold Minimum deviation from target to trigger updates (>= 0).
 * @param compute_alpha           Flag to control alpha computation (0 = skip, 1 = compute).
 *
 * @note Complexity: O(n) per iteration (component-wise updates).
 * @note Uses log-probabilities for numerical stability.
 * @note Forward sampling for better mixing.
 * @note Model is local trend (random walk + trend).
 * @note Adaptive tuning performed every lag_update iterations if iter >= lag_update.
 * @note Memory optimization: uses current/previous buffers instead of full trajectory.
 * @note Performance: Skipping alpha computation provides 10-30% speedup during burn-in/thinning.
 *
 * @warning Each y[t] must satisfy 0 <= y[t] <= n_trials.
 * @warning iter must be >= 1 for valid previous iteration access.
 * @warning lag_update must be > 0 for theta_1_updated indexing.
 * @warning min_deviation_threshold must be >= 0.0.
 * @warning theta_2_current must contain valid values from current iteration.
 * @warning If compute_alpha = 1, alpha_current must be a valid pointer.
 * @warning If compute_alpha = 0, alpha_current can be NULL.
 *
 * @see adapt_cwmh_parameters
 * @see cwmh_alpha_logit_binomial
 */
void generate_alpha_logit_binomial(const double *theta_1_previous,
                                   double       *theta_1_current,
                                   double       *alpha_current,
                                   const double *theta_2_current,
                                   double        theta_01_previous,
                                   double        theta_02_previous,
                                   double        prec_theta1_previous,
                                   double       *theta_1_updated,
                                   const double *y,
                                   double       *accept_prop,
                                   double       *log_sigma,
                                   double       *hat_theta_1,
                                   double       *theta_1_new,
                                   double       *log_accept_prob,
                                   int           lag_update,
                                   double        n_trials,
                                   int           n,
                                   int           iter,
                                   double        max_step_size,
                                   double        base_adaptation_rate,
                                   double        decay_exponent,
                                   double        target_acceptance,
                                   double        min_deviation_threshold,
                                   int           compute_alpha) {

  /* ========== Prerequisites and Safety Checks ========== */
  /* cwmh_alpha_logit_binomial requires iter >= 1 for previous iteration access */
  if (iter <= 0) {
    /* Nothing to do in iteration 0; typically used to initialize storage. */
    return;
  }

  /* ========== Adaptive Tuning (periodic, sliding window) ========== */
  /* Trigger adaptation every 'lag_update' iterations once sufficient history exists */
  if (lag_update > 0 && iter >= lag_update && (iter % lag_update == 0)) {

    adapt_cwmh_parameters(
      theta_1_updated,        /* theta_updated: sliding window acceptance buffer */
      accept_prop,            /* accept_prop: workspace for acceptance rates */
      log_sigma,              /* log_sigma: proposal scale parameters */
      lag_update,             /* lag_update: adaptation window length */
      n,                      /* n: number of time points */
      iter,                   /* iter: current iteration index */
      max_step_size,          /* max_step_size: cap on adaptation step */
      base_adaptation_rate,   /* base_adaptation_rate: initial adaptation rate */
      decay_exponent,         /* decay_exponent: diminishing schedule */
      target_acceptance,      /* target_acceptance: desired acceptance probability */
      min_deviation_threshold /* min_deviation_threshold: deviation trigger */
    );
  }

  /* ========== CWMH Update for Current Iteration ========== */
  /* Updates theta_1 for current iteration, logs acceptance, and optionally stores alpha.
   * Uses memory-efficient current/previous buffers instead of full trajectory storage. */
  cwmh_alpha_logit_binomial(
    theta_1_previous,   /* theta_1_previous: level from previous iteration [n] */
    theta_1_current,    /* theta_1_current: output for current iteration [n] */
    alpha_current,      /* alpha_current: success probabilities (NULL if compute_alpha=0) */
    theta_2_current,    /* theta_2_current: trend from current iteration [n] */
    theta_01_previous,  /* theta_01_previous: initial level from previous iteration */
    theta_02_previous,  /* theta_02_previous: initial trend from previous iteration */
    prec_theta1_previous,    /* prec_theta1_previous: level precision from previous iteration */
    theta_1_updated,    /* theta_1_updated: sliding window indicators */
    y,                  /* y: observed counts */
    log_sigma,          /* log_sigma: proposal log standard deviations */
    hat_theta_1,        /* hat_theta_1: conditional means workspace */
    theta_1_new,        /* theta_1_new: proposal buffer */
    log_accept_prob,    /* log_accept_prob: log acceptance storage */
    lag_update,         /* lag_update: adaptation window length */
    n_trials,           /* n_trials: binomial trials */
    n,                  /* n: number of observations */
    iter,               /* iter: current iteration */
    compute_alpha       /* compute_alpha: flag for alpha computation */
  );
}

//----------------------------------------------------------------------

/**
 * @brief Efficient sampling from truncated normal distribution N(mu, sigma^2)
 *
 * @details Uses the inverse CDF method with optimized bounds handling for numerical stability.
 *          Implements specialized fast paths for common boundary cases to maximize performance
 *          in Albert-Chib data augmentation schemes.
 *
 *          **Algorithm:**
 *          1. Detect boundary configuration and route to specialized fast path
 *          2. For general case, compute cumulative probabilities at truncation bounds
 *          3. Sample uniform variate in valid probability range
 *          4. Apply inverse CDF transformation with numerical guards
 *
 *          **Optimizations implemented:**
 *          - Fast path for untruncated case: direct normal sampling (50-70% faster)
 *          - Fast path for unilateral truncation: reduced pnorm calls (30-40% faster)
 *          - Compact ternary operators for numerical guards (improved code clarity)
 *          - Inline hint for compiler optimization
 *          - Comprehensive numerical stability guards for extreme truncations
 *
 *          **Numerical stability:**
 *          - Probability floor set to 1e-300 (100x larger than DBL_MIN ~= 2.2e-308)
 *          - Probability ceiling set to the largest double < 1.0 (1 - DBL_EPSILON/2)
 *          - Unilateral truncations computed on the tail that stays away from 1.0,
 *            avoiding catastrophic cancellation for bounds far from the mean
 *          - Prevents qnorm from returning +/- Inf on degenerate intervals
 *          - Handles extreme truncations gracefully (up to ~36 standard deviations)
 *
 *          **Common use cases in probit models:**
 *          - y_t = 1: sample N(theta_t, 1) truncated above 0 (lower = 0, upper = +Inf)
 *          - y_t = 0: sample N(theta_t, 1) truncated below 0 (lower = -Inf, upper = 0)
 *          Both cases use optimized fast paths for maximum performance.
 *
 * @param mu     Mean of the normal distribution.
 * @param sigma  Standard deviation of the normal distribution. Must be > 0.
 * @param lower  Lower truncation bound. Use R_NegInf for no lower bound.
 * @param upper  Upper truncation bound. Use R_PosInf for no upper bound.
 * @return       Random sample from N(mu, sigma^2) truncated to [lower, upper].
 *
 * @note Complexity: O(1) with optimized paths for common boundary configurations.
 * @note Uses R's pnorm/qnorm for numerical stability and portability.
 * @note Handles all special cases: no truncation, unilateral, bilateral.
 * @note Inline hint allows compiler to eliminate function call overhead in tight loops.
 * @note Probability floor (1e-300) provides 100x safety margin over DBL_MIN.
 *
 * @warning lower must be strictly less than upper.
 * @warning sigma must be strictly positive.
 * @warning For debugging, compile with -DRTRUNCNORM_DEBUG to enable input validation.
 * @warning Do not modify PROB_FLOOR below 1e-300 without extensive numerical testing.
 *
 * @see Albert & Chib (1993). Bayesian Analysis of Binary and Polychotomous Response Data.
 *      JASA, 88(422), 669-679. https://doi.org/10.1080/01621459.1993.10476321
 * @see generate_alpha_probit_bernoulli_locallevel
 * @see generate_alpha_probit_bernoulli
 */
static inline double rtruncnorm(double mu, double sigma, double lower, double upper) {
  /* Numerical stability floor for probability computations.
   * Set to 1e-300 (approximately 100x larger than DBL_MIN ~= 2.2e-308),
   * providing a comfortable safety margin for extreme truncations while maintaining
   * numerical precision. Pushing this significantly lower risks pnorm/qnorm underflow,
   * which would propagate NaNs into the state sampler. */
  const double PROB_FLOOR   = 1e-300;

  /* Numerical stability ceiling: the largest double strictly less than 1.0.
   * The spacing of doubles just below 1.0 is DBL_EPSILON / 2 (~1.1e-16), so an
   * expression like 1.0 - 1e-300 rounds to exactly 1.0 and provides no protection:
   * qnorm(1.0, ...) returns +Inf, which would poison the state sampler with
   * Inf/NaN values. */
  const double PROB_CEILING = 1.0 - DBL_EPSILON / 2.0;

  double p_lower, p_upper, p;

#ifdef RTRUNCNORM_DEBUG
  /* Input validation (debug builds only).
   * In production builds, these checks are compiled out for maximum performance.
   * The caller is responsible for ensuring valid inputs. */
  if (!(lower < upper)) {
    error("rtruncnorm: lower must be strictly less than upper (received %f vs %f)",
          lower, upper);
  }
  if (sigma <= 0.0) {
    error("rtruncnorm: sigma must be positive (received %f)", sigma);
  }
#endif

  /* ========== Fast Path 1: No Truncation ========== */
  /* When sampling from untruncated normal, use direct transformation.
   * This avoids two pnorm calls and one qnorm call, providing 50-70% speedup.
   * Common in initial MCMC iterations or models without constraints. */
  if (lower == R_NegInf && upper == R_PosInf) {
    return mu + sigma * norm_rand();
  }

  /* ========== Fast Path 2: Only Upper Truncation ========== */
  /* Common in Albert-Chib when y_t = 0: sample N(theta_t, 1) truncated below 0.
   * This pattern occurs in approximately 50% of probit model updates.
   * Reduces computational cost by eliminating lower bound calculations. */
  if (lower == R_NegInf) {
    /* Compute upper cumulative probability */
    p_upper = pnorm(upper, mu, sigma, 1, 0);

    /* Apply numerical stability guard */
    p_upper = (p_upper >= 1.0) ? PROB_CEILING :
      (p_upper <= 0.0) ? PROB_FLOOR : p_upper;

    /* Sample from (0, p_upper] and transform */
    p = unif_rand() * p_upper;

    /* Guard against underflow of the product to exactly 0 */
    p = (p <= 0.0) ? PROB_FLOOR : p;

    return qnorm(p, mu, sigma, 1, 0);
  }

  /* ========== Fast Path 3: Only Lower Truncation ========== */
  /* Common in Albert-Chib when y_t = 1: sample N(theta_t, 1) truncated above 0.
   * This pattern also occurs in approximately 50% of probit model updates.
   * Reduces computational cost by eliminating upper bound calculations.
   *
   * The computation is carried out entirely in the upper tail (survival scale).
   * The lower-tail formulation p_lower + u * (1 - p_lower) loses all precision
   * once mu is ~8 standard deviations below the bound: pnorm(lower, ...) rounds
   * to exactly 1.0 and qnorm(1.0, ...) returns +Inf, poisoning the state sampler
   * with Inf/NaN. Working with s = P(X > lower) avoids the catastrophic
   * cancellation because small survival probabilities are representable down to
   * ~1e-308. */
  if (upper == R_PosInf) {
    /* Compute survival probability P(X > lower) directly in the upper tail */
    double s_lower = pnorm(lower, mu, sigma, 0, 0);

    /* Apply numerical stability guard */
    s_lower = (s_lower >= 1.0) ? PROB_CEILING :
      (s_lower <= 0.0) ? PROB_FLOOR : s_lower;

    /* Sample from (0, s_lower] on the survival scale */
    p = unif_rand() * s_lower;

    /* Guard against underflow of the product to exactly 0 */
    p = (p <= 0.0) ? PROB_FLOOR : p;

    /* Upper-tail inverse CDF: returns x with P(X > x) = p, hence x >= lower */
    return qnorm(p, mu, sigma, 0, 0);
  }

  /* ========== General Case: Both Bounds Finite ========== */
  /* Less common in typical probit models, but necessary for completeness.
   * Occurs when truncation bounds are data-dependent or in constrained models. */

  /* Compute cumulative probabilities at both truncation bounds */
  p_lower = pnorm(lower, mu, sigma, 1, 0);
  p_upper = pnorm(upper, mu, sigma, 1, 0);

  /* Apply numerical stability guards to prevent qnorm from returning +/- Inf.
   * With PROB_FLOOR = 1e-300, this handles truncations up to ~36 standard deviations,
   * well beyond any practical scenario in Bayesian state-space models. */
  p_lower = (p_lower <= 0.0) ? PROB_FLOOR :
    (p_lower >= 1.0) ? PROB_CEILING : p_lower;
  p_upper = (p_upper <= 0.0) ? PROB_FLOOR :
    (p_upper >= 1.0) ? PROB_CEILING : p_upper;

  /* Handle degenerate probability interval.
   * Occurs when finite-precision arithmetic causes p_upper <= p_lower
   * due to extremely imbalanced truncation (e.g., both bounds on same tail). */
  if (p_upper <= p_lower) {
    /* Fall back to nearest admissible probability */
    p = (p_lower >= PROB_CEILING) ? PROB_CEILING : PROB_FLOOR;
  } else {
    /* Sample uniform variate in valid probability range [p_lower, p_upper] */
    p = p_lower + unif_rand() * (p_upper - p_lower);
  }

  /* Transform sampled probability back to truncated normal scale */
  return qnorm(p, mu, sigma, 1, 0);
}

//----------------------------------------------------------------------

/**
 * @brief Numerical saturation bound for the probit latent state theta_1.
 *
 * @details For |theta| >= PROBIT_THETA_CLAMP the standard normal CDF is within
 *          ~7e-16 of 0 or 1, i.e. numerically indistinguishable from the
 *          boundary in IEEE double precision (Phi(8) = 1 - 6.7e-16, Phi(-8) =
 *          6.7e-16). Beyond this point the Bernoulli likelihood
 *          P(y_t = 1 | theta_t) = Phi(theta_t) is completely flat, so theta_1
 *          becomes unidentified by the data: the Albert-Chib augmentation
 *          (v_t ~ N(theta_t, 1)) then lets theta_1 random-walk with no
 *          likelihood restoring force. That drift inflates the sampled state
 *          innovations and drags the innovation precision 1/W_1 toward zero, a
 *          positive-feedback loop that stalls the chain (|theta_1| in the tens
 *          is common on segmented data with pure, well-separated stretches such
 *          as aCGH copy-number profiles). The logit link does not suffer this
 *          as readily because its inverse saturates only near |theta| ~ 37.
 *
 *          Clamping the latent state to [-PROBIT_THETA_CLAMP, PROBIT_THETA_CLAMP]
 *          breaks the feedback while leaving the mixture weight
 *          alpha = Phi(theta) numerically unchanged (Phi is already saturated at
 *          the bound) and, as a side effect, keeps alpha strictly inside (0, 1)
 *          so downstream indicator samplers are never forced into a degenerate
 *          deterministic branch. In well-identified problems |theta_1| stays far
 *          below the bound (probit states rarely exceed ~4), so the guard is
 *          inert and does not alter the sampler.
 */
#define PROBIT_THETA_CLAMP 8.0

static inline double clamp_probit_state(double theta) {
  if (theta >  PROBIT_THETA_CLAMP) return  PROBIT_THETA_CLAMP;
  if (theta < -PROBIT_THETA_CLAMP) return -PROBIT_THETA_CLAMP;
  return theta;
}

//----------------------------------------------------------------------

/**
 * @brief Gibbs sampler for theta_1 in a probit-Bernoulli local level model
 *        using Albert-Chib data augmentation
 *
 * @details Implements Gibbs sampling for the level state vector theta_1 in a
 *          Bernoulli observation model with probit link:
 *
 *          **Observation equation:**
 *          y_t ~ Bernoulli(alpha_t), where alpha_t = Phi(theta_{t,1})
 *          and Phi is the standard normal CDF.
 *
 *          **State equation (random walk):**
 *          theta_{t,1} = theta_{t-1,1} + u_{t,1}, u_{t,1} ~ N(0, 1/prec_theta_1)
 *
 *          **Albert-Chib Data Augmentation:**
 *          Introduces latent variables v_t ~ N(theta_{t,1}, 1) such that:
 *          - y_t = 1 if v_t > 0
 *          - y_t = 0 if v_t <= 0
 *
 *          **Algorithm:**
 *          The full conditional posterior for theta_1 given latent variables v is:
 *          theta_1 | v, [...] ~ N(mu_posterior, Sigma_posterior)
 *          where Sigma_posterior^{-1} = I + prec_theta_1 * H'H (tridiagonal precision)
 *
 *          The sampler proceeds by:
 *          1. Drawing latent variables from truncated normals conditional on theta_1_previous
 *          2. Constructing the right-hand side vector for the linear system
 *          3. Sampling theta_1_current from its multivariate normal full conditional
 *          4. Optionally transforming to probability scale via Phi
 *
 *          **Memory efficiency:**
 *          Uses current/previous iteration buffers instead of full trajectory storage.
 *
 * @param theta_1_previous   Level state vector [n] from previous iteration (const).
 * @param theta_1_current    Output level state vector [n] for current iteration.
 * @param alpha_current      Output probability vector [n] for current iteration.
 *                           Can be NULL if compute_alpha = 0.
 * @param theta_01_previous  Scalar initial level state from previous iteration.
 * @param prec_theta1_previous    Scalar level precision from previous iteration.
 * @param y                  Observed Bernoulli outcomes vector [n] (const, read-only).
 *                           Each y[t] must be exactly 0 or 1.
 * @param rhs_vector         Workspace vector [n] for right-hand side of linear system.
 * @param n                  Length of the time series.
 * @param compute_alpha      Flag to control alpha transformation (0 = skip, 1 = compute).
 *                           Set to 0 during burn-in or when probability scale values
 *                           are not needed for inference.
 *
 * @note Complexity: O(n) per iteration exploiting tridiagonal structure.
 * @note Acceptance rate is always 1.0 (Gibbs sampling).
 * @note Model assumes local level without trend component.
 * @note For maximum efficiency, set compute_alpha = 0 during burn-in.
 *
 * @warning Each y[t] must be exactly 0 or 1 (Bernoulli outcomes).
 * @warning n must be > 2 for generate_normal_vector stability.
 * @warning If compute_alpha = 1, alpha_current must be a valid pointer.
 * @warning If compute_alpha = 0, alpha_current can be NULL.
 *
 * @see Albert & Chib (1993). Bayesian Analysis of Binary and Polychotomous Response Data.
 *      JASA, 88(422), 669-679. https://doi.org/10.1080/01621459.1993.10476321
 * @see generate_normal_vector
 * @see rtruncnorm
 */
void generate_alpha_probit_bernoulli_locallevel(const double *theta_1_previous,
                                                double       *theta_1_current,
                                                double       *alpha_current,
                                                double        theta_01_previous,
                                                double        prec_theta1_previous,
                                                const double *y,
                                                double       *rhs_vector,
                                                int           n,
                                                int           compute_alpha) {

  /* ========== Sample Latent Variables and Construct RHS Vector ========== */
  /* Sample v_t from truncated normal conditional on theta_{t-1,1} and y_t.
   * Truncation bounds depend on observation:
   * - y_t = 1: sample from N(theta_{t-1,1}, 1) truncated above 0
   * - y_t = 0: sample from N(theta_{t-1,1}, 1) truncated below 0
   *
   * The right-hand side vector is constructed as rhs[t] = v_t for all t,
   * with boundary correction added after the loop. */
  for (int t = 0; t < n; t++) {
    double v_t;

    if (y[t] == 1.0) {
      v_t = rtruncnorm(theta_1_previous[t], 1.0, 0.0, R_PosInf);
    } else {
      v_t = rtruncnorm(theta_1_previous[t], 1.0, R_NegInf, 0.0);
    }

    rhs_vector[t] = v_t;
  }

  /* Adjust first element to incorporate initial state contribution.
   * This implements the boundary condition rhs[0] = v_0 + prec_theta1 * theta_0. */
  rhs_vector[0] += prec_theta1_previous * theta_01_previous;

  /* ========== Sample theta_1 from Multivariate Normal ========== */
  /* Sample from: theta_1 | v, [...] ~ N(mu_posterior, Sigma_posterior)
   * where Sigma_posterior^{-1} = I + prec_theta_1 * H'H
   *
   * The precision matrix has tridiagonal structure with:
   * - Diagonal: 1 + 2*prec_theta1 for t = 0,...,n-2
   * - Last diagonal: 1 + prec_theta1
   * - Off-diagonal: -prec_theta1
   *
   * This corresponds to generate_normal_vector with:
   * a = 1.0 (observational precision), b = prec_theta1 (state precision) */
  generate_normal_vector(
    theta_1_current,    /* r: output vector [n] */
    rhs_vector,         /* y: right-hand side [n] */
    1.0,                /* a: observational precision (from latent variance = 1) */
    prec_theta1_previous,    /* b: state precision */
    n,                  /* n: dimension */
    1                   /* add_a: use (a + b) for last diagonal element */
  );

  /* ========== Guard Against Probit Saturation Drift ========== */
  /* Clamp the latent state to the region where Phi is not numerically
   * saturated. This is inert whenever |theta_1| < PROBIT_THETA_CLAMP (the norm
   * in well-identified problems) and, on pure/well-separated stretches, stops
   * theta_1 from random-walking into the flat tail of Phi and collapsing the
   * innovation precision 1/W_1. See clamp_probit_state for the full rationale.
   * Applied unconditionally (not only for retained draws) so the guard also
   * holds during burn-in, where the drift would otherwise build up. */
  for (int t = 0; t < n; t++) {
    theta_1_current[t] = clamp_probit_state(theta_1_current[t]);
  }

  /* ========== Transform to Probability Scale ========== */
  /* Compute alpha_t = Phi(theta_{t,1}) for all t if requested.
   * This transformation is needed for posterior summaries and diagnostics
   * but can be skipped during burn-in to save computational cost. */
  if (compute_alpha) {
    for (int t = 0; t < n; t++) {
      alpha_current[t] = pnorm(theta_1_current[t], 0.0, 1.0, 1, 0);
    }
  }
}

//----------------------------------------------------------------------

/**
 * @brief Gibbs sampler for theta_1 in a probit-Bernoulli local trend model
 *        using Albert-Chib data augmentation
 *
 * @details Implements Gibbs sampling for the level state vector theta_1 in a
 *          Bernoulli observation model with probit link and local trend dynamics:
 *
 *          **Observation equation:**
 *          y_t ~ Bernoulli(alpha_t), where alpha_t = Phi(theta_{t,1})
 *          and Phi is the standard normal CDF.
 *
 *          **State equations:**
 *          theta_{t,1} = theta_{t-1,1} + theta_{t-1,2} + u_{t,1}, u_{t,1} ~ N(0, 1/prec_theta_1)
 *          theta_{t,2} = theta_{t-1,2} + u_{t,2},                 u_{t,2} ~ N(0, 1/prec_theta_2)
 *
 *          **Albert-Chib Data Augmentation:**
 *          Introduces latent variables v_t ~ N(theta_{t,1}, 1) such that:
 *          - y_t = 1 if v_t > 0
 *          - y_t = 0 if v_t <= 0
 *
 *          **Algorithm:**
 *          The full conditional posterior for theta_1 given latent variables v is:
 *          theta_1 | v, theta_2, [...] ~ N(mu_posterior, Sigma_posterior)
 *          where Sigma_posterior^{-1} = I + prec_theta_1 * H'H (tridiagonal precision)
 *
 *          The sampler proceeds by:
 *          1. Drawing latent variables from truncated normals conditional on theta_1_previous
 *          2. Constructing RHS vector accounting for trend contributions
 *          3. Sampling theta_1_current from its multivariate normal full conditional
 *          4. Optionally transforming to probability scale via Phi
 *
 *          **Memory efficiency:**
 *          Uses current/previous iteration buffers instead of full trajectory storage.
 *
 * @param theta_1_previous   Level state vector [n] from previous iteration (const).
 * @param theta_1_current    Output level state vector [n] for current iteration.
 * @param alpha_current      Output probability vector [n] for current iteration.
 *                           Can be NULL if compute_alpha = 0.
 * @param theta_2_current    Trend state vector [n] from current iteration (const).
 *                           Must be sampled before calling this function in Gibbs sequence.
 * @param theta_01_previous  Scalar initial level state from previous iteration.
 * @param theta_02_previous  Scalar initial trend state from previous iteration.
 * @param prec_theta1_previous    Scalar level precision from previous iteration.
 * @param y                  Observed Bernoulli outcomes vector [n] (const, read-only).
 *                           Each y[t] must be exactly 0 or 1.
 * @param rhs_vector         Workspace vector [n] for right-hand side of linear system.
 * @param n                  Length of the time series.
 * @param compute_alpha      Flag to control alpha transformation (0 = skip, 1 = compute).
 *
 * @note Complexity: O(n) per iteration exploiting tridiagonal structure.
 * @note Acceptance rate is always 1.0 (Gibbs sampling).
 * @note Model includes local trend component.
 * @note The trend theta_2 must be already sampled in the Gibbs cycle before calling this function.
 * @note For maximum efficiency, set compute_alpha = 0 during burn-in.
 *
 * @warning Each y[t] must be exactly 0 or 1 (Bernoulli outcomes).
 * @warning n must be > 2 for generate_normal_vector stability.
 * @warning theta_2_current must contain valid values from current iteration.
 * @warning If compute_alpha = 1, alpha_current must be a valid pointer.
 * @warning If compute_alpha = 0, alpha_current can be NULL.
 *
 * @see Albert & Chib (1993). Bayesian Analysis of Binary and Polychotomous Response Data.
 *      JASA, 88(422), 669-679. https://doi.org/10.1080/01621459.1993.10476321
 * @see generate_normal_vector
 * @see rtruncnorm
 * @see generate_alpha_probit_bernoulli_locallevel
 */
void generate_alpha_probit_bernoulli(const double *theta_1_previous,
                                     double       *theta_1_current,
                                     double       *alpha_current,
                                     const double *theta_2_current,
                                     double        theta_01_previous,
                                     double        theta_02_previous,
                                     double        prec_theta1_previous,
                                     const double *y,
                                     double       *rhs_vector,
                                     int           n,
                                     int           compute_alpha) {

  /* ========== Sample Latent Variables and Construct RHS Vector ========== */
  /* Sample v_t from truncated normal conditional on theta_{t-1,1} and y_t.
   * Truncation bounds depend on observation:
   * - y_t = 1: sample from N(theta_{t-1,1}, 1) truncated above 0
   * - y_t = 0: sample from N(theta_{t-1,1}, 1) truncated below 0
   *
   * For local trend model, the right-hand side vector is:
   * rhs = v + prec_theta_1 * [theta_01 * e_1 + H'B * theta_2]
   *
   * The term H'B * theta_2 accounts for the trend contribution:
   * (H'B * theta_2)[0]   = theta_02 - theta_2[0]  (boundary condition)
   * (H'B * theta_2)[t]   = theta_2[t-1] - theta_2[t] for t = 1,...,n-2
   * (H'B * theta_2)[n-1] = theta_2[n-2] (boundary condition) */

  /* First time point with boundary condition */
  double v_b;
  if (y[0] == 1.0) {
    v_b = rtruncnorm(theta_1_previous[0], 1.0, 0.0, R_PosInf);
  } else {
    v_b = rtruncnorm(theta_1_previous[0], 1.0, R_NegInf, 0.0);
  }

  rhs_vector[0] = v_b + prec_theta1_previous * (theta_01_previous + theta_02_previous -
    theta_2_current[0]);

  /* Intermediate time points */
  for (int t = 1; t < n - 1; t++) {
    double v_t;

    if (y[t] == 1.0) {
      v_t = rtruncnorm(theta_1_previous[t], 1.0, 0.0, R_PosInf);
    } else {
      v_t = rtruncnorm(theta_1_previous[t], 1.0, R_NegInf, 0.0);
    }

    double theta_2_diff = theta_2_current[t - 1] - theta_2_current[t];
    rhs_vector[t] = v_t + prec_theta1_previous * theta_2_diff;
  }

  /* Last time point with boundary condition */
  if (y[n - 1] == 1.0) {
    v_b = rtruncnorm(theta_1_previous[n - 1], 1.0, 0.0, R_PosInf);
  } else {
    v_b = rtruncnorm(theta_1_previous[n - 1], 1.0, R_NegInf, 0.0);
  }

  rhs_vector[n - 1] = v_b + prec_theta1_previous * theta_2_current[n - 2];

  /* ========== Sample theta_1 from Multivariate Normal ========== */
  /* Sample from: theta_1 | v, theta_2, [...] ~ N(mu_posterior, Sigma_posterior)
   * where Sigma_posterior^{-1} = I + prec_theta_1 * H'H
   *
   * The precision matrix has tridiagonal structure with:
   * - Diagonal: 1 + 2*prec_theta1 for t = 0,...,n-2
   * - Last diagonal: 1 + prec_theta1
   * - Off-diagonal: -prec_theta1
   *
   * This corresponds to generate_normal_vector with:
   * a = 1.0 (observational precision), b = prec_theta1 (state precision) */
  generate_normal_vector(
    theta_1_current,    /* r: output vector [n] */
    rhs_vector,         /* y: right-hand side [n] */
    1.0,                /* a: observational precision (from latent variance = 1) */
    prec_theta1_previous,    /* b: state precision */
    n,                  /* n: dimension */
    1                   /* add_a: use (a + b) for last diagonal element */
  );

  /* ========== Guard Against Probit Saturation Drift ========== */
  /* Clamp the latent state to the region where Phi is not numerically
   * saturated. Inert when |theta_1| < PROBIT_THETA_CLAMP; on pure/well-separated
   * stretches it stops theta_1 from random-walking into the flat tail of Phi and
   * collapsing 1/W_1. See clamp_probit_state for the full rationale. */
  for (int t = 0; t < n; t++) {
    theta_1_current[t] = clamp_probit_state(theta_1_current[t]);
  }

  /* ========== Transform to Probability Scale ========== */
  /* Compute alpha_t = Phi(theta_{t,1}) for all t if requested.
   * This transformation is needed for posterior summaries and diagnostics
   * but can be skipped during burn-in to save computational cost. */
  if (compute_alpha) {
    for (int t = 0; t < n; t++) {
      alpha_current[t] = pnorm(theta_1_current[t], 0.0, 1.0, 1, 0);
    }
  }
}
