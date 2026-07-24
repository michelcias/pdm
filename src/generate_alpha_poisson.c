/**
 * @file generate_alpha_poisson.c
 * @brief Sampling for Poisson state-space models with log link
 * @author Michel H.  Montoril
 * @date 2025-12-20
 * @version 1.0
 *
 * @details This file contains optimized functions for MCMC sampling in Poisson
 *          state-space models with log link function:
 *          - Log-Poisson models: Component-wise Metropolis-Hastings with adaptive tuning
 *          - Memory-efficient implementations using current/previous iteration buffers
 *          - Integration with configurable adaptive threshold parameters
 *          - Conditional alpha computation for performance optimization
 *
 *          **Log-Poisson models:**
 *          Use component-wise Metropolis-Hastings updates for the level state vector
 *          with adaptive proposal tuning based on acceptance rates.
 */

#include <R.h>
#include <Rmath.h>
#include "cwmh_adaptive.h"  /* adapt_cwmh_parameters */
#include "cwmh_poisson.h"   /* cwmh_alpha_log_poisson_* */
#include "generate_alpha_poisson.h"

/**
 * @brief Component-wise Metropolis-Hastings sampler for theta_1 in a log-Poisson
 *        local level model with adaptive tuning
 *
 * @details Implements an optimized component-wise Metropolis-Hastings algorithm to sample the
 *          level state vector theta_1 in a Poisson observation model with log link:
 *
 *          **Observation equation:**
 *          y_t ~ Poisson(alpha_t), where alpha_t = exp(theta_{t,1})
 *
 *          **State equation (random walk):**
 *          theta_{t,1} = theta_{t-1,1} + u_{t,1}, u_{t,1} ~ N(0, W_1)
 *
 *          This routine integrates:
 *          - Adaptive proposal tuning (log_sigma) via recent acceptance proportions (accept_prop)
 *          - Component-wise Metropolis-Hastings update for theta_1 (nonlinear observation
 *            with log link)
 *          - Conditional alpha computation for performance optimization
 *
 *          The adaptation follows a diminishing adaptation schedule and is executed
 *          periodically over a sliding window of size lag_update.  The actual state
 *          update is delegated to cwmh_alpha_log_poisson_locallevel, which handles boundary
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
 * @param alpha_current           Output rate vector [n] for current iteration.
 *                                Can be NULL if compute_alpha = 0.
 * @param theta_01_previous       Scalar initial level state from previous iteration.
 * @param prec_theta1_previous    Scalar level precision from previous iteration.
 * @param theta_1_updated         Sliding window matrix [lag_update * n] of acceptance indicators.
 * @param y                       Observed Poisson counts vector [n] (const, read-only).
 *                                Each y[t] must be a non-negative integer.
 * @param accept_prop             Workspace vector [n] for acceptance proportions.
 * @param log_sigma               Input/output vector [n] of log proposal standard deviations.
 * @param hat_theta_1             Workspace vector [n] for conditional means.
 * @param theta_1_new             Workspace vector [n] for proposed values.
 * @param log_accept_prob         Workspace vector [n] for log acceptance probabilities.
 * @param lag_update              Sliding window size for adaptation frequency (> 0).
 *                                Set to 0 to disable adaptation.
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
 * @note Performance:  Skipping alpha computation provides 10-30% speedup during burn-in/thinning.
 *
 * @warning Each y[t] must be non-negative.
 * @warning iter must be >= 1 for valid theta_1_previous access.
 * @warning lag_update must be > 0 for theta_1_updated indexing.
 * @warning min_deviation_threshold must be >= 0.0.
 * @warning If compute_alpha = 1, alpha_current must be a valid pointer.
 * @warning If compute_alpha = 0, alpha_current can be NULL.
 *
 * @see adapt_cwmh_parameters
 * @see cwmh_alpha_log_poisson_locallevel
 */
void generate_alpha_log_poisson_locallevel(const double *theta_1_previous,
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
                                           int           n,
                                           int           iter,
                                           double        max_step_size,
                                           double        base_adaptation_rate,
                                           double        decay_exponent,
                                           double        target_acceptance,
                                           double        min_deviation_threshold,
                                           int           compute_alpha) {

  /* ========== Prerequisites and Safety Checks ========== */
  /* cwmh_alpha_log_poisson_locallevel requires iter >= 1 for previous iteration access */
  if (iter <= 0) {
    /* Nothing to do in iteration 0; typically used to initialize storage.  */
    return;
  }

  /* ========== Adaptive Tuning (periodic, sliding window) ========== */
  /* Trigger adaptation every 'lag_update' iterations once sufficient history exists */
  if (lag_update > 0 && iter >= lag_update && (iter % lag_update == 0)) {

    adapt_cwmh_parameters(
      theta_1_updated,        /* theta_updated:  sliding window acceptance buffer */
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
   * Uses memory-efficient current/previous buffers instead of full trajectory storage.  */
  cwmh_alpha_log_poisson_locallevel(
    theta_1_previous,     /* theta_1_previous: state from previous iteration [n] */
    theta_1_current,      /* theta_1_current: output for current iteration [n] */
    alpha_current,        /* alpha_current: rates (NULL if compute_alpha=0) */
    theta_01_previous,    /* theta_01_previous: initial level from previous iteration */
    prec_theta1_previous, /* prec_theta1_previous: level precision from previous iteration */
    theta_1_updated,      /* theta_1_updated: sliding window indicators */
    y,                    /* y: observed counts */
    log_sigma,            /* log_sigma: proposal log standard deviations */
    hat_theta_1,          /* hat_theta_1: conditional means workspace */
    theta_1_new,          /* theta_1_new: proposal buffer */
    log_accept_prob,      /* log_accept_prob: log acceptance storage */
    lag_update,           /* lag_update: adaptation window length */
    n,                    /* n: number of observations */
    iter,                 /* iter: current iteration */
    compute_alpha         /* compute_alpha:  flag for alpha computation */
  );
}

//----------------------------------------------------------------------

/**
 * @brief Component-wise Metropolis-Hastings sampler for theta_1 in a log-Poisson
 *        local trend model with adaptive tuning
 *
 * @details Implements an optimized component-wise Metropolis-Hastings algorithm to sample the
 *          level state vector theta_1 with a Poisson observation model and local
 *          trend state-space evolution:
 *
 *          **Observation equation:**
 *          y_t ~ Poisson(alpha_t), where alpha_t = exp(theta_{t,1})
 *
 *          **State equations:**
 *          theta_{t,1} = theta_{t-1,1} + theta_{t-1,2} + u_{t,1}, u_{t,1} ~ N(0, W_1)
 *          theta_{t,2} = theta_{t-1,2} + u_{t,2},                 u_{t,2} ~ N(0, W_2)
 *
 *          This routine integrates:
 *          - Adaptive proposal tuning (log_sigma) via recent acceptance proportions (accept_prop)
 *          - Component-wise Metropolis-Hastings update for theta_1 (nonlinear observation
 *            with log link)
 *          - Conditional alpha computation for performance optimization
 *
 *          The adaptation follows a diminishing adaptation schedule and is executed
 *          periodically over a sliding window of size lag_update. The actual state
 *          update is delegated to cwmh_alpha_log_poisson, which handles boundary
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
 * @param alpha_current           Output rate vector [n] for current iteration.
 *                                Can be NULL if compute_alpha = 0.
 * @param theta_2_current         Trend state vector [n] from current iteration (const).
 *                                Must be sampled before calling this function.
 * @param theta_01_previous       Scalar initial level state from previous iteration.
 * @param theta_02_previous       Scalar initial trend state from previous iteration.
 * @param prec_theta1_previous    Scalar level precision from previous iteration.
 * @param theta_1_updated         Sliding window matrix [lag_update * n] of acceptance indicators.
 * @param y                       Observed Poisson counts vector [n] (const, read-only).
 * @param accept_prop             Workspace vector [n] for acceptance proportions.
 * @param log_sigma               Input/output vector [n] of log proposal standard deviations.
 * @param hat_theta_1             Workspace vector [n] for conditional means.
 * @param theta_1_new             Workspace vector [n] for proposed values.
 * @param log_accept_prob         Workspace vector [n] for log acceptance probabilities.
 * @param lag_update              Sliding window size for adaptation frequency (> 0).
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
 * @warning Each y[t] must be non-negative.
 * @warning iter must be >= 1 for valid previous iteration access.
 * @warning lag_update must be > 0 for theta_1_updated indexing.
 * @warning min_deviation_threshold must be >= 0.0.
 * @warning theta_2_current must contain valid values from current iteration.
 * @warning If compute_alpha = 1, alpha_current must be a valid pointer.
 * @warning If compute_alpha = 0, alpha_current can be NULL.
 *
 * @see adapt_cwmh_parameters
 * @see cwmh_alpha_log_poisson
 */
void generate_alpha_log_poisson(const double *theta_1_previous,
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
                                int           n,
                                int           iter,
                                double        max_step_size,
                                double        base_adaptation_rate,
                                double        decay_exponent,
                                double        target_acceptance,
                                double        min_deviation_threshold,
                                int           compute_alpha) {

  /* ========== Prerequisites and Safety Checks ========== */
  /* cwmh_alpha_log_poisson requires iter >= 1 for previous iteration access */
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
      base_adaptation_rate,   /* base_adaptation_rate:  initial adaptation rate */
      decay_exponent,         /* decay_exponent: diminishing schedule */
      target_acceptance,      /* target_acceptance: desired acceptance probability */
      min_deviation_threshold /* min_deviation_threshold: deviation trigger */
    );
  }

  /* ========== CWMH Update for Current Iteration ========== */
  /* Updates theta_1 for current iteration, logs acceptance, and optionally stores alpha.
   * Uses memory-efficient current/previous buffers instead of full trajectory storage. */
  cwmh_alpha_log_poisson(
    theta_1_previous,     /* theta_1_previous:  level from previous iteration [n] */
    theta_1_current,      /* theta_1_current: output for current iteration [n] */
    alpha_current,        /* alpha_current: rates (NULL if compute_alpha=0) */
    theta_2_current,      /* theta_2_current: trend from current iteration [n] */
    theta_01_previous,    /* theta_01_previous:  initial level from previous iteration */
    theta_02_previous,    /* theta_02_previous: initial trend from previous iteration */
    prec_theta1_previous, /* prec_theta1_previous: level precision from previous iteration */
    theta_1_updated,      /* theta_1_updated: sliding window indicators */
    y,                    /* y: observed counts */
    log_sigma,            /* log_sigma:  proposal log standard deviations */
    hat_theta_1,          /* hat_theta_1: conditional means workspace */
    theta_1_new,          /* theta_1_new: proposal buffer */
    log_accept_prob,      /* log_accept_prob: log acceptance storage */
    lag_update,           /* lag_update: adaptation window length */
    n,                    /* n: number of observations */
    iter,                 /* iter:  current iteration */
    compute_alpha         /* compute_alpha: flag for alpha computation */
  );
}