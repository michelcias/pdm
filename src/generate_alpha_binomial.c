/**
 * @file generate_alpha_binomial.c
 * @brief Component-wise Metropolis-Hastings sampling for logit-binomial state-space
 *        models - Optimized version.
 * @author Michel H. Montoril
 * @date 2025-09-22
 * @version 1.1
 *
 * @details This file contains optimized functions for adaptive MCMC algorithms, including:
 *          - Component-wise MH updates for local level binomial models
 *          - Component-wise MH updates for local trend binomial models
 *          - Memory optimizations and computational efficiency improvements
 */

#include <R.h>
#include <Rmath.h>

#include "cwmh_adaptive.h"  /* adapt_cwmh_parameters */
#include "cwmh_binomial.h"
#include "generate_alpha_binomial.h"

/**
 * @brief Component-wise Metropolis-Hastings sampler for theta_1 in a logit-binomial
 *        local level model - Optimized version with configurable threshold.
 *
 * @details Implements an optimized component-wise Metropolis-Hastings algorithm to sample the
 *          level state vector theta_1 in a binomial observation model with logit link:
 *          y_t ~ Binomial(n_trials, alpha_t),
 *          where alpha_t = logit^{-1}(theta_{1,t}).
 *
 *          The state equation is a random walk:
 *          theta_{1,t} = theta_{1,t-1} + u_{1,t},
 *          with u_{1,t} ~ N(0, 1/prec_theta_1). This excludes trend components.
 *
 *          Iteration timing: Uses theta_01[iter-1] and prec_theta_1[iter-1] because
 *          those are sampled later in the Gibbs sequence.
 *
 *          **Optimizations implemented:**
 *          - Cached precision computations to avoid repeated sqrt/division
 *          - Reduced memory allocation by eliminating redundant arrays
 *          - Sliding window memory optimization for theta_1_updated
 *          - Stable log-probability computations
 *          - Configurable threshold for adaptation sensitivity control
 *
 *          This routine glues together:
 *          - Adaptive proposal tuning (log_sigma) via recent acceptance proportions (accept_prop)
 *          - Component-wise Metropolis-Hastings update for theta_1 (nonlinear observation
 *            with logit link)
 *
 *          The adaptation follows a diminishing adaptation schedule and is executed
 *          periodically over a sliding window of size lag_update. The actual state
 *          update is delegated to cwmh_alpha_logit_binomial_locallevel, which handles boundary
 *          conditions and log-acceptance.
 *
 *          Adaptation cadence:
 *          - Performed when iter > lag_update and (iter - 1) % lag_update == 0, i.e.,
 *          at iterations (lag_update + 1), (2*lag_update + 1), (3*lag_update + 1), ...
 *
 *          **Version 1.2 updates:**
 *          Enhanced flexibility by exposing min_deviation_threshold parameter, allowing
 *          fine-grained control over adaptation sensitivity. Recommended values include
 *          1.0/lag_update for practical applications, smaller values for sensitive adaptation,
 *          and larger values for conservative behavior.
 *
 * @param theta_1              Matrix of level states (vectorized B x n), input/output.
 * @param theta_01             Vector of initial level states (size B).
 * @param theta_1_updated      Sliding window matrix of acceptance indicators
 *                             (vectorized lag_update x n), output. Uses circular indexing.
 * @param alpha                Matrix of transformed probabilities (vectorized B x n),
 *                             output. Each alpha[t] = logit^{-1}(theta_1[t]).
 * @param prec_theta_1         Vector of level precision parameters (size B).
 * @param y                    Vector of observed binomial counts (size n).
 *                             Each y[k] must satisfy 0 <= y[k] <= n_trials.
 * @param acceptance_probs     Vector of acceptance proportions for each component (size n).
 *                             Used to monitor MCMC performance and guide adaptive tuning.
 * @param log_sigma            Vector of log proposal standard deviations (size n).
 * @param hat_theta_1          Temporary vector for conditional means (size n).
 * @param theta_1_new          Temporary vector for proposed values (size n).
 * @param log_accept_prob      Temporary vector for log acceptance probabilities (size n).
 * @param lag_update           Integer scalar, sliding window size for adaptation frequency.
 *                             Adaptation occurs every lag_update iterations when
 *                             iter >= lag_update. Set to 0 to disable adaptation.
 * @param n_trials             Number of Bernoulli trials (double).
 * @param n                    Length of the time series.
 * @param iter                 Current MCMC iteration (0-based).
 * @param max_step_size        Double scalar, maximum adaptation step size for log_sigma
 *                             updates. Prevents excessive proposal variance changes during
 *                             adaptation.
 * @param base_adaptation_rate Double scalar, initial adaptation rate before decay.
 *                             Controls the magnitude of log_sigma adjustments.
 * @param decay_exponent       Double scalar, exponent for diminishing adaptation schedule.
 *                             Step size = min(max_step_size, base_adaptation_rate / iter^decay_exponent).
 *                             Typical values: 0.3-0.8 for robust convergence.
 * @param target_acceptance    Double scalar, target acceptance rate for adaptive tuning.
 *                             Typical values: 0.44 (univariate) or 0.234 (multivariate).
 *                             Adaptation adjusts log_sigma to achieve this rate.
 * @param min_deviation_threshold Double scalar, minimum absolute deviation from target_acceptance
 *                             required to trigger log_sigma updates. Must be >= 0. Recommended
 *                             values: 1.0/lag_update for practical applications, smaller values
 *                             for sensitive adaptation, 0.0 to disable threshold filtering.
 *
 * @note Complexity: O(n) per iteration (component-wise updates).
 * @note Uses log-probabilities for numerical stability.
 * @note Forward sampling for better mixing.
 * @note Model is local level (no trend).
 * @note Adaptive tuning performed every lag_update iterations if iter >= lag_update.
 * @note Memory optimization: theta_1_updated uses sliding window instead of full matrix.
 * @note Threshold flexibility: Caller can specify any non-negative threshold value.
 *
 * @warning Each y[k] must satisfy 0 ≤ y[k] ≤ n_trials.
 * @warning Results are invalid if theta_01 or prec_theta_1 do not contain sufficient
 *          history (iter < 1).
 * @warning lag_update must be > 0 for theta_1_updated indexing.
 * @warning min_deviation_threshold must be >= 0.0.
 *
 * @see adapt_cwmh_parameters
 * @see cwmh_alpha_logit_binomial_locallevel
 */
void generate_alpha_logit_binomial_locallevel(double *theta_1,
                                              double *theta_01,
                                              double *theta_1_updated,
                                              double *alpha,
                                              double *prec_theta_1,
                                              double *y,
                                              double *accept_prop,
                                              double *log_sigma,
                                              double *hat_theta_1,
                                              double *theta_1_new,
                                              double *log_accept_prob,
                                              int     lag_update,
                                              double  n_trials,
                                              int     n,
                                              int     iter,
                                              double  max_step_size,
                                              double  base_adaptation_rate,
                                              double  decay_exponent,
                                              double  target_acceptance,
                                              double  min_deviation_threshold) {

  /* ========== Prerequisites and Safety Checks ========== */
  // cwmh_alpha_logit_binomial_locallevel uses prev_iter = iter - 1 for theta_01 and prec_theta_1
  if (iter <= 0) {
    // Nothing to do in iteration 0; typically used to initialize storage.
    return;
  }

  /* ========== Adaptive Tuning (periodic, sliding window) ========== */
  // Trigger adaptation every 'lag_update' iterations once sufficient history exists
  if (lag_update > 0 && iter >= lag_update && (iter % lag_update == 0)) {

    adapt_cwmh_parameters(
      theta_1_updated,        /* theta_updated */
      accept_prop,            /* accept_prop */
      log_sigma,              /* log_sigma */
      lag_update,             /* lag_update */
      n,                      /* n */
      iter,                   /* iter */
      max_step_size,          /* max_step_size */
      base_adaptation_rate,   /* base_adaptation_rate */
      decay_exponent,         /* decay_exponent */
      target_acceptance,      /* target_acceptance */
      min_deviation_threshold /* min_deviation_threshold */
    );
  }

  /* ========== CWMH Update for Current Iteration ========== */
  // Updates theta_1 block for 'iter', logs acceptance, and stores alpha = ilogit(theta_1)
  cwmh_alpha_logit_binomial_locallevel(
    theta_1,           /* theta_1 */
    theta_01,          /* theta_01 */
    theta_1_updated,   /* theta_1_updated */
    alpha,             /* alpha */
    prec_theta_1,      /* prec_theta_1 */
    y,                 /* y */
    log_sigma,         /* log_sigma */
    hat_theta_1,       /* hat_theta_1 */
    theta_1_new,       /* theta_1_new */
    log_accept_prob,   /* log_accept_prob */
    lag_update,        /* lag_update */
    n_trials,          /* n_trials */
    n,                 /* n */
    iter               /* iter */
  );
}

//----------------------------------------------------------------------

/**
 * @brief Component-wise Metropolis-Hastings sampler for theta_1 in a logit-binomial
 *        dynamic model (with local trend) - Optimized version.
 *
 * @details Implements an optimized component-wise Metropolis-Hastings algorithm to sample the
 *          level state vector theta_1 with a binomial observation model and local
 *          trend state-space evolution:
 *          y_t ~ Binomial(n_trials, alpha_t),
 *          where alpha_t = logit^{-1}(theta_{1,t}).
 *
 *          State equation:
 *          theta_{1,t} = theta_{1,t-1} + theta_{2,t-1} + u_{1,t},
 *          with u_{1,t} ~ N(0, 1/prec_theta_1).
 *
 *          Iteration timing: Uses theta_01[iter-1] and prec_theta_1[iter-1] because
 *          those are sampled later in the Gibbs sequence.
 *
 *          **Optimizations implemented:**
 *          - Cached precision computations to avoid repeated sqrt/division
 *          - Reduced memory allocation by eliminating redundant arrays
 *          - Sliding window memory optimization for theta_1_updated
 *          - Stable log-probability computations
 *
 *          This routine glues together:
 *          - Adaptive proposal tuning (log_sigma) via recent acceptance proportions (accept_prop)
 *          - Component-wise Metropolis-Hastings update for theta_1 (nonlinear observation
 *            with logit link)
 *
 *          The adaptation follows a diminishing adaptation schedule and is executed
 *          periodically over a sliding window of size lag_update. The actual state
 *          update is delegated to cwmh_alpha_logit_binomial, which handles boundary
 *          conditions and log-acceptance.
 *
 *          Adaptation cadence:
 *          - Performed when iter > lag_update and (iter - 1) % lag_update == 0, i.e.,
 *          at iterations (lag_update + 1), (2*lag_update + 1), (3*lag_update + 1), ...
 *
 * @param theta_1              Matrix of level states (vectorized B x n), input/output.
 * @param theta_2              Matrix of trend states (vectorized B x n), input only.
 * @param theta_01             Vector of initial level states (size B).
 * @param theta_02             Vector of initial trend states (size B).
 * @param theta_1_updated      Sliding window matrix of acceptance indicators
 *                             (vectorized lag_update x n), output. Uses circular indexing.
 * @param alpha                Matrix of transformed probabilities (vectorized B x n), output.
 * @param prec_theta_1         Vector of level precision parameters (size B).
 * @param y                    Vector of observed binomial counts (size n).
 * @param acceptance_probs     Vector of acceptance proportions for each component (size n).
 *                             Used to monitor MCMC performance and guide adaptive tuning.
 * @param log_sigma            Vector of log proposal standard deviations (size n).
 * @param hat_theta_1          Temporary vector for conditional means (size n).
 * @param theta_1_new          Temporary vector for proposed values (size n).
 * @param log_accept_prob      Temporary vector for log acceptance probabilities (size n).
 * @param lag_update           Integer scalar, sliding window size for adaptation frequency.
 *                             Adaptation occurs every lag_update iterations when
 *                             iter >= lag_update. Set to 0 to disable adaptation.
 * @param n_trials             Number of Bernoulli trials (double).
 * @param n                    Length of the time series.
 * @param iter                 Current MCMC iteration (0-based).
 * @param max_step_size        Double scalar, maximum adaptation step size for log_sigma
 *                             updates. Prevents excessive proposal variance changes during
 *                             adaptation.
 * @param base_adaptation_rate Double scalar, initial adaptation rate before decay.
 *                             Controls the magnitude of log_sigma adjustments.
 * @param decay_exponent       Double scalar, exponent for diminishing adaptation schedule.
 *                             Step size = min(max_step_size, base_adaptation_rate / iter^decay_exponent).
 *                             Typical values: 0.3-0.8 for robust convergence.
 * @param target_acceptance    Double scalar, target acceptance rate for adaptive tuning.
 *                             Typical values: 0.44 (univariate) or 0.234 (multivariate).
 *                             Adaptation adjusts log_sigma to achieve this rate.
 *
 * @note Complexity: O(n) per iteration (component-wise updates).
 * @note Uses log-probabilities for numerical stability.
 * @note Forward sampling for better mixing.
 * @note Model is local trend (random walk + trend).
 * @note Adaptive tuning performed every lag_update iterations if iter >= lag_update.
 * @note Memory optimization: theta_1_updated uses sliding window instead of full matrix.
 *
 * @warning Each y[k] must satisfy 0 ≤ y[k] ≤ n_trials.
 * @warning Results are invalid if theta_01 or prec_theta_1 do not contain sufficient
 *          history (iter < 1).
 * @warning lag_update must be > 0 for theta_1_updated indexing.
 *
 * @see adapt_cwmh_parameters
 * @see cwmh_alpha_logit_binomial
 */
void generate_alpha_logit_binomial(double *theta_1,
                                   double *theta_2,
                                   double *theta_01,
                                   double *theta_02,
                                   double *theta_1_updated,
                                   double *alpha,
                                   double *prec_theta_1,
                                   double *y,
                                   double *accept_prop,
                                   double *log_sigma,
                                   double *hat_theta_1,
                                   double *theta_1_new,
                                   double *log_accept_prob,
                                   int     lag_update,
                                   double  n_trials,
                                   int     n,
                                   int     iter,
                                   double  max_step_size,
                                   double  base_adaptation_rate,
                                   double  decay_exponent,
                                   double  target_acceptance) {

  /* ========== Prerequisites and Safety Checks ========== */
  // cwmh_alpha_logit_binomial uses prev_iter = iter - 1 for theta_01 and prec_theta_1
  if (iter <= 0) {
    // Nothing to do in iteration 0; typically used to initialize storage.
    return;
  }

  /* ========== Adaptive Tuning (periodic, sliding window) ========== */
  // Trigger adaptation every 'lag_update' iterations once sufficient history exists
  if (lag_update > 0 && iter >= lag_update && (iter % lag_update == 0)) {
    adapt_cwmh_parameters(
      theta_1_updated,       /* theta_updated */
      accept_prop,           /* accept_prop */
      log_sigma,             /* log_sigma */
      lag_update,            /* lag_update */
      n,                     /* n */
      iter,                  /* iter */
      max_step_size,         /* max_step_size */
      base_adaptation_rate,  /* base_adaptation_rate */
      decay_exponent,        /* decay_exponent */
      target_acceptance      /* target_acceptance */
    );
  }

  /* ========== CWMH Update for Current Iteration ========== */
  // Updates theta_1 block for 'iter', logs acceptance, and stores alpha = ilogit(theta_1)
  cwmh_alpha_logit_binomial(
    theta_1,           /* theta_1 */
    theta_2,           /* theta_2 */
    theta_01,          /* theta_01 */
    theta_02,          /* theta_02 */
    theta_1_updated,   /* theta_1_updated */
    alpha,             /* alpha */
    prec_theta_1,      /* prec_theta_1 */
    y,                 /* y */
    log_sigma,         /* log_sigma */
    hat_theta_1,       /* hat_theta_1 */
    theta_1_new,       /* theta_1_new */
    log_accept_prob,   /* log_accept_prob */
    lag_update,        /* lag_update */
    n_trials,          /* n_trials */
    n,                 /* n */
    iter               /* iter */
  );
}
