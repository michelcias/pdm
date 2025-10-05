/**
 * @file generate_alpha_binomial.c
 * @brief Component-wise Metropolis-Hastings sampling for logit-binomial state-space
 *        models - Optimized version.
 * @author Michel H. Montoril
 * @date 2025-09-23
 * @version 1.2
 *
 * @details This file contains optimized functions for adaptive MCMC algorithms, including:
 *          - Component-wise MH updates for local level binomial models
 *          - Component-wise MH updates for local trend binomial models
 *          - Memory optimizations and computational efficiency improvements
 *          - Integration with configurable adaptive threshold parameters
 *
 * @changelog
 * - v1.2 (2025-09-23): Updated function signatures to include min_deviation_threshold
 *   parameter, providing flexible control over adaptation sensitivity while maintaining
 *   optimal performance characteristics.
 */

#include <R.h>
#include <Rmath.h>

#include "cwmh_adaptive.h"  /* adapt_cwmh_parameters */
#include "cwmh_binomial.h"
#include "utils.h"          /* generate_normal_vector */
#include "generate_alpha_binomial.h"

/**
 * @brief Component-wise Metropolis-Hastings sampler for theta_1 in a logit-binomial
 *        local level model - Optimized version with configurable threshold.
 *
 * @details Implements an optimized component-wise Metropolis-Hastings algorithm to sample the
 *          level state vector theta_1 in a binomial observation model with logit link:
 *          y_t ~ Binomial(n_trials, alpha_t),
 *          where alpha_t = logit^{-1}(theta_{t,1}).
 *
 *          The state equation is a random walk:
 *          theta_{t,1} = theta_{t-1,1} + u_{t,1},
 *          with u_{t,1} ~ N(0, 1/prec_theta_1). This excludes trend components.
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
  // Updates theta_1 block for 'iter', logs acceptance, and stores alpha = ilogit(theta_1)
  cwmh_alpha_logit_binomial_locallevel(
    theta_1,         /* theta_1: state trajectory matrix */
    theta_01,        /* theta_01: initial level states */
    theta_1_updated, /* theta_1_updated: sliding window indicators */
    alpha,           /* alpha: success probability samples */
    prec_theta_1,    /* prec_theta_1: level precision draws */
    y,               /* y: observed counts */
    log_sigma,       /* log_sigma: proposal log standard deviations */
    hat_theta_1,     /* hat_theta_1: conditional means workspace */
    theta_1_new,     /* theta_1_new: proposal buffer */
    log_accept_prob, /* log_accept_prob: log acceptance storage */
    lag_update,      /* lag_update: adaptation window length */
    n_trials,        /* n_trials: binomial trials */
    n,               /* n: number of observations */
    iter             /* iter: current iteration */
  );
}

//----------------------------------------------------------------------

/**
 * @brief Component-wise Metropolis-Hastings sampler for theta_1 in a logit-binomial
 *        dynamic model (with local trend) - Optimized version with configurable threshold.
 *
 * @details Implements an optimized component-wise Metropolis-Hastings algorithm to sample the
 *          level state vector theta_1 with a binomial observation model and local
 *          trend state-space evolution:
 *          y_t ~ Binomial(n_trials, alpha_t),
 *          where alpha_t = logit^{-1}(theta_{t,1}).
 *
 *          State equation:
 *          theta_{t,1} = theta_{t-1,1} + theta_{t-1,2} + u_{t,1},
 *          with u_{t,1} ~ N(0, 1/prec_theta_1).
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
 *          update is delegated to cwmh_alpha_logit_binomial, which handles boundary
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
 * @param min_deviation_threshold Double scalar, minimum absolute deviation from target_acceptance
 *                             required to trigger log_sigma updates. Must be >= 0. Recommended
 *                             values: 1.0/lag_update for practical applications, smaller values
 *                             for sensitive adaptation, 0.0 to disable threshold filtering.
 *
 * @note Complexity: O(n) per iteration (component-wise updates).
 * @note Uses log-probabilities for numerical stability.
 * @note Forward sampling for better mixing.
 * @note Model is local trend (random walk + trend).
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
                                   double  target_acceptance,
                                   double  min_deviation_threshold) {

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
  // Updates theta_1 block for 'iter', logs acceptance, and stores alpha = ilogit(theta_1)
  cwmh_alpha_logit_binomial(
    theta_1,         /* theta_1: level state trajectories */
    theta_2,         /* theta_2: trend state trajectories */
    theta_01,        /* theta_01: initial level states */
    theta_02,        /* theta_02: initial trend states */
    theta_1_updated, /* theta_1_updated: sliding window indicators */
    alpha,           /* alpha: success probability samples */
    prec_theta_1,    /* prec_theta_1: level precision draws */
    y,               /* y: observed counts */
    log_sigma,       /* log_sigma: proposal log standard deviations */
    hat_theta_1,     /* hat_theta_1: conditional means workspace */
    theta_1_new,     /* theta_1_new: proposal buffer */
    log_accept_prob, /* log_accept_prob: log acceptance storage */
    lag_update,      /* lag_update: adaptation window length */
    n_trials,        /* n_trials: binomial trials */
    n,               /* n: number of observations */
    iter             /* iter: current iteration */
  );
}

/**
 * @brief Efficient sampling from truncated normal distribution N(mu, sigma^2).
 *
 * @details Uses the inverse CDF method with proper bounds handling for numerical stability.
 *          Handles extreme cases where truncation bounds are far from the mean.
 *
 * @param mu     Mean of the normal distribution.
 * @param sigma  Standard deviation of the normal distribution.
 * @param lower  Lower truncation bound (-Inf for no lower bound).
 * @param upper  Upper truncation bound (+Inf for no upper bound).
 * @return       Random sample from the truncated distribution.
 *
 * @note Uses R's pnorm/qnorm for numerical stability.
 * @note Handles boundary cases gracefully.
 */
static double rtruncnorm(double mu, double sigma, double lower, double upper) {
  double p_lower, p_upper, u, p;

  // Compute cumulative probabilities at bounds
  if (lower == R_NegInf) {
    p_lower = 0.0;
  } else {
    p_lower = pnorm(lower, mu, sigma, 1, 0);
  }

  if (upper == R_PosInf) {
    p_upper = 1.0;
  } else {
    p_upper = pnorm(upper, mu, sigma, 1, 0);
  }

  // Sample uniform in valid probability range
  u = unif_rand();
  p = p_lower + u * (p_upper - p_lower);

  // Transform back to truncated normal
  return qnorm(p, mu, sigma, 1, 0);
}

/**
 * @brief Gibbs sampler for theta_1 in a probit-Bernoulli local level model
 *        using Albert-Chib data augmentation.
 *
 * @details Implements efficient Gibbs sampling for the level state vector theta_1
 *          in a Bernoulli observation model with probit link:
 *          y_t ~ Bernoulli(alpha_t),
 *          where alpha_t = Phi(theta_{t,1}) and Phi is the standard normal CDF.
 *
 *          State equation (random walk):
 *          theta_{t,1} = theta_{t-1,1} + u_{t,1},
 *          with u_{t,1} ~ N(0, 1/prec_theta_1).
 *
 *          **Albert-Chib Data Augmentation:**
 *          Introduces latent variables v_t ~ N(theta_{t,1}, 1) such that:
 *          - y_t = 1 if v_t > 0
 *          - y_t = 0 if v_t <= 0
 *
 *          **Algorithm:**
 *          1. Sample latent variables v_t from truncated normals given theta_1, y
 *          2. Sample theta_1 from multivariate normal given v using generate_normal_vector
 *
 *          The full conditional posterior is:
 *          theta_1 | v, [...] ~ N(mu_posterior, Sigma_posterior)
 *          where Sigma_posterior^{-1} = I + prec_theta_1 * H'H (tridiagonal precision)
 *                mu_posterior = Sigma_posterior * [v + prec_theta_1 * theta_01 * e_1]
 *
 * @param theta_1              Matrix of level states (vectorized B x n), input/output.
 * @param theta_01             Vector of initial level states (size B).
 * @param alpha                Matrix of transformed probabilities (vectorized B x n), output.
 * @param prec_theta_1         Vector of level precision parameters (size B).
 * @param y                    Vector of observed Bernoulli outcomes (size n).
 *                             Each y[k] must be exactly 0 or 1.
 * @param v_latent             Working vector for latent variables (size n).
 * @param rhs_vector           Working vector for right-hand side of system (size n).
 * @param n                    Length of the time series.
 * @param iter                 Current MCMC iteration (0-based).
 *
 * @note Complexity: O(n) per iteration (exploiting tridiagonal structure).
 * @note Always achieves acceptance rate of 1.0 (Gibbs sampling).
 * @note Model is local level (no trend component).
 *
 * @warning Each y[k] must be exactly 0 or 1 (Bernoulli outcomes).
 * @warning Results are invalid if theta_01 or prec_theta_1 do not contain
 *          sufficient history (iter < 1).
 * @warning n must be > 0 for generate_normal_vector to work correctly.
 *
 * @see Albert and Chib (1993) "Bayesian Analysis of Binary and Polychotomous Response Data"
 * @see generate_normal_vector
 */
void generate_alpha_probit_bernoulli_locallevel(double *theta_1,
                                                double *theta_01,
                                                double *alpha,
                                                double *prec_theta_1,
                                                double *y,
                                                double *v_latent,
                                                double *rhs_vector,
                                                int     n,
                                                int     iter) {

  /* ========== Prerequisites and Safety Checks ========== */
  if (iter <= 0) {
    return;
  }

  int prev_iter = iter - 1;
  int current_pos = iter * n;
  int prev_pos = prev_iter * n;

  double prec_1 = prec_theta_1[prev_iter];
  double theta_0 = theta_01[prev_iter];

  // Pointers to previous iteration values
  double *theta_1_prev = &theta_1[prev_pos];

  /* ========== Step 1: Sample Latent Variables (Data Augmentation) ========== */
  // For each time point t, sample v_t from truncated normal
  // v_t | y_t, theta_{t,1} ~ N(theta_{t,1}, 1) with appropriate truncation

  for (int t = 0; t < n; t++) {
    double theta_1_mean = theta_1_prev[t];

    if (y[t] == 1.0) {
      // y_t = 1: sample from N(theta_{t,1}, 1) truncated above 0
      v_latent[t] = rtruncnorm(theta_1_mean, 1.0, 0.0, R_PosInf);
    } else {
      // y_t = 0: sample from N(theta_{t,1}, 1) truncated below 0
      v_latent[t] = rtruncnorm(theta_1_mean, 1.0, R_NegInf, 0.0);
    }
  }

  /* ========== Step 2: Construct Right-Hand Side Vector ========== */
  // Following the conditional posterior mean structure:
  // For local level: rhs = v + prec_theta_1 * theta_01 * e_1
  // where e_1 = (1, 0, 0, ..., 0)' is the first unit vector

  rhs_vector[0] = v_latent[0] + prec_1 * theta_0;
  for (int t = 1; t < n; t++) {
    rhs_vector[t] = v_latent[t];
  }

  /* ========== Step 3: Sample theta_1 from Multivariate Normal ========== */
  // Sample from: theta_1 | v, [...] ~ N(mu_posterior, Sigma_posterior)
  // where Sigma_posterior^{-1} = I + prec_theta_1 * H'H
  //
  // The precision matrix has tridiagonal structure with:
  // - Diagonal: 1 + 2*prec_1 for t = 0,...,n-2
  // - Last diagonal: 1 + prec_1
  // - Off-diagonal: -prec_1
  //
  // This corresponds to generate_normal_vector with:
  // a = 1.0 (observational precision), b = prec_1 (state precision)

  generate_normal_vector(
    theta_1,       /* r: output matrix */
    rhs_vector,    /* y: right-hand side */
    1.0,           /* a: observational precision (from latent variance = 1) */
    prec_1,        /* b: state precision */
    n,             /* n: dimension */
    iter,          /* iter: current iteration */
    1              /* add_a: use (a + b) for last diagonal element */
  );

  /* ========== Step 4: Transform to Probability Scale ========== */
  // Compute alpha_t = Phi(theta_{t,1}) for all t
  double *alpha_curr = &alpha[current_pos];
  double *theta_1_curr = &theta_1[current_pos];

  for (int t = 0; t < n; t++) {
    alpha_curr[t] = pnorm(theta_1_curr[t], 0.0, 1.0, 1, 0);
  }
}

//----------------------------------------------------------------------

/**
 * @brief Gibbs sampler for theta_1 in a probit-Bernoulli local trend model
 *        using Albert-Chib data augmentation.
 *
 * @details Similar to generate_alpha_probit_bernoulli_locallevel but includes
 *          trend component in the state evolution:
 *
 *          State equations:
 *          theta_{t,1} = theta_{t-1,1} + theta_{t-1,2} + u_{t,1},
 *          theta_{t,2} = theta_{t-1,2} + u_{t,2},
 *          with u_{t,1} ~ N(0, 1/prec_theta_1), u_{t,2} ~ N(0, 1/prec_theta_2).
 *
 *          **Algorithm:**
 *          1. Sample latent variables v_t from truncated normals
 *          2. Adjust right-hand side to account for trend component
 *          3. Sample theta_1 using generate_normal_vector
 *
 *          The full conditional incorporates the trend through additional terms
 *          in the posterior mean vector.
 *
 * @param theta_1              Matrix of level states (vectorized B x n), input/output.
 * @param theta_2              Matrix of trend states (vectorized B x n), input only.
 * @param theta_01             Vector of initial level states (size B).
 * @param theta_02             Vector of initial trend states (size B).
 * @param alpha                Matrix of transformed probabilities (vectorized B x n), output.
 * @param prec_theta_1         Vector of level precision parameters (size B).
 * @param y                    Vector of observed Bernoulli outcomes (size n).
 * @param v_latent             Working vector for latent variables (size n).
 * @param rhs_vector           Working vector for right-hand side of system (size n).
 * @param n                    Length of the time series.
 * @param iter                 Current MCMC iteration (0-based).
 *
 * @note All other properties same as generate_alpha_probit_bernoulli_locallevel.
 * @note The trend theta_2 must be already sampled in the Gibbs cycle.
 * @note Complexity: O(n) per iteration.
 *
 * @warning Theta_2 matrix must contain valid values for current iteration.
 *
 * @see Albert and Chib (1993) "Bayesian Analysis of Binary and Polychotomous Response Data"
 * @see generate_normal_vector
 */
void generate_alpha_probit_bernoulli(double *theta_1,
                                     double *theta_2,
                                     double *theta_01,
                                     double *theta_02,
                                     double *alpha,
                                     double *prec_theta_1,
                                     double *y,
                                     double *v_latent,
                                     double *rhs_vector,
                                     int     n,
                                     int     iter) {

  /* ========== Prerequisites ========== */
  if (iter <= 0) {
    return;
  }

  int prev_iter = iter - 1;
  int current_pos = iter * n;
  int prev_pos = prev_iter * n;

  double prec_1 = prec_theta_1[prev_iter];
  double theta_0_level = theta_01[prev_iter];
  double theta_0_trend = theta_02[prev_iter];

  // Pointers for efficiency
  double *theta_1_prev = &theta_1[prev_pos];
  double *theta_2_curr = &theta_2[current_pos];

  /* ========== Step 1: Sample Latent Variables ========== */

  for (int t = 0; t < n; t++) {
    double theta_1_mean = theta_1_prev[t];

    if (y[t] == 1.0) {
      v_latent[t] = rtruncnorm(theta_1_mean, 1.0, 0.0, R_PosInf);
    } else {
      v_latent[t] = rtruncnorm(theta_1_mean, 1.0, R_NegInf, 0.0);
    }
  }

  /* ========== Step 2: Construct Right-Hand Side with Trend ========== */
  // For local trend model, the right-hand side becomes:
  // rhs = v + prec_theta_1 * [(theta_01 + theta_02) * e_1 + H'B * theta_2]
  //
  // The term H'B * theta_2 accounts for the trend contribution:
  // (H'B * theta_2)[0] = theta_2[0] - theta_02  (boundary condition)
  // (H'B * theta_2)[t] = theta_2[t] - theta_2[t-1] for t = 1,...,n-1

  rhs_vector[0] = v_latent[0] + prec_1 * (theta_0_level + theta_0_trend +
  theta_2_curr[0] - theta_0_trend);

  for (int t = 1; t < n; t++) {
    double theta_2_diff = theta_2_curr[t] - theta_2_curr[t - 1];
    rhs_vector[t] = v_latent[t] + prec_1 * theta_2_diff;
  }

  /* ========== Step 3: Sample theta_1 from Multivariate Normal ========== */
  generate_normal_vector(
    theta_1,       /* r: output matrix */
    rhs_vector,    /* y: right-hand side */
    1.0,           /* a: observational precision */
    prec_1,        /* b: state precision */
    n,             /* n: dimension */
    iter,          /* iter: current iteration */
    1              /* add_a: use (a + b) for last diagonal */
  );

  /* ========== Step 4: Transform to Probability Scale ========== */
  double *alpha_curr = &alpha[current_pos];
  double *theta_1_curr = &theta_1[current_pos];

  for (int t = 0; t < n; t++) {
    alpha_curr[t] = pnorm(theta_1_curr[t], 0.0, 1.0, 1, 0);
  }
}
