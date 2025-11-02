/**
 * @file cwmh_binomial.h
 * @brief Header for component-wise Metropolis-Hastings sampling in logit-binomial state-space models
 * @author Michel H. Montoril
 * @date 2025-10-11
 * @version 1.0
 *
 * @details This header declares optimized functions for component-wise Metropolis-Hastings (CWMH)
 *          sampling in binomial dynamic models with logit link. All functions utilize
 *          memory-efficient current/previous iteration buffers and incorporate adaptive
 *          proposal tuning mechanisms.
 *
 *          **Key features:**
 *          - Component-wise MH for local level binomial models
 *          - Component-wise MH for local trend binomial models
 *          - Memory-efficient O(n) temporary storage (current/previous buffers)
 *          - Precision caching to avoid repeated expensive calculations
 *          - Conditional alpha computation with branch hoisting optimization
 *          - Fast memcpy path for efficient state updates
 *          - Sliding window acceptance tracking for adaptive MCMC
 *
 *          **Mathematical framework:**
 *          All samplers target the conditional posterior distribution of theta_1
 *          in binomial state-space models with logit link:
 *          y_t ~ Binomial(n_trials, alpha_t), where alpha_t = logit^{-1}(theta_{t,1})
 *
 *          The component-wise Metropolis-Hastings algorithm samples each element
 *          theta_{t,1} sequentially, using Gaussian random walk proposals with
 *          adaptive standard deviations based on acceptance rates.
 *
 *          **Precision structure:**
 *          Conditional distributions have standard deviations derived from the
 *          tridiagonal precision matrix structure of polynomial dynamic models:
 *          - Regular elements: sd_regular = 1/sqrt(prec_theta_1 * 2)
 *          - Boundary element: sd_last = 1/sqrt(prec_theta_1)
 */

#ifndef CWMH_BINOMIAL_H
#define CWMH_BINOMIAL_H

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
                                          int           compute_alpha);

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
                               int           compute_alpha);

#endif /* CWMH_BINOMIAL_H */
