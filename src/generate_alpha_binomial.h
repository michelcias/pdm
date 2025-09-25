/**
 * @file generate_alpha_binomial.h
 * @brief Header for component-wise MH sampling in logit-binomial state-space models.
 * @author Michel H. Montoril
 * @date 2025-08-10
 * @version 1.0
 *
 * @details This header declares functions for component-wise adaptive MCMC algorithms,
 *          including:
 *          - Component-wise MH updates for local level binomial models
 *          - Component-wise MH updates for local trend binomial models
 */
#ifndef GENERATE_ALPHA_BINOMIAL_H
#define GENERATE_ALPHA_BINOMIAL_H

/**
 * @brief Component-wise Metropolis-Hastings sampler for theta_1 in a logit-binomial
 *        local level model - Optimized version.
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
 *
 * @note Complexity: O(n) per iteration (component-wise updates).
 * @note Uses log-probabilities for numerical stability.
 * @note Forward sampling for better mixing.
 * @note Model is local level (no trend).
 * @note Adaptive tuning performed every lag_update iterations if iter >= lag_update.
 * @note Memory optimization: theta_1_updated uses sliding window instead of full matrix.
 *
 * @warning Each y[k] must satisfy 0 ≤ y[k] ≤ n_trials.
 * @warning Results are invalid if theta_01 or prec_theta_1 do not contain sufficient
 *          history (iter < 1).
 * @warning lag_update must be > 0 for theta_1_updated indexing.
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
                                              double  min_deviation_threshold);

/**
 * @brief Component-wise Metropolis-Hastings sampler for theta_1 in a logit-binomial
 *        local trend model with configurable adaptation threshold.
 *
 * @details Implements an optimized component-wise Metropolis-Hastings algorithm to sample the
 *          level state vector theta_1 in a binomial observation model with local trend
 *          state-space evolution. The adaptation threshold parameter provides flexible
 *          control over when proposal variance adjustments are triggered.
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
                                   double  target_acceptance);

#endif /* GENERATE_ALPHA_BINOMIAL_H */
