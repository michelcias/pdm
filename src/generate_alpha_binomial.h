/**
 * @file generate_alpha_binomial.h
 * @brief Header for component-wise Metropolis-Hastings sampling in logit-binomial models
 * @author Michel H. Montoril
 * @date 2025-09-23
 * @version 1.2
 *
 * @details This header declares functions for adaptive MCMC sampling in binomial
 *          state-space models with logit link, including:
 *          - Component-wise MH updates for local level binomial models
 *          - Component-wise MH updates for local trend binomial models
 *          - Configurable adaptation threshold parameters for enhanced flexibility
 *
 * @changelog
 * - v1.2 (2025-09-23): Updated function signatures to include min_deviation_threshold
 *   parameter, providing fine-grained control over adaptation sensitivity.
 */

#ifndef GENERATE_ALPHA_BINOMIAL_H
#define GENERATE_ALPHA_BINOMIAL_H

/**
 * @brief Component-wise Metropolis-Hastings sampler for theta_1 in a logit-binomial
 *        local level model with configurable adaptation threshold.
 *
 * @details Implements an optimized component-wise Metropolis-Hastings algorithm to sample the
 *          level state vector theta_1 in a binomial observation model with logit link.
 *          The adaptation threshold parameter provides flexible control over when
 *          proposal variance adjustments are triggered.
 *
 * @param theta_1              Matrix of level states (vectorized B x n), input/output.
 * @param theta_01             Vector of initial level states (size B).
 * @param theta_1_updated      Sliding window matrix of acceptance indicators
 *                             (vectorized lag_update x n), output. Uses circular indexing.
 * @param alpha                Matrix of transformed probabilities (vectorized B x n), output.
 * @param prec_theta_1         Vector of level precision parameters (size B).
 * @param y                    Vector of observed binomial counts (size n).
 * @param accept_prop          Vector of acceptance proportions for each component (size n).
 * @param log_sigma            Vector of log proposal standard deviations (size n).
 * @param hat_theta_1          Temporary vector for conditional means (size n).
 * @param theta_1_new          Temporary vector for proposed values (size n).
 * @param log_accept_prob      Temporary vector for log acceptance probabilities (size n).
 * @param lag_update           Integer scalar, sliding window size for adaptation frequency.
 * @param n_trials             Number of Bernoulli trials (double).
 * @param n                    Length of the time series.
 * @param iter                 Current MCMC iteration (0-based).
 * @param max_step_size        Double scalar, maximum adaptation step size.
 * @param base_adaptation_rate Double scalar, initial adaptation rate before decay.
 * @param decay_exponent       Double scalar, exponent for diminishing adaptation schedule.
 * @param target_acceptance    Double scalar, target acceptance rate for adaptive tuning.
 * @param min_deviation_threshold Double scalar, minimum absolute deviation from target_acceptance
 *                             required to trigger log_sigma updates. Must be >= 0.
 *
 * @note Model is local level (random walk only, no trend).
 * @note Uses sliding window memory optimization.
 * @note Configurable threshold allows fine-tuned adaptation sensitivity.
 * @note Recommended threshold: 1.0/lag_update for practical applications.
 *
 * @warning min_deviation_threshold must be >= 0.0.
 * @warning lag_update must be > 0 for sliding window indexing.
 *
 * @see adapt_cwmh_parameters
 * @see cwmh_alpha_logit_binomial_locallevel
 * @since version 1.2
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
 * @param accept_prop          Vector of acceptance proportions for each component (size n).
 * @param log_sigma            Vector of log proposal standard deviations (size n).
 * @param hat_theta_1          Temporary vector for conditional means (size n).
 * @param theta_1_new          Temporary vector for proposed values (size n).
 * @param log_accept_prob      Temporary vector for log acceptance probabilities (size n).
 * @param lag_update           Integer scalar, sliding window size for adaptation frequency.
 * @param n_trials             Number of Bernoulli trials (double).
 * @param n                    Length of the time series.
 * @param iter                 Current MCMC iteration (0-based).
 * @param max_step_size        Double scalar, maximum adaptation step size.
 * @param base_adaptation_rate Double scalar, initial adaptation rate before decay.
 * @param decay_exponent       Double scalar, exponent for diminishing adaptation schedule.
 * @param target_acceptance    Double scalar, target acceptance rate for adaptive tuning.
 * @param min_deviation_threshold Double scalar, minimum absolute deviation from target_acceptance
 *                             required to trigger log_sigma updates. Must be >= 0.
 *
 * @note Model is local trend (random walk + trend component).
 * @note Uses sliding window memory optimization.
 * @note Configurable threshold allows fine-tuned adaptation sensitivity.
 * @note Recommended threshold: 1.0/lag_update for practical applications.
 *
 * @warning min_deviation_threshold must be >= 0.0.
 * @warning lag_update must be > 0 for sliding window indexing.
 *
 * @see adapt_cwmh_parameters
 * @see cwmh_alpha_logit_binomial
 * @since version 1.2
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
                                   double  min_deviation_threshold);

#endif /* GENERATE_ALPHA_BINOMIAL_H */
