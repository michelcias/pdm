/**
 * @file generate_alpha_poisson. h
 * @brief Header for sampling in Poisson state-space models
 * @author Michel H. Montoril
 * @date 2025-12-20
 * @version 1.0
 *
 * @details This header declares optimized functions for MCMC sampling in Poisson
 *          state-space models with log link function.  All functions utilize
 *          memory-efficient current/previous iteration buffers.
 *
 *          **Key features:**
 *          - Log-Poisson:  Component-wise MH with adaptive tuning
 *          - Memory-efficient O(n) temporary storage
 *          - Conditional alpha computation for performance optimization
 *          - Support for local level and local trend models
 *
 *          **Log-Poisson models:**
 *          Use component-wise Metropolis-Hastings updates with adaptive proposal tuning
 *          based on acceptance rates monitored through a sliding window mechanism.
 */

#ifndef GENERATE_ALPHA_POISSON_H
#define GENERATE_ALPHA_POISSON_H

/**
 * @brief Component-wise MH sampler for theta_1 in log-Poisson local level model
 *        with adaptive tuning
 *
 * @details Samples level state vector using component-wise Metropolis-Hastings with
 *          adaptive proposal tuning.
 *
 *          Model: y_t ~ Poisson(alpha_t), alpha_t = exp(theta_{t,1})
 *          State:  theta_{t,1} = theta_{t-1,1} + u_{t,1}
 *
 * @param theta_1_previous        Level state [n] from previous iteration (const)
 * @param theta_1_current         Output level state [n] for current iteration
 * @param alpha_current           Output rates [n] (NULL if compute_alpha=0)
 * @param theta_01_previous       Scalar initial level from previous iteration
 * @param prec_theta1_previous    Scalar level precision from previous iteration
 * @param theta_1_updated         Sliding window [lag_update * n] of acceptances
 * @param y                       Observed counts [n] (const)
 * @param accept_prop             Workspace [n] for acceptance proportions
 * @param log_sigma               Input/output [n] log proposal scales
 * @param hat_theta_1             Workspace [n] for conditional means
 * @param theta_1_new             Workspace [n] for proposals
 * @param log_accept_prob         Workspace [n] for log acceptance
 * @param lag_update              Adaptation window size (> 0)
 * @param n                       Time series length
 * @param iter                    Current iteration (0-based, must be >= 1)
 * @param max_step_size           Maximum adaptation step
 * @param base_adaptation_rate    Base adaptation rate
 * @param decay_exponent          Adaptation decay exponent
 * @param target_acceptance       Target acceptance rate
 * @param min_deviation_threshold Minimum deviation to trigger update
 * @param compute_alpha           Flag for alpha computation (0/1)
 *
 * @note Complexity: O(n) per iteration
 * @note Performance: 10-30% faster when compute_alpha = 0
 *
 * @see cwmh_alpha_log_poisson_locallevel
 * @see adapt_cwmh_parameters
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
                                           int           compute_alpha);

/**
 * @brief Component-wise MH sampler for theta_1 in log-Poisson local trend model
 *        with adaptive tuning
 *
 * @details Samples level state vector using component-wise Metropolis-Hastings with
 *          adaptive proposal tuning for local trend dynamics.
 *
 *          Model: y_t ~ Poisson(alpha_t), alpha_t = exp(theta_{t,1})
 *          State: theta_{t,1} = theta_{t-1,1} + theta_{t-1,2} + u_{t,1}
 *
 * @param theta_1_previous        Level state [n] from previous iteration (const)
 * @param theta_1_current         Output level state [n] for current iteration
 * @param alpha_current           Output rates [n] (NULL if compute_alpha=0)
 * @param theta_2_current         Trend state [n] from current iteration (const)
 * @param theta_01_previous       Scalar initial level from previous iteration
 * @param theta_02_previous       Scalar initial trend from previous iteration
 * @param prec_theta1_previous    Scalar level precision from previous iteration
 * @param theta_1_updated         Sliding window [lag_update * n] of acceptances
 * @param y                       Observed counts [n] (const)
 * @param accept_prop             Workspace [n] for acceptance proportions
 * @param log_sigma               Input/output [n] log proposal scales
 * @param hat_theta_1             Workspace [n] for conditional means
 * @param theta_1_new             Workspace [n] for proposals
 * @param log_accept_prob         Workspace [n] for log acceptance
 * @param lag_update              Adaptation window size (> 0)
 * @param n                       Time series length
 * @param iter                    Current iteration (0-based, must be >= 1)
 * @param max_step_size           Maximum adaptation step
 * @param base_adaptation_rate    Base adaptation rate
 * @param decay_exponent          Adaptation decay exponent
 * @param target_acceptance       Target acceptance rate
 * @param min_deviation_threshold Minimum deviation to trigger update
 * @param compute_alpha           Flag for alpha computation (0/1)
 *
 * @note Complexity: O(n) per iteration
 * @note theta_2_current must be sampled before calling
 * @note Performance: 10-30% faster when compute_alpha = 0
 *
 * @see cwmh_alpha_log_poisson
 * @see adapt_cwmh_parameters
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
                                int           compute_alpha);

#endif /* GENERATE_ALPHA_POISSON_H */
