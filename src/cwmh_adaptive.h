/**
 * @file cwmh_adaptive.h
 * @brief Header for adaptive parameter tuning in component-wise Metropolis-Hastings (CWMH)
 *        sampling.
 * @author Michel H. Montoril
 * @date 2025-08-09
 * @version 1.0
 *
 * @details This header declares core functions for adaptive MCMC algorithms, including:
 *          - Online adaptation of proposal variance (log_sigma) in CWMH chains
 *          - Sliding window acceptance rate monitoring
 *          - Diminishing adaptation for robust convergence
 */

#ifndef CWMH_ADAPTIVE_H
#define CWMH_ADAPTIVE_H

/**
 * @brief Adapts proposal variance for each component in a Component-Wise Metropolis-Hastings
 *        (CWMH) MCMC sampler.
 *
 * @details This function updates the proposal log standard deviations (log_sigma) for each
 *          dimension to achieve a target acceptance rate in adaptive MCMC. The adaptation uses a
 *          sliding window of recent updates to compute acceptance rates for each component, then
 *          adjusts log_sigma towards the target acceptance rate using a diminishing adaptation
 *          step.
 *
 *          The adaptation step size is computed as:
 *              step_size = min(max_step_size, base_adaptation_rate / iter^decay_exponent)
 *          The log_sigma update rule is:
 *              log_sigma[k] += sign(acceptance_probs[k] - target_acceptance) * step_size
 *          where acceptance_probs[k] is the observed acceptance rate for component k in the last
 *          lag_update iterations.
 *
 * @param theta_updated   Vectorized (B x n) matrix of acceptance indicators (1 if accepted,
 *                        0 if not).
 * @param acceptance_probs Output vector (size n) of acceptance rates for each parameter component.
 * @param log_sigma       Input/output vector (size n) of log proposal standard deviations
 *                        (updated in-place).
 * @param lag_update      Number of recent iterations to use for acceptance rate calculation
 *                        (sliding window).
 * @param n               Number of components (dimensions) in the parameter vector.
 * @param iter            Current MCMC iteration (0-based). Must be >= lag_update.
 * @param max_step_size   Maximum adaptation step size.
 * @param base_adaptation_rate  Base adaptation rate (initial step size).
 * @param decay_exponent  Exponent controlling decay speed of adaptation step size
 *                        (e.g. 0.5 = sqrt, 1.0 = linear).
 * @param target_acceptance     Target acceptance rate for parameter optimization (e.g. 0.44
 *                        univariate, 0.234 multivariate).
 * @return None (results are written to acceptance_probs and log_sigma).
 *
 * @note Complexity: O(n * lag_update) per call.
 * @note Requires iter >= lag_update for valid adaptation history.
 * @note Common choices: max_step_size = 0.01, base_adaptation_rate = 1.0-10.0,
 *       decay_exponent = 0.3-0.8.
 * @note Numerical stability depends on adaptation parameters.
 *
 * @warning Results are invalid if theta_updated does not contain sufficient history
 *          (iter < lag_update).
 * @warning Inappropriate adaptation rates and step sizes may lead to poor MCMC mixing.
 *
 */
void adapt_cwmh_parameters(double *theta_updated,
                           double *acceptance_probs,
                           double *log_sigma,
                           int lag_update,
                           int n,
                           int iter,
                           double max_step_size,
                           double base_adaptation_rate,
                           double decay_exponent,
                           double target_acceptance);

#endif /* CWMH_ADAPTIVE_H */
