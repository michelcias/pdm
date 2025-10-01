/**
 * @file cwmh_adaptive.h
 * @brief Header for adaptive parameter tuning in component-wise Metropolis-Hastings (CWMH)
 *        sampling - Optimized version.
 * @author Michel H. Montoril
 * @date 2025-09-30
 * @version 1.2
 *
 * @details This header declares optimized functions for adaptive MCMC algorithms, including:
 *          - Online adaptation of proposal variance (log_sigma) in CWMH chains with sliding
 *            window memory optimization
 *          - Efficient acceptance rate monitoring via circular buffer indexing
 *          - Diminishing adaptation for robust convergence with configurable sensitivity
 *          - Vectorized operations with loop unrolling for optimal performance
 *          - Enhanced cache validation to handle varying lag_update parameters
 *          - Backward compatibility support for legacy code
 *
 * @changelog
 * - v1.2 (2025-09-30): Simplified to single vectorized implementation. Enhanced cache
 *   validation for lag_update parameter changes. Updated documentation to reflect
 *   performance characteristics and cache behavior improvements.
 * - v1.1 (2025-09-23): Added min_deviation_threshold parameter for flexible adaptation.
 */

#ifndef CWMH_ADAPTIVE_H
#define CWMH_ADAPTIVE_H

/**
 * @brief Adapts proposal variance for each component in a Component-Wise Metropolis-Hastings
 *        (CWMH) MCMC sampler - Optimized version compatible with sliding window implementation.
 *
 * @details This function updates the proposal log standard deviations (log_sigma) for each
 *          dimension to achieve a target acceptance rate in adaptive MCMC. The adaptation uses a
 *          sliding window of recent updates to compute acceptance rates for each component, then
 *          adjusts log_sigma towards the target acceptance rate using a diminishing adaptation
 *          step.
 *
 *          **Key optimizations implemented:**
 *          - Cached step size computation to avoid expensive power operations
 *          - Vectorized acceptance proportion calculations with loop unrolling
 *          - Precomputed inverse lag_update to replace division with multiplication
 *          - Validated cache for inv_lag_update to handle varying window sizes correctly
 *          - Threshold-based filtering to reduce spurious updates from noise
 *          - Optimized memory access patterns for modern CPU cache hierarchies
 *
 *          **Memory layout compatibility:** This version is designed to work with the optimized
 *          sliding window implementation in cwmh_binomial.c, where theta_updated is organized
 *          as a (lag_update x n) matrix with circular indexing.
 *
 *          **Adaptation schedule:**
 *          The adaptation step size diminishes over iterations according to:
 *              step_size = min(max_step_size, base_adaptation_rate / iter^decay_exponent)
 *
 *          **Update rule:**
 *          For each component k where |accept_prop[k] - target| > threshold:
 *              log_sigma[k] += sign(accept_prop[k] - target_acceptance) * step_size
 *
 * @param theta_updated   Sliding window matrix of acceptance indicators (lag_update x n),
 *                        organized in row-major order with circular indexing. Each element
 *                        is 1 if accepted, 0 if rejected.
 * @param accept_prop     Output vector (size n) of acceptance proportions for each parameter
 *                        component. Results are written here.
 * @param log_sigma       Input/output vector (size n) of log proposal standard deviations
 *                        (updated in-place).
 * @param lag_update      Number of recent iterations to use for acceptance rate calculation
 *                        (sliding window size). Must be > 0. Typical values: 20-200.
 * @param n               Number of components (dimensions) in the parameter vector. Must be > 0.
 *                        Typical values: 10-1000 for time series models.
 * @param iter            Current MCMC iteration (0-based). Must be >= lag_update.
 * @param max_step_size   Maximum adaptation step size. Must be > 0. Typical values: 0.01-0.1.
 * @param base_adaptation_rate  Base adaptation rate (initial step size). Must be > 0.
 *                        Typical values: 0.5-10.0.
 * @param decay_exponent  Exponent controlling decay speed of adaptation step size
 *                        (e.g. 0.5 = sqrt, 1.0 = linear). Must be > 0. Typical values: 0.3-0.8.
 * @param target_acceptance     Target acceptance rate for parameter optimization
 *                        (e.g. 0.44 univariate, 0.234 multivariate). Must be in (0,1).
 * @param min_deviation_threshold Minimum absolute deviation from target_acceptance required
 *                        to trigger log_sigma updates. Must be >= 0. A practical choice is
 *                        1.0/lag_update, which corresponds to the deviation caused by a single
 *                        additional acceptance/rejection in the sliding window. For example,
 *                        with lag_update=50, use 0.02. Values of 0.0 disable the threshold.
 *
 * @return None (results are written to accept_prop and log_sigma).
 *
 * @note Computational complexity: O(n * lag_update) per call with low constant factor.
 * @note Memory requirements: O(1) additional space beyond input/output arrays.
 * @note Cache performance: Optimized for modern CPU cache hierarchies with sequential access.
 * @note Numerical stability: Uses threshold-based updates and stable arithmetic.
 * @note Cache validation: Automatically detects lag_update changes between function calls
 *       and invalidates the cached inv_lag_update value to ensure mathematically correct
 *       acceptance proportion calculations.
 * @note Performance: Vectorized implementation provides optimal performance for typical
 *       problem sizes without additional complexity.
 *
 * @warning Results are invalid if theta_updated does not contain sufficient history
 *          (iter < lag_update).
 * @warning Thread safety: This function is not thread-safe due to static caching.
 *
 * @see cwmh_alpha_logit_binomial_locallevel
 * @see cwmh_alpha_logit_binomial
 * @see Roberts and Rosenthal (2007), "Coupling and Ergodicity of Adaptive MCMC"
 */
void adapt_cwmh_parameters(double *theta_updated,
                           double *accept_prop,
                           double *log_sigma,
                           int lag_update,
                           int n,
                           int iter,
                           double max_step_size,
                           double base_adaptation_rate,
                           double decay_exponent,
                           double target_acceptance,
                           double min_deviation_threshold);

/**
 * @brief Legacy wrapper for adapt_cwmh_parameters with backward compatibility.
 *
 * @details This function provides backward compatibility with existing code by calling
 *          the optimized adapt_cwmh_parameters function with a default threshold value.
 *          The default threshold (1.0/lag_update) matches the previous hardcoded behavior.
 *
 *          **Usage recommendation:** New code should use adapt_cwmh_parameters directly
 *          with an explicit threshold parameter for better control and clarity.
 *
 * @param theta_updated   Sliding window matrix of acceptance indicators (lag_update x n).
 * @param accept_prop     Output vector (size n) of acceptance proportions.
 * @param log_sigma       Input/output vector (size n) of log proposal standard deviations.
 * @param lag_update      Sliding window size for acceptance rate calculation.
 * @param n               Number of parameter components.
 * @param iter            Current MCMC iteration (0-based).
 * @param max_step_size   Maximum adaptation step size.
 * @param base_adaptation_rate  Base adaptation rate.
 * @param decay_exponent  Adaptation decay exponent.
 * @param target_acceptance     Target acceptance rate.
 *
 * @return None (results are written to accept_prop and log_sigma).
 *
 * @note This function uses a default min_deviation_threshold of 1.0/lag_update.
 * @note Consider migrating to adapt_cwmh_parameters for explicit threshold control.
 *
 * @deprecated Use adapt_cwmh_parameters with explicit min_deviation_threshold instead.
 *
 * @see adapt_cwmh_parameters
 */
void adapt_cwmh_parameters_legacy(double *theta_updated,
                                  double *accept_prop,
                                  double *log_sigma,
                                  int lag_update,
                                  int n,
                                  int iter,
                                  double max_step_size,
                                  double base_adaptation_rate,
                                  double decay_exponent,
                                  double target_acceptance);

/**
 * @brief Reset adaptation cache (utility function for testing and debugging).
 *
 * @details Clears internal caches to ensure fresh computations. This function is primarily
 *          intended for unit testing and debugging scenarios where predictable behavior
 *          is required across multiple function calls. Enhanced to reset lag_update cache
 *          validation field.
 *
 *          **Usage scenarios:**
 *          - Unit testing that requires deterministic cache behavior
 *          - Debugging adaptation issues by forcing cache regeneration
 *          - Performance benchmarking with controlled cache states
 *          - Switching between different parameter sets in the same session
 *          - Ensuring correct behavior when lag_update changes between calls
 *
 * @return None.
 *
 * @note This function is thread-safe as it only modifies static cache variables.
 * @note Version 1.2 enhancement: Now includes cached_lag_update reset.
 * @note Performance impact is minimal as cache recomputation is fast.
 *
 * @warning Only call this function when necessary.
 * @warning Particularly important between tests using different lag_update values.
 *
 * @see adapt_cwmh_parameters
 * @since version 1.0
 * @version 1.2
 */
void reset_adaptation_cache(void);

#endif /* CWMH_ADAPTIVE_H */
