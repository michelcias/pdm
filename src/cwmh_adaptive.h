/**
 * @file cwmh_adaptive.h
 * @brief Header for adaptive parameter tuning in component-wise Metropolis-Hastings (CWMH)
 *        sampling - Optimized version.
 * @author Michel H. Montoril
 * @date 2025-10-02
 * @version 1.2
 *
 * @details This header declares optimized functions for adaptive MCMC algorithms, including:
 *          - Online adaptation of proposal variance (log_sigma) in CWMH chains with sliding
 *            window memory optimization
 *          - Efficient acceptance rate monitoring via circular buffer indexing
 *          - Diminishing adaptation for robust convergence with configurable sensitivity
 *          - Vectorized operations with loop unrolling for optimal performance
 *          - Enhanced cache validation to handle varying lag_update parameters
 *          - Safe handling of large arrays with overflow protection
 *
 * @changelog
 * - v1.2 (2025-10-02): Removed legacy function and unused constants. Fixed header file
 *   extension, improved large array handling with size_t for indices, enhanced thread
 *   safety documentation, and standardized version references. Added NaN/infinite validation.
 * - v1.2 (2025-09-30): Simplified to single vectorized implementation. Enhanced cache
 *   validation for lag_update parameter changes. Updated documentation to reflect
 *   performance characteristics and cache behavior improvements.
 * - v1.1 (2025-09-23): Added min_deviation_threshold parameter for flexible adaptation.
 */

#ifndef CWMH_ADAPTIVE_H
#define CWMH_ADAPTIVE_H

#include <R.h>
#include <Rmath.h>
#include <string.h>  /* for memset */
#include <stddef.h>  /* for size_t */
#include <stdint.h>  /* for SIZE_MAX */

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
 *          - Safe handling of large arrays using size_t for index calculations
 *
 *          **Thread safety:** This function is NOT thread-safe due to internal static caching.
 *          Multiple threads calling this function simultaneously may experience:
 *          - Cache corruption leading to incorrect step size calculations
 *          - Race conditions in cache validation logic
 *          - Inconsistent inv_lag_update values between threads
 *          For multi-threaded applications, ensure external synchronization or use separate
 *          instances per thread with thread-local storage.
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
 *          The observed acceptance rate accept_prop[k] is computed over the last lag_update
 *          iterations using the sliding window stored in theta_updated.
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
 *                        Maximum safe value: SIZE_MAX / n to prevent overflow.
 * @param n               Number of components (dimensions) in the parameter vector. Must be > 0.
 *                        Typical values: 10-1000 for time series models.
 *                        Maximum safe value: SIZE_MAX / lag_update to prevent overflow.
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
 *       acceptance proportion calculations. This prevents incorrect normalization when
 *       window sizes change between calls.
 * @note Performance: Vectorized implementation provides optimal performance for typical
 *       problem sizes (n = 10 to 1000, lag_update = 20 to 200) without additional complexity
 *       of problem-size-dependent algorithm selection.
 * @note Large arrays: Uses size_t internally for safe index calculations with large arrays.
 *
 * @note Common parameter choices: max_step_size = 0.01-0.1, base_adaptation_rate = 1.0-10.0,
 *       decay_exponent = 0.3-0.8, lag_update = 50-200, min_deviation_threshold = 1.0/lag_update.
 * @note Threshold selection: The practical choice 1.0/lag_update corresponds to the deviation
 *       from a single additional acceptance/rejection in the sliding window. For lag_update=50
 *       and target_acceptance=0.44, this means adaptation occurs only when moving from the
 *       expected 22 acceptances to 21 or 23 acceptances (deviation >= 0.02).
 * @note Adaptation schedule: Uses diminishing adaptation (step_size -> 0 as iter -> infinity)
 *       for theoretical convergence guarantees (Roberts and Rosenthal, 2007).
 *
 * @warning Results are invalid if theta_updated does not contain sufficient history
 *          (iter < lag_update).
 * @warning Inappropriate adaptation rates and step sizes may lead to poor MCMC mixing.
 * @warning Large n (> 10^6) may require additional memory management considerations.
 * @warning Thread safety: This function is NOT thread-safe due to static caching.
 *          Use external synchronization in multi-threaded environments.
 * @warning Cache invalidation: The cache automatically resets when lag_update changes,
 *          ensuring correctness but potentially causing slight overhead on first call
 *          with a new window size.
 * @warning Overflow protection: Ensure lag_update * n <= SIZE_MAX to prevent index overflow.
 *          The function performs basic overflow checks but cannot guarantee safety for all
 *          possible input combinations.
 *
 * @see cwmh_alpha_logit_binomial_locallevel
 * @see cwmh_alpha_logit_binomial
 * @see Roberts and Rosenthal (2007), "Coupling and Ergodicity of Adaptive MCMC"
 * @since version 1.0
 * @version 1.2
 *
 * @example
 * @code
 * // Recommended usage with practical threshold based on window size
 * double practical_threshold = 1.0 / lag_update;  // e.g., 0.02 for lag_update=50
 * if (iter >= lag_update && (iter % lag_update == 0)) {
 *     adapt_cwmh_parameters(
 *         theta_1_updated,        // Sliding window of acceptances
 *         accept_prop,            // Output: current acceptance rates
 *         log_sigma,              // Input/output: proposal scales
 *         50,                     // lag_update: sliding window size
 *         n,                      // Number of components
 *         iter,                   // Current iteration
 *         0.05,                   // max_step_size
 *         2.0,                    // base_adaptation_rate
 *         0.6,                    // decay_exponent
 *         0.44,                   // target_acceptance
 *         practical_threshold     // min_deviation_threshold = 0.02
 *     );
 * }
 *
 * // More conservative adaptation (require 2+ acceptance changes)
 * adapt_cwmh_parameters(..., 2.0 / lag_update);  // e.g., 0.04 for lag_update=50
 *
 * // Very sensitive adaptation (single acceptance change triggers update)
 * adapt_cwmh_parameters(..., 1.0 / lag_update);  // e.g., 0.02 for lag_update=50
 *
 * // Disable threshold (update for any deviation, not recommended)
 * adapt_cwmh_parameters(..., 0.0);               // Maximum sensitivity
 * @endcode
 */
void adapt_cwmh_parameters(double *theta_updated,
                           double *accept_prop,
                           double *log_sigma,
                           int     lag_update,
                           int     n,
                           int     iter,
                           double  max_step_size,
                           double  base_adaptation_rate,
                           double  decay_exponent,
                           double  target_acceptance,
                           double  min_deviation_threshold);


/**
 * @brief Reset adaptation cache (utility function for testing and debugging).
 *
 * @details Clears internal caches to ensure fresh computations. This function is primarily
 *          intended for unit testing and debugging scenarios where predictable behavior
 *          is required across multiple function calls. Enhanced to reset lag_update cache
 *          validation field to ensure proper recalculation when window size changes.
 *
 *          **Thread safety:** This function is thread-safe for resetting the cache, but
 *          should not be called concurrently with adapt_cwmh_parameters. In multi-threaded
 *          environments, ensure proper synchronization to avoid race conditions.
 *
 *          **Usage scenarios:**
 *          - Unit testing that requires deterministic cache behavior
 *          - Debugging adaptation issues by forcing cache regeneration
 *          - Performance benchmarking with controlled cache states
 *          - Switching between different parameter sets in the same session
 *          - Ensuring correct behavior when lag_update changes between calls
 *
 *          **Cache fields reset:**
 *          - cached_step_size: Adaptation step size cache
 *          - cached_iter: Iteration number for step size validation
 *          - cached_base_rate: Base adaptation rate for validation
 *          - cached_max_step: Maximum step size for validation
 *          - cached_decay_exp: Decay exponent for validation
 *          - inv_lag_update: Precomputed inverse of lag_update
 *          - cached_lag_update: Lag update value for validation
 *
 * @return None.
 *
 * @note This function modifies static cache variables atomically.
 * @note Performance impact is minimal as cache recomputation is fast.
 * @note Calling this function will cause the next adapt_cwmh_parameters call to
 *       recompute all cached values.
 * @note Cache recomputation occurs automatically on next function call.
 *
 * @warning Only call this function when necessary, as it removes performance benefits
 *          of caching until values are recomputed.
 * @warning Particularly important to call between tests that use different lag_update values
 *          to prevent cache-related test failures.
 * @warning Not required in production code as cache validation handles parameter changes
 *          automatically.
 * @warning Do not call concurrently with adapt_cwmh_parameters in multi-threaded code.
 *
 * @see adapt_cwmh_parameters
 * @since version 1.0
 * @version 1.2
 */
void reset_adaptation_cache(void);

#endif /* CWMH_ADAPTIVE_H */
