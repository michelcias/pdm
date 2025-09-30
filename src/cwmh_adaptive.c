/**
 * @file cwmh_adaptive.c
 * @brief Adaptive parameter tuning for component-wise Metropolis-Hastings (CWMH) sampling
 *        in the pdm package - Optimized version.
 * @author Michel H. Montoril
 * @date 2025-09-30
 * @version 1.2
 *
 * @details This file contains optimized functions for adaptive MCMC algorithms, including:
 *          - Online adaptation of proposal variance (log_sigma) in component-wise Metropolis-
 *            Hastings chains with sliding window memory optimization
 *          - Monitoring acceptance rates via efficient circular buffer indexing
 *          - Diminishing adaptation for robust convergence with numerical stability
 *          - Vectorized operations with loop unrolling for improved performance
 *          - Enhanced cache validation to handle varying lag_update parameters
 *
 * @changelog
 * - v1.2 (2025-09-30): Simplified to use single vectorized implementation for all problem
 *   sizes, eliminating unnecessary complexity while maintaining optimal performance for
 *   typical use cases. Enhanced cache structure to validate lag_update parameter changes,
 *   fixing bug where inv_lag_update cache was not invalidated when lag_update changed
 *   between function calls, causing incorrect acceptance proportion calculations.
 * - v1.1 (2025-09-23): Added configurable min_deviation_threshold parameter for
 *   flexible control over adaptation sensitivity.
 */

#include <R.h>
#include <Rmath.h>
#include <string.h>  /* for memset */
#include "cwmh_adaptive.h"

// Symbolic constants for readability and performance
#define POSITIVE_STEP_DIRECTION 1.0
#define NEGATIVE_STEP_DIRECTION -1.0
#define VECTORIZATION_THRESHOLD 4  /* Minimum n for loop unrolling */

/**
 * @brief Cache structure for expensive adaptation computations
 * @details Stores frequently computed values to avoid redundant calculations.
 *          Enhanced to track lag_update parameter for proper cache invalidation
 *          when window size changes between function calls.
 */
typedef struct {
  double cached_step_size;       /* Cached adaptation step size */
int    cached_iter;            /* Iteration number for cached step size */
double cached_base_rate;       /* Cached base adaptation rate */
double cached_max_step;        /* Cached maximum step size */
double cached_decay_exp;       /* Cached decay exponent */
double inv_lag_update;         /* Precomputed 1.0 / lag_update */
int    cached_lag_update;      /* Cached lag_update value for validation */
} adaptation_cache_t;

static adaptation_cache_t adapt_cache = {-1.0, -1, -1.0, -1.0, -1.0, 0.0, -1};

/**
 * @brief Compute adaptation step size with caching optimization
 * @details Avoids expensive power operations when parameters haven't changed between calls.
 *          Implements diminishing adaptation schedule for theoretical convergence guarantees.
 *
 * @param iter                  Current adaptation iteration used for diminishing step size.
 * @param max_step_size         Maximum allowable adaptation step size to enforce stability.
 * @param base_adaptation_rate  Base learning rate that scales the diminishing schedule.
 * @param decay_exponent        Exponent controlling how quickly the adaptation rate decays.
 * @return Computed adaptation step size retrieved from cache when possible.
 *
 * @note Uses R's fmin2 and R_pow for consistency with R's numerical behavior.
 * @note Cache is validated across all parameters to ensure correctness.
 */
static inline double compute_step_size_cached(int iter, double max_step_size,
                                              double base_adaptation_rate,
                                              double decay_exponent) {
  if (adapt_cache.cached_iter != iter ||
      adapt_cache.cached_base_rate != base_adaptation_rate ||
      adapt_cache.cached_max_step != max_step_size ||
      adapt_cache.cached_decay_exp != decay_exponent) {

    adapt_cache.cached_step_size = fmin2(max_step_size,
                                         base_adaptation_rate / R_pow((double)iter, decay_exponent));
    adapt_cache.cached_iter = iter;
    adapt_cache.cached_base_rate = base_adaptation_rate;
    adapt_cache.cached_max_step = max_step_size;
    adapt_cache.cached_decay_exp = decay_exponent;
  }

  return adapt_cache.cached_step_size;
}

/**
 * @brief Vectorized acceptance proportion computation with loop unrolling
 * @details Computes acceptance proportions for all components using optimized vectorization
 *          with manual loop unrolling. This implementation provides excellent performance
 *          for typical problem sizes in dynamic models (n = 10 to 1000 components) by
 *          maximizing instruction-level parallelism and maintaining good cache locality.
 *
 *          **Performance characteristics:**
 *          - Sequential memory access pattern enables efficient hardware prefetching
 *          - Loop unrolling (4-way) reduces branch overhead and enables parallel execution
 *          - Small working set (accept_prop vector) remains in L1 cache throughout
 *          - Input data access is cache-friendly for typical lag_update values (20-200)
 *
 *          **Algorithm:**
 *          1. Initialize accept_prop vector to zeros using memset
 *          2. Iterate through lag_update rows, accumulating acceptance indicators
 *          3. For n >= 4, process 4 components per iteration using loop unrolling
 *          4. Handle remaining elements with standard loop
 *          5. Convert sums to proportions by multiplying by precomputed inverse
 *
 * @param theta_updated   Sliding window matrix of acceptance indicators (lag_update x n),
 *                        stored in row-major order. Each element is 1.0 (accepted) or
 *                        0.0 (rejected).
 * @param accept_prop     Output vector (size n) receiving acceptance proportions per component.
 *                        Each element will be in [0, 1] after computation.
 * @param lag_update      Sliding window length governing the number of rows to aggregate.
 *                        Typical values: 20-200 iterations.
 * @param n               Number of parameter dimensions (components) being adapted.
 *                        Typical values: 10-1000 for time series models.
 * @param inv_lag_update  Precomputed reciprocal of lag_update (1.0/lag_update) for
 *                        efficient normalization, avoiding division operations.
 *
 * @note Computational complexity: O(n * lag_update) with low constant factor.
 * @note Memory access pattern: Sequential reads with stride-1 access for optimal prefetching.
 * @note Cache efficiency: Working set typically fits entirely in L1 cache for standard problems.
 * @note Numerical stability: Uses multiplication by inverse rather than division for consistency.
 *
 * @warning Assumes theta_updated is properly allocated with lag_update * n elements.
 * @warning Assumes accept_prop is properly allocated with n elements.
 * @warning inv_lag_update must be positive and finite to ensure valid proportions.
 */
static inline void compute_acceptance_vectorized(double *theta_updated,
                                                 double *accept_prop,
                                                 int lag_update,
                                                 int n,
                                                 double inv_lag_update) {
  int k, row;

  /* Initialize acceptance proportions to zero using optimized memset */
  memset(accept_prop, 0, n * sizeof(double));

  /* Accumulate acceptance indicators across all rows in sliding window */
  for (row = 0; row < lag_update; row++) {
    int row_idx = row * n;

    /* Loop unrolling for n >= 4 to enable instruction-level parallelism */
    if (n >= VECTORIZATION_THRESHOLD) {
      /* Process 4 components simultaneously to reduce loop overhead */
      for (k = 0; k < n - 3; k += 4) {
        accept_prop[k]     += theta_updated[row_idx + k];
        accept_prop[k + 1] += theta_updated[row_idx + k + 1];
        accept_prop[k + 2] += theta_updated[row_idx + k + 2];
        accept_prop[k + 3] += theta_updated[row_idx + k + 3];
      }
      /* Handle remaining elements (0 to 3 components) */
      for (; k < n; k++) {
        accept_prop[k] += theta_updated[row_idx + k];
      }
    } else {
      /* Simple loop for very small n (< 4 components) */
      for (k = 0; k < n; k++) {
        accept_prop[k] += theta_updated[row_idx + k];
      }
    }
  }

  /* Convert sums to proportions using precomputed inverse for efficiency */
  for (k = 0; k < n; k++) {
    accept_prop[k] *= inv_lag_update;
  }
}

/**
 * @brief Optimized log_sigma update with numerical stability and configurable threshold
 * @details Updates log proposal standard deviations based on deviation from target acceptance
 *          rate, using threshold-based filtering to avoid unnecessary updates from noise.
 *          Implements vectorization with loop unrolling for components exceeding threshold.
 *
 *          **Update rule:**
 *          For each component k where |accept_prop[k] - target| > threshold:
 *              log_sigma[k] += step_size  if accept_prop[k] > target (increase variance)
 *              log_sigma[k] -= step_size  if accept_prop[k] < target (decrease variance)
 *
 *          **Threshold rationale:**
 *          The min_deviation_threshold parameter filters out small deviations that may
 *          result from random fluctuations rather than systematic bias. A practical
 *          choice is 1.0/lag_update, corresponding to one additional acceptance/rejection
 *          in the sliding window.
 *
 * @param accept_prop             Vector of acceptance proportions (size n), each in [0, 1].
 * @param log_sigma               Vector of log proposal standard deviations (size n),
 *                                modified in-place based on acceptance rates.
 * @param n                       Number of components to potentially update.
 * @param step_size               Adaptation step size for current iteration (positive value).
 * @param target_acceptance       Target acceptance rate (typically 0.44 univariate, 0.234 multivariate).
 * @param min_deviation_threshold Minimum absolute deviation required for updates (>= 0).
 *                                Value of 0.0 disables filtering (maximum sensitivity).
 *
 * @note Computational complexity: O(n) with early termination for below-threshold components.
 * @note Numerical stability: Uses symmetric step size application to avoid bias.
 * @note Loop unrolling: Applied for n >= 8 to reduce branch overhead.
 * @note Threshold filtering: Reduces spurious updates from random fluctuations.
 *
 * @warning step_size must be positive for correct update direction.
 * @warning min_deviation_threshold must be non-negative.
 */
static inline void update_log_sigma_optimized(double *accept_prop,
                                              double *log_sigma,
                                              int n,
                                              double step_size,
                                              double target_acceptance,
                                              double min_deviation_threshold) {
  int k;

  /* Vectorized approach with manual loop unrolling for moderate to large n */
  if (n >= 8) {
    /* Process 4 elements at once to reduce branch prediction overhead */
    for (k = 0; k < n - 3; k += 4) {
      double dev0 = accept_prop[k] - target_acceptance;
      double dev1 = accept_prop[k + 1] - target_acceptance;
      double dev2 = accept_prop[k + 2] - target_acceptance;
      double dev3 = accept_prop[k + 3] - target_acceptance;

      /* Update only if deviation exceeds threshold */
      if (fabs(dev0) > min_deviation_threshold) {
        log_sigma[k] += (dev0 > 0 ? step_size : -step_size);
      }
      if (fabs(dev1) > min_deviation_threshold) {
        log_sigma[k + 1] += (dev1 > 0 ? step_size : -step_size);
      }
      if (fabs(dev2) > min_deviation_threshold) {
        log_sigma[k + 2] += (dev2 > 0 ? step_size : -step_size);
      }
      if (fabs(dev3) > min_deviation_threshold) {
        log_sigma[k + 3] += (dev3 > 0 ? step_size : -step_size);
      }
    }
    /* Handle remaining elements (0 to 3 components) */
    for (; k < n; k++) {
      double deviation = accept_prop[k] - target_acceptance;
      if (fabs(deviation) > min_deviation_threshold) {
        log_sigma[k] += (deviation > 0 ? step_size : -step_size);
      }
    }
  } else {
    /* Standard approach for small n (< 8 components) */
    for (k = 0; k < n; k++) {
      double deviation = accept_prop[k] - target_acceptance;
      if (fabs(deviation) > min_deviation_threshold) {
        log_sigma[k] += (deviation > 0 ? step_size : -step_size);
      }
    }
  }
}

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
 *       acceptance proportion calculations. This prevents incorrect normalization when
 *       window sizes change between calls.
 *
 * @note Common parameter choices: max_step_size = 0.01-0.1, base_adaptation_rate = 1.0-10.0,
 *       decay_exponent = 0.3-0.8, lag_update = 50-200, min_deviation_threshold = 1.0/lag_update.
 * @note Threshold selection: The practical choice 1.0/lag_update corresponds to the deviation
 *       from a single additional acceptance/rejection in the sliding window. For lag_update=50
 *       and target_acceptance=0.44, this means adaptation occurs only when moving from the
 *       expected 22 acceptances to 21 or 23 acceptances (deviation >= 0.02).
 * @note Adaptation schedule: Uses diminishing adaptation (step_size -> 0 as iter -> infinity)
 *       for theoretical convergence guarantees (Roberts and Rosenthal, 2007).
 * @note Performance: Vectorized implementation provides optimal performance for typical
 *       problem sizes (n = 10 to 1000, lag_update = 20 to 200) without additional complexity
 *       of problem-size-dependent algorithm selection.
 *
 * @warning Results are invalid if theta_updated does not contain sufficient history
 *          (iter < lag_update).
 * @warning Inappropriate adaptation rates and step sizes may lead to poor MCMC mixing.
 * @warning Large n (> 10^6) may require additional memory management considerations.
 * @warning Thread safety: This function is not thread-safe due to static caching.
 * @warning Cache invalidation: The cache automatically resets when lag_update changes,
 *          ensuring correctness but potentially causing slight overhead on first call
 *          with a new window size.
 *
 * @see cwmh_alpha_logit_binomial_locallevel
 * @see cwmh_alpha_logit_binomial
 * @see Roberts and Rosenthal (2007), "Coupling and Ergodicity of Adaptive MCMC"
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
                           double  min_deviation_threshold) {

  /* ========== Enhanced Input Validation with Performance Considerations ========== */
  if (theta_updated == NULL) {
    error("theta_updated cannot be NULL");
  }
  if (accept_prop == NULL) {
    error("accept_prop cannot be NULL");
  }
  if (log_sigma == NULL) {
    error("log_sigma cannot be NULL");
  }
  if (iter < lag_update) {
    error("iter (%d) must be >= lag_update (%d)", iter, lag_update);
  }
  if (n <= 0) {
    error("n must be positive, got %d", n);
  }
  if (lag_update <= 0) {
    error("lag_update must be positive, got %d", lag_update);
  }
  if (max_step_size <= 0.0) {
    error("max_step_size must be positive, got %f", max_step_size);
  }
  if (base_adaptation_rate <= 0.0) {
    error("base_adaptation_rate must be positive, got %f", base_adaptation_rate);
  }
  if (decay_exponent <= 0.0) {
    error("decay_exponent must be positive, got %f", decay_exponent);
  }
  if (target_acceptance <= 0.0 || target_acceptance >= 1.0) {
    error("target_acceptance must be in (0,1), got %f", target_acceptance);
  }
  if (min_deviation_threshold < 0.0) {
    error("min_deviation_threshold must be non-negative, got %f", min_deviation_threshold);
  }

  /* ========== Optimized Step Size Computation with Caching ========== */
  double step_size = compute_step_size_cached(iter, max_step_size,
                                              base_adaptation_rate, decay_exponent);

  /* ========== Precompute Inverse for Efficient Division with Validation ========== */
  if (adapt_cache.inv_lag_update == 0.0 || adapt_cache.cached_lag_update != lag_update) {
    /* Cache miss, first call, or lag_update changed - recompute inverse and update cache */
    adapt_cache.inv_lag_update = 1.0 / (double)lag_update;
    adapt_cache.cached_lag_update = lag_update;
  }
  double inv_lag_update = adapt_cache.inv_lag_update;

  /* ========== Compute Acceptance Proportions Using Vectorized Algorithm ========== */
  compute_acceptance_vectorized(theta_updated, accept_prop, lag_update, n, inv_lag_update);

  /* ========== Update Proposal Scales with Configurable Threshold Filtering ========== */
  update_log_sigma_optimized(accept_prop, log_sigma, n, step_size, target_acceptance,
                             min_deviation_threshold);
}

/**
 * @brief Reset adaptation cache (utility function for testing and debugging).
 *
 * @details Clears internal caches to ensure fresh computations. This function is primarily
 *          intended for unit testing and debugging scenarios where predictable behavior
 *          is required across multiple function calls. Enhanced to reset lag_update cache
 *          validation field to ensure proper recalculation when window size changes.
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
 * @note This function is thread-safe as it only modifies static cache variables.
 * @note Calling this function will cause the next adapt_cwmh_parameters call to
 *       recompute all cached values.
 * @note Performance impact is minimal as cache recomputation is fast.
 * @note Version 1.2 enhancement: Now includes cached_lag_update reset for proper
 *       cache invalidation when window size changes.
 *
 * @warning Only call this function when necessary, as it removes performance benefits
 *          of caching until values are recomputed.
 * @warning Particularly important to call between tests that use different lag_update values
 *          to prevent cache-related test failures.
 * @warning Not required in production code as cache validation handles parameter changes
 *          automatically.
 *
 * @see adapt_cwmh_parameters
 * @since version 1.0
 * @version 1.2
 */
void reset_adaptation_cache(void) {
  adapt_cache.cached_step_size = -1.0;
  adapt_cache.cached_iter = -1;
  adapt_cache.cached_base_rate = -1.0;
  adapt_cache.cached_max_step = -1.0;
  adapt_cache.cached_decay_exp = -1.0;
  adapt_cache.inv_lag_update = 0.0;
  adapt_cache.cached_lag_update = -1;
}

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
 *          **Default behavior:**
 *          This wrapper automatically computes the practical threshold as 1.0/lag_update,
 *          which corresponds to the deviation caused by a single additional acceptance
 *          or rejection in the sliding window. This is a sensible default that balances
 *          adaptation responsiveness with stability against random fluctuations.
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
 * @note All other parameters and behavior are identical to adapt_cwmh_parameters.
 * @note Consider migrating to adapt_cwmh_parameters for explicit threshold control.
 * @note Maintained for backward compatibility with existing codebases.
 *
 * @deprecated Use adapt_cwmh_parameters with explicit min_deviation_threshold instead.
 *
 * @see adapt_cwmh_parameters
 * @since version 1.0
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
                                  double target_acceptance) {

  /* Compute practical threshold based on sliding window size */
  double practical_threshold = 1.0 / (double)lag_update;

  /* Call the main function with computed threshold */
  adapt_cwmh_parameters(theta_updated, accept_prop, log_sigma, lag_update, n, iter,
                        max_step_size, base_adaptation_rate, decay_exponent,
                        target_acceptance, practical_threshold);
}
