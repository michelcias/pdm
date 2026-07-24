/**
 * @file mcmc_progress_bar.h
 * @brief Efficient progress bar interface for MCMC samplers
 * @author Michel H. Montoril
 * @date 2025-01-16
 * @version 1.0
 *
 * @details Provides a lightweight, computationally optimized progress bar system
 *          designed specifically for iterative MCMC algorithms. The implementation
 *          minimizes performance overhead through adaptive update frequency and
 *          pre-computed formatting parameters.
 *
 *          Key features:
 *          - Adaptive update frequency based on total iterations and bar width
 *          - Robust handling of edge cases (small iteration counts, large values)
 *          - Smoothed time estimation with configurable warmup period
 *          - Dynamic separator sizing for aligned output
 *          - Minimal memory footprint (single struct with scalar fields)
 *          - Unicode fallback for terminal compatibility
 *
 *          Typical overhead: < 0.01% of total execution time for standard MCMC runs
 */

#ifndef MCMC_PROGRESS_BAR_H
#define MCMC_PROGRESS_BAR_H

#include <R.h>
#include <time.h>

/**
 * @brief Progress bar state and configuration structure
 *
 * @details Encapsulates all state needed to efficiently update and display
 *          progress information. Designed for minimal memory footprint and
 *          cache-friendly access patterns.
 */
typedef struct {
  int total_iterations;      /**< Total number of iterations in the chain */
  int bar_width;             /**< Width of the progress bar in characters */
  int update_step;           /**< Frequency of updates (every N iterations) */
  int separator_length;      /**< Length of separator lines for alignment */
  int warmup_iterations;     /**< Minimum iterations before showing ETA */
  clock_t start_time;        /**< Timestamp at initialization */
  int burnin;                /**< Number of burn-in iterations (for display) */
  int thinning;              /**< Thinning interval (for display) */
} ProgressBar;

/**
 * @brief Initialize progress bar with optimal parameters
 *
 * @details Computes adaptive update frequency, validates bar width constraints,
 *          and pre-calculates separator length for aligned output. The update
 *          frequency is chosen to provide smooth visual feedback (approximately
 *          one update per bar segment) while minimizing performance impact.
 *
 *          The warmup period for time estimation is set to the maximum of 10
 *          iterations or 1% of total iterations, ensuring stable extrapolation
 *          while providing early feedback for long runs.
 *
 * @param total_iterations Total number of MCMC iterations
 * @param bar_width Desired width of progress bar (clamped to 10-120)
 * @param burnin Number of burn-in iterations (for informational display)
 * @param thinning Thinning interval (for informational display)
 * @param verbose Whether to display progress bar (0 = no, 1 = yes)
 *
 * @return Initialized ProgressBar structure, or struct with total_iterations = 0 if verbose = 0
 *
 * @note The function ensures update_step >= 1 even for very small iteration counts
 * @note Separator length is dynamically calculated to accommodate all output lines
 * @note Warmup iterations default to max(10, total_iterations / 100)
 */
ProgressBar progress_bar_init(int total_iterations,
                              int bar_width,
                              int burnin,
                              int thinning,
                              int verbose);

/**
 * @brief Display initial progress bar header
 *
 * @details Prints a formatted header section containing sampling information
 *          and visual separators. The separator length is pre-computed during
 *          initialization to ensure proper alignment with subsequent updates.
 *
 * @param pb Pointer to initialized ProgressBar structure
 *
 * @note Does nothing if pb->total_iterations == 0 (verbose disabled)
 * @note Flushes console output to ensure immediate display
 */
void progress_bar_start(const ProgressBar *pb);

/**
 * @brief Update progress bar display with current iteration status
 *
 * @details Efficiently updates the progress bar using carriage return for
 *          in-place refresh. Time estimates are computed using simple linear
 *          extrapolation after the warmup period, with robust handling of
 *          edge cases.
 *
 *          The function performs minimal computation per call:
 *          - Progress percentage calculation
 *          - Filled bar segment count
 *          - Time extrapolation (if past warmup)
 *          - Formatted output rendering
 *
 *          Updates are automatically throttled by the caller checking against
 *          update_step, so this function assumes it should display when called.
 *
 * @param pb Pointer to initialized ProgressBar structure
 * @param current_iteration Current iteration number (1-indexed)
 *
 * @note Does nothing if pb->total_iterations == 0 (verbose disabled)
 * @note Forces final update at current_iteration == total_iterations - 1
 * @note Time estimation shows "Estimating..." during warmup period
 * @note Flushes console output for immediate visual feedback
 */
void progress_bar_update(const ProgressBar *pb, int current_iteration);

/**
 * @brief Display final progress bar summary
 *
 * @details Prints completion message with total execution time, average
 *          iteration speed, and number of retained samples. The separator
 *          length dynamically adjusts to accommodate all summary information.
 *
 * @param pb Pointer to initialized ProgressBar structure
 * @param n_chain Number of retained posterior samples
 *
 * @note Does nothing if pb->total_iterations == 0 (verbose disabled)
 * @note Adds appropriate line spacing before and after summary
 * @note Flushes console output to ensure complete display
 */
void progress_bar_finish(const ProgressBar *pb, int n_chain);

#endif /* MCMC_PROGRESS_BAR_H */
