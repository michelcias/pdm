/**
 * @file mcmc_progress_bar.c
 * @brief Implementation of efficient progress bar for MCMC samplers
 * @author Michel H. Montoril
 * @date 2026-07-25
 * @version 1.1
 */

#include "mcmc_progress_bar.h"
#include <R.h>
#include <Rmath.h>
#include <stdio.h>
#include <string.h>

/**
 * @brief Compute maximum of two integers
 */
static inline int imax(int a, int b) {
  return (a > b) ? a : b;
}

/**
 * @brief Compute minimum of two integers
 */
static inline int imin(int a, int b) {
  return (a < b) ? a : b;
}

ProgressBar progress_bar_init(int total_iterations,
                              int bar_width,
                              int burnin,
                              int thinning,
                              int verbose) {
  ProgressBar pb;

  /* Zero-initialize all fields so that no member is ever read uninitialized.
   * This matters because the sampler loops evaluate "ii % pb.update_step"
   * unconditionally; an uninitialized update_step of 0 would raise SIGFPE
   * (integer division by zero). */
  memset(&pb, 0, sizeof(ProgressBar));

  /* Early exit if verbose disabled */
  if (!verbose) {
    pb.total_iterations = 0;
    pb.update_step      = 1;  /* Safe divisor; progress_bar_update() no-ops anyway */
    return pb;
  }

  /* Clamp bar width to reasonable bounds */
  pb.bar_width = imin(120, imax(10, bar_width));
  pb.total_iterations = total_iterations;
  pb.burnin = burnin;
  pb.thinning = thinning;

  /* Compute adaptive update frequency with robust handling of small iteration counts
   * Using ceil ensures we get at least one update per bar segment, and fmax ensures
   * we never get a step size of zero even for pathological cases */
  double step_size = (total_iterations - 1) / (double)pb.bar_width;
  pb.update_step = (int)fmax(1.0, ceil(step_size));

  /* Set warmup period: use 10 iterations or 1% of total, whichever is larger
   * This ensures stable time estimates while providing early feedback for long runs */
  pb.warmup_iterations = imax(10, total_iterations / 100);

  /* Pre-compute separator length for aligned output
   * Format: |[bar_width]| 100% | ETA: 00:00:00 (approximately 24-26 chars overhead) */
  int bar_line_length = pb.bar_width + 25;

  /* Calculate length of info line */
  char info_buffer[256];
  int info_length = snprintf(info_buffer, sizeof(info_buffer),
                             "  Total iterations: %d (burnin = %d, thinning = %d)",
                             total_iterations, burnin, thinning);

  /* Use maximum of the two, with small padding */
  pb.separator_length = imax(bar_line_length, info_length + 2);

  /* Record start time for duration tracking */
  pb.start_time = clock();

  return pb;
}

void progress_bar_start(const ProgressBar *pb) {
  /* Skip if verbose disabled */
  if (pb->total_iterations == 0) return;

  /* Print header section */
  Rprintf("\n");
  for (int k = 0; k < pb->separator_length; k++) Rprintf("=");
  Rprintf("\n  MCMC Sampling Progress\n");
  Rprintf("  Total iterations: %d (burnin = %d, thinning = %d)\n",
          pb->total_iterations, pb->burnin, pb->thinning);
  for (int k = 0; k < pb->separator_length; k++) Rprintf("=");
  Rprintf("\n\n");
  R_FlushConsole();
}

void progress_bar_update(const ProgressBar *pb, int current_iteration) {
  /* Skip if verbose disabled */
  if (pb->total_iterations == 0) return;

  /* Calculate progress percentage */
  double percent = 100.0 * current_iteration / (pb->total_iterations - 1);

  /* Calculate filled portion of bar */
  int filled = (int)(pb->bar_width * current_iteration / (pb->total_iterations - 1));

  /* Calculate time estimates with warmup period for stability */
  double elapsed = (double)(clock() - pb->start_time) / CLOCKS_PER_SEC;

  /* Print progress bar using carriage return for in-place update */
  Rprintf("\r |");
  for (int k = 0; k < filled; k++) Rprintf("█");
  for (int k = filled; k < pb->bar_width; k++) Rprintf("·");

  /* Display time estimate only after warmup period */
  if (current_iteration >= pb->warmup_iterations) {
    double est_total = elapsed / current_iteration * (pb->total_iterations - 1);
    double remaining = est_total - elapsed;

    int remain_hrs = (int)(remaining / 3600);
    int remain_min = (int)((remaining - remain_hrs * 3600) / 60);
    int remain_sec = (int)(remaining - remain_hrs * 3600 - remain_min * 60);

    Rprintf("| %3.0f%% | ETA: %02d:%02d:%02d   ",
            percent, remain_hrs, remain_min, remain_sec);
  } else {
    Rprintf("| %3.0f%% | Estimating...    ", percent);
  }

  R_FlushConsole();
}

void progress_bar_finish(const ProgressBar *pb, int n_draws) {
  /* Skip if verbose disabled */
  if (pb->total_iterations == 0) return;

  /* Calculate total execution time */
  double total_time = (double)(clock() - pb->start_time) / CLOCKS_PER_SEC;
  double iter_per_sec = (pb->total_iterations - 1) / total_time;

  int total_hrs = (int)(total_time / 3600);
  int total_min = (int)((total_time - total_hrs * 3600) / 60);
  int total_sec = (int)(total_time - total_hrs * 3600 - total_min * 60);

  /* Pre-compute summary line lengths for dynamic separator sizing */
  char time_buffer[256];
  char samples_buffer[256];

  int time_length = snprintf(time_buffer, sizeof(time_buffer),
                             "  Total time: %02d:%02d:%02d | Speed: %.1f iter/sec",
                             total_hrs, total_min, total_sec, iter_per_sec);

  int samples_length = snprintf(samples_buffer, sizeof(samples_buffer),
                                "  Samples retained (n_draws): %d", n_draws);

  /* Use maximum length for separator */
  int sep_length = pb->separator_length;
  sep_length = imax(sep_length, time_length);
  sep_length = imax(sep_length, samples_length);

  /* Print summary section */
  Rprintf("\n\n");
  for (int k = 0; k < sep_length; k++) Rprintf("=");
  Rprintf("\n  MCMC completed successfully!\n");
  Rprintf("%s\n", time_buffer);
  Rprintf("%s\n", samples_buffer);
  for (int k = 0; k < sep_length; k++) Rprintf("=");
  Rprintf("\n\n");
  R_FlushConsole();
}
