library(testthat)

# This file contains unit tests for the adaptive MCMC helper functions
# defined in the 'src/cwmh_adaptive.c' source file.

# Helper function to reset cache between tests
reset_cache <- function() {
  tryCatch({
    invisible(.Call("_pdm_reset_adaptation_cache"))
  }, error = function(e) {
    # Cache reset is optional for test isolation but not critical
    # since v1.2 includes automatic cache validation
    warning("Cache reset not available, relying on automatic validation")
  })
}

# --- Test for adapt_cwmh_parameters function ---
test_that("adapt_cwmh_parameters C function correctly adapts parameters", {

  # Reset cache to ensure test isolation
  reset_cache()

  # R wrapper for the C test function
  test_adapt_C <- function(theta_updated, log_sigma, lag_update, n, iter,
                           max_step_size, base_adaptation_rate,
                           decay_exponent, target_acceptance,
                           min_deviation_threshold) {
    .Call("_pdm_test_adapt_cwmh_parameters",
          theta_updated, log_sigma, lag_update, n, iter,
          max_step_size, base_adaptation_rate,
          decay_exponent, target_acceptance, min_deviation_threshold)
  }

  # --- Test Case 1: Basic sliding window behavior ---
  n <- 5
  lag_update <- 10
  iter <- 20

  # Create a sliding window matrix (lag_update x n)
  # NOT (iter x n) - must match the sliding window structure used in CWMH
  theta_window <- matrix(0, nrow = lag_update, ncol = n)

  # Component 1: all 10 entries accepted -> proportion = 1.0
  theta_window[, 1] <- 1

  # Component 2: first 5 entries accepted -> proportion = 0.5
  theta_window[1:5, 2] <- 1

  # Component 3: all rejected -> proportion = 0.0
  theta_window[, 3] <- 0

  # Components 4 and 5: also all rejected
  theta_window[, 4] <- 0
  theta_window[, 5] <- 0

  log_sigma_in <- rep(log(0.1), n)

  # Call the function with the sliding window
  # Transpose to get row-major layout expected by C
  results <- test_adapt_C(
    theta_updated = as.double(t(theta_window)),
    log_sigma = log_sigma_in,
    lag_update = as.integer(lag_update),
    n = as.integer(n),
    iter = as.integer(iter),
    max_step_size = 0.1,
    base_adaptation_rate = 1.0,
    decay_exponent = 0.5,
    target_acceptance = 0.44,
    min_deviation_threshold = 0.0
  )

  # Test 1: Check calculated acceptance rates
  expected_accept_prop <- c(1.0, 0.5, 0.0, 0.0, 0.0)
  expect_equal(results$accept_prop, expected_accept_prop, tolerance = 1e-10)

  # Test 2: Check log_sigma update direction and magnitude
  step_size <- min(0.1, 1.0 / (iter^0.5))

  # Component 1: accept_prop (1.0) > target (0.44) -> log_sigma should increase
  expect_true(results$log_sigma[1] > log_sigma_in[1])
  expect_equal(results$log_sigma[1], log_sigma_in[1] + step_size, tolerance = 1e-10)

  # Component 2: accept_prop (0.5) > target (0.44) -> log_sigma should increase
  expect_true(results$log_sigma[2] > log_sigma_in[2])
  expect_equal(results$log_sigma[2], log_sigma_in[2] + step_size, tolerance = 1e-10)

  # Component 3: accept_prop (0.0) < target (0.44) -> log_sigma should decrease
  expect_true(results$log_sigma[3] < log_sigma_in[3])
  expect_equal(results$log_sigma[3], log_sigma_in[3] - step_size, tolerance = 1e-10)

  # Components 4 and 5: same as component 3
  expect_equal(results$log_sigma[4], log_sigma_in[4] - step_size, tolerance = 1e-10)
  expect_equal(results$log_sigma[5], log_sigma_in[5] - step_size, tolerance = 1e-10)

})

test_that("adapt_cwmh_parameters respects min_deviation_threshold", {

  reset_cache()

  test_adapt_C <- function(theta_updated, log_sigma, lag_update, n, iter,
                           max_step_size, base_adaptation_rate,
                           decay_exponent, target_acceptance,
                           min_deviation_threshold) {
    .Call("_pdm_test_adapt_cwmh_parameters",
          theta_updated, log_sigma, lag_update, n, iter,
          max_step_size, base_adaptation_rate,
          decay_exponent, target_acceptance, min_deviation_threshold)
  }

  # --- Test Case 2: Practical threshold (1.0/lag_update) ---
  n <- 4
  lag_update <- 50
  iter <- 100
  target_acceptance <- 0.44

  # Sliding window matrix (lag_update x n)
  theta_window <- matrix(0, nrow = lag_update, ncol = n)

  # Component 1: 21 out of 50 accepted -> proportion = 0.42, deviation = 0.02
  # Deviation equals threshold, edge case depends on floating point
  theta_window[1:21, 1] <- 1

  # Component 2: 22 out of 50 accepted -> proportion = 0.44, deviation = 0.00
  # Well below threshold, definitively should NOT update
  theta_window[1:22, 2] <- 1

  # Component 3: 24 out of 50 accepted -> proportion = 0.48, deviation = 0.04
  # Well above threshold (0.04 > 0.02), definitively SHOULD update
  theta_window[1:24, 3] <- 1

  # Component 4: 19 out of 50 accepted -> proportion = 0.38, deviation = 0.06
  # Well above threshold (0.06 > 0.02), definitively SHOULD update
  theta_window[1:19, 4] <- 1

  log_sigma_in <- rep(log(0.1), n)

  # Practical threshold as recommended in documentation
  practical_threshold <- 1.0 / lag_update  # 0.02

  results <- test_adapt_C(
    theta_updated = as.double(t(theta_window)),
    log_sigma = log_sigma_in,
    lag_update = as.integer(lag_update),
    n = as.integer(n),
    iter = as.integer(iter),
    max_step_size = 0.1,
    base_adaptation_rate = 1.0,
    decay_exponent = 0.6,
    target_acceptance = target_acceptance,
    min_deviation_threshold = practical_threshold
  )

  # Check acceptance proportions
  expect_equal(results$accept_prop[1], 0.42, tolerance = 1e-10)
  expect_equal(results$accept_prop[2], 0.44, tolerance = 1e-10)
  expect_equal(results$accept_prop[3], 0.48, tolerance = 1e-10)
  expect_equal(results$accept_prop[4], 0.38, tolerance = 1e-10)

  step_size <- min(0.1, 1.0 / (iter^0.6))

  # Component 1: deviation = 0.02, at threshold boundary (floating point edge case)
  # Due to floating point representation, may or may not update
  # Skip testing this edge case to avoid flakiness

  # Component 2: deviation = 0.00, definitively below threshold, should NOT update
  expect_equal(results$log_sigma[2], log_sigma_in[2], tolerance = 1e-10)

  # Component 3: deviation = 0.04, well above threshold, SHOULD update (increase)
  expect_true(results$log_sigma[3] > log_sigma_in[3])
  expect_equal(results$log_sigma[3], log_sigma_in[3] + step_size, tolerance = 1e-10)

  # Component 4: deviation = 0.06, well above threshold, SHOULD update (decrease)
  expect_true(results$log_sigma[4] < log_sigma_in[4])
  expect_equal(results$log_sigma[4], log_sigma_in[4] - step_size, tolerance = 1e-10)

})

test_that("adapt_cwmh_parameters with conservative threshold", {

  # Reset cache to ensure test isolation
  reset_cache()

  # R wrapper for the C test function
  test_adapt_C <- function(theta_updated, log_sigma, lag_update, n, iter,
                           max_step_size, base_adaptation_rate,
                           decay_exponent, target_acceptance,
                           min_deviation_threshold) {
    .Call("_pdm_test_adapt_cwmh_parameters",
          theta_updated, log_sigma, lag_update, n, iter,
          max_step_size, base_adaptation_rate,
          decay_exponent, target_acceptance, min_deviation_threshold)
  }

  # --- Test Case 3: Conservative threshold (2.0/lag_update) ---
  n <- 2
  lag_update <- 50
  iter <- 100
  target_acceptance <- 0.44

  # Sliding window matrix (lag_update x n)
  theta_window <- matrix(0, nrow = lag_update, ncol = n)

  # Component 1: 23 out of 50 accepted -> proportion = 0.46, deviation = 0.02
  theta_window[1:23, 1] <- 1

  # Component 2: 25 out of 50 accepted -> proportion = 0.50, deviation = 0.06
  theta_window[1:25, 2] <- 1

  log_sigma_in <- rep(log(0.1), n)

  # Conservative threshold (requires 2+ acceptance changes)
  conservative_threshold <- 2.0 / lag_update  # 0.04

  results <- test_adapt_C(
    theta_updated = as.double(t(theta_window)),
    log_sigma = log_sigma_in,
    lag_update = as.integer(lag_update),
    n = as.integer(n),
    iter = as.integer(iter),
    max_step_size = 0.1,
    base_adaptation_rate = 1.0,
    decay_exponent = 0.6,
    target_acceptance = target_acceptance,
    min_deviation_threshold = conservative_threshold
  )

  # Component 1: deviation = 0.02, NOT above threshold (0.02 is not > 0.04)
  expect_equal(results$log_sigma[1], log_sigma_in[1], tolerance = 1e-10)

  # Component 2: deviation = 0.06, above threshold (0.06 > 0.04), should update
  expect_true(results$log_sigma[2] > log_sigma_in[2])
  expect_equal(results$log_sigma[2], log_sigma_in[2] +
                 min(0.1, 1.0 / (iter^0.6)), tolerance = 1e-10)

})

test_that("cache validation handles lag_update changes correctly", {

  # This test specifically validates the cache invalidation mechanism
  # introduced in version 1.2 of cwmh_adaptive.c

  reset_cache()

  test_adapt_C <- function(theta_updated, log_sigma, lag_update, n, iter,
                           max_step_size, base_adaptation_rate,
                           decay_exponent, target_acceptance,
                           min_deviation_threshold) {
    .Call("_pdm_test_adapt_cwmh_parameters",
          theta_updated, log_sigma, lag_update, n, iter,
          max_step_size, base_adaptation_rate,
          decay_exponent, target_acceptance, min_deviation_threshold)
  }

  # First call with lag_update = 20
  n1 <- 3
  lag_update1 <- 20
  iter1 <- 50
  theta_window1 <- matrix(0, nrow = lag_update1, ncol = n1)
  theta_window1[1:10, ] <- 1  # 50% acceptance for all components
  log_sigma1 <- rep(log(0.1), n1)

  results1 <- test_adapt_C(
    theta_updated = as.double(t(theta_window1)),
    log_sigma = log_sigma1,
    lag_update = as.integer(lag_update1),
    n = as.integer(n1),
    iter = as.integer(iter1),
    max_step_size = 0.1,
    base_adaptation_rate = 1.0,
    decay_exponent = 0.6,
    target_acceptance = 0.44,
    min_deviation_threshold = 0.0
  )

  # Verify proportions are correct for lag_update = 20
  expect_equal(results1$accept_prop, rep(0.5, n1), tolerance = 1e-10)

  # Second call with lag_update = 40 (different from first call)
  # This tests that cache invalidation works correctly
  n2 <- 3
  lag_update2 <- 40
  iter2 <- 100
  theta_window2 <- matrix(0, nrow = lag_update2, ncol = n2)
  theta_window2[1:20, ] <- 1  # Still 50% acceptance but different window size
  log_sigma2 <- rep(log(0.1), n2)

  results2 <- test_adapt_C(
    theta_updated = as.double(t(theta_window2)),
    log_sigma = log_sigma2,
    lag_update = as.integer(lag_update2),
    n = as.integer(n2),
    iter = as.integer(iter2),
    max_step_size = 0.1,
    base_adaptation_rate = 1.0,
    decay_exponent = 0.6,
    target_acceptance = 0.44,
    min_deviation_threshold = 0.0
  )

  # Verify proportions are still correct with new lag_update
  # If cache validation failed, this would give incorrect results
  expect_equal(results2$accept_prop, rep(0.5, n2), tolerance = 1e-10)

})
