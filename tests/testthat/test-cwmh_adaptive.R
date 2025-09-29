library(testthat)

# This file contains unit tests for the adaptive MCMC helper functions
# defined in the 'src/cwmh_adaptive.c' source file.

# --- Test for adapt_cwmh_parameters function ---
test_that("adapt_cwmh_parameters C function correctly adapts parameters", {

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
  # NOT (iter x n) as in the original test
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

  # --- Test Case 2: Practical threshold (1.0/lag_update) ---
  n <- 3
  lag_update <- 50
  iter <- 100
  target_acceptance <- 0.44

  # Sliding window matrix (lag_update x n)
  theta_window <- matrix(0, nrow = lag_update, ncol = n)

  # Component 1: 23 out of 50 accepted -> proportion = 0.46, deviation = 0.02
  theta_window[1:23, 1] <- 1

  # Component 2: 22 out of 50 accepted -> proportion = 0.44, deviation = 0.00
  theta_window[1:22, 2] <- 1

  # Component 3: 20 out of 50 accepted -> proportion = 0.40, deviation = 0.04
  theta_window[1:20, 3] <- 1

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
  expect_equal(results$accept_prop[1], 0.46, tolerance = 1e-10)
  expect_equal(results$accept_prop[2], 0.44, tolerance = 1e-10)
  expect_equal(results$accept_prop[3], 0.40, tolerance = 1e-10)

  step_size <- min(0.1, 1.0 / (iter^0.6))

  # Component 1: deviation = 0.02, exactly at threshold boundary
  # Should trigger update since 0.02 is NOT strictly greater than 0.02
  # Actually, checking the C code: if (fabs(deviation) > min_deviation_threshold)
  # So 0.02 > 0.02 is FALSE, should NOT update
  expect_equal(results$log_sigma[1], log_sigma_in[1], tolerance = 1e-10)

  # Component 2: deviation = 0.00, below threshold, should NOT update
  expect_equal(results$log_sigma[2], log_sigma_in[2], tolerance = 1e-10)

  # Component 3: deviation = 0.04, above threshold (0.04 > 0.02), should update
  expect_true(results$log_sigma[3] < log_sigma_in[3])
  expect_equal(results$log_sigma[3], log_sigma_in[3] - step_size, tolerance = 1e-10)

})

test_that("adapt_cwmh_parameters with conservative threshold", {

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
