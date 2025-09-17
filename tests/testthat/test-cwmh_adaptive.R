library(testthat)

# This file contains unit tests for the adaptive MCMC helper functions
# defined in the 'src/cwmh_adaptive.c' source file.

# --- Test for adapt_cwmh_parameters function ---
test_that("adapt_cwmh_parameters C function correctly adapts parameters", {

  # R wrapper for the C test function
  test_adapt_C <- function(theta_updated, log_sigma, lag_update, n, iter,
                           max_step_size, base_adaptation_rate,
                           decay_exponent, target_acceptance) {
    .Call("_pdm_test_adapt_cwmh_parameters",
          theta_updated, log_sigma, lag_update, n, iter,
          max_step_size, base_adaptation_rate,
          decay_exponent, target_acceptance)
  }

  # --- Test Case 1: Acceptance rate is too high/low ---
  n <- 5
  lag_update <- 10
  iter <- 20

  # Create a matrix of acceptance indicators (theta_updated)
  # This is a vectorized (iter x n) matrix. We simulate the last 'lag_update' iterations.
  # Acceptance rate for component 1 = 10/10 = 1.0
  # Acceptance rate for component 2 = 5/10 = 0.5
  # Acceptance rate for component 3 = 0/10 = 0.0
  theta_history <- matrix(0, nrow = iter, ncol = n)
  theta_history[(iter - lag_update + 1):iter, 1] <- 1
  theta_history[(iter - lag_update + 1):(iter - lag_update + 5), 2] <- 1

  log_sigma_in <- rep(log(0.1), n)

  # Call the function
  # We transpose the history matrix with t() before converting to a vector.
  # This aligns the memory layout with what the C function expects (row-major).
  results <- test_adapt_C(
    theta_updated = as.double(t(theta_history)),
    log_sigma = log_sigma_in,
    lag_update = as.integer(lag_update),
    n = as.integer(n),
    iter = as.integer(iter),
    max_step_size = 0.1,
    base_adaptation_rate = 1.0,
    decay_exponent = 0.5,
    target_acceptance = 0.44
  )

  # Test 1: Check calculated acceptance rates
  expected_accept_prop <- c(1.0, 0.5, 0.0, 0.0, 0.0)
  expect_equal(results$accept_prop, expected_accept_prop)

  # Test 2: Check log_sigma update direction and magnitude
  step_size <- min(0.1, 1.0 / (iter^0.5))

  # Component 1: accept_prop (1.0) > target (0.44) -> log_sigma should increase
  expect_true(results$log_sigma[1] > log_sigma_in[1])
  expect_equal(results$log_sigma[1], log_sigma_in[1] + step_size)

  # Component 2: accept_prop (0.5) > target (0.44) -> log_sigma should increase
  expect_true(results$log_sigma[2] > log_sigma_in[2])
  expect_equal(results$log_sigma[2], log_sigma_in[2] + step_size)

  # Component 3: accept_prop (0.0) < target (0.44) -> log_sigma should decrease
  expect_true(results$log_sigma[3] < log_sigma_in[3])
  expect_equal(results$log_sigma[3], log_sigma_in[3] - step_size)

})

