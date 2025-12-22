library(testthat)

# This file contains unit tests for the Poisson state sampling functions
# defined in 'src/generate_alpha_poisson.c' and 'src/cwmh_poisson.c'.
# These are integration tests as they also involve the adaptive layer.

#==============================================================================
# CWMH WRAPPERS - LOG-POISSON
#==============================================================================

test_that("cwmh_alpha_log_poisson_locallevel runs and is reproducible", {

  # Reset adaptation cache
  .Call("_pdm_reset_adaptation_cache")

  # R wrapper for the C test function
  test_C <- function(theta_1_in, theta_01_in, prec_theta1_in, y, log_sigma_in) {
    .Call("_pdm_test_cwmh_alpha_log_poisson_locallevel",
          theta_1_in, theta_01_in, prec_theta1_in, y, log_sigma_in)
  }

  # Define inputs
  n <- 5
  theta_1_in <- c(0.5, 0.6, 0.4, 0.7, 0.5)
  theta_01_in <- 0.5
  prec_theta1_in <- 50.0
  y <- c(2, 3, 1, 4, 2)  # Poisson counts
  log_sigma_in <- rep(log(0.1), n)

  # Test 1: Check output structure and types
  set.seed(601)
  result <- test_C(theta_1_in, theta_01_in, prec_theta1_in, y, log_sigma_in)

  expect_true(is.list(result))
  expect_equal(names(result), c("theta_1", "alpha"))
  expect_true(is.numeric(result$theta_1))
  expect_true(is.numeric(result$alpha))
  expect_equal(length(result$theta_1), n)
  expect_equal(length(result$alpha), n)

  # Test 2: Verify reproducibility
  .Call("_pdm_reset_adaptation_cache")  # Reset before reproducibility test
  set.seed(601)
  result1 <- test_C(theta_1_in, theta_01_in, prec_theta1_in, y, log_sigma_in)

  .Call("_pdm_reset_adaptation_cache")  # Reset before reproducibility test
  set.seed(601)
  result2 <- test_C(theta_1_in, theta_01_in, prec_theta1_in, y, log_sigma_in)
  expect_equal(result1, result2)

  # Test 3: Check basic properties (rates must be positive)
  expect_true(all(result$alpha > .Machine$double.eps))
})


test_that("cwmh_alpha_log_poisson (local trend) runs and is reproducible", {

  # Reset adaptation cache
  .Call("_pdm_reset_adaptation_cache")

  # R wrapper
  test_C <- function(theta_1_in, theta_2_in, theta_01_in, theta_02_in, prec_theta1_in, y) {
    .Call("_pdm_test_cwmh_alpha_log_poisson",
          theta_1_in, theta_2_in, theta_01_in, theta_02_in, prec_theta1_in, y)
  }

  # Define inputs
  n <- 5
  theta_1_in <- c(0.5, 0.6, 0.4, 0.7, 0.5)
  theta_2_in <- c(0.05, 0.06, -0.02, 0.04, 0.05)
  theta_01_in <- 0.5
  theta_02_in <- 0.05
  prec_theta1_in <- 50.0
  y <- c(2, 3, 1, 4, 2)  # Poisson counts

  # Test 1: Check output structure and types
  set.seed(602)
  result <- test_C(theta_1_in, theta_2_in, theta_01_in, theta_02_in, prec_theta1_in, y)

  expect_true(is.list(result))
  expect_equal(names(result), c("theta_1", "alpha"))
  expect_true(is.numeric(result$theta_1))
  expect_true(is.numeric(result$alpha))
  expect_equal(length(result$theta_1), n)
  expect_equal(length(result$alpha), n)

  # Test 2: Verify reproducibility
  .Call("_pdm_reset_adaptation_cache")
  set.seed(602)
  result1 <- test_C(theta_1_in, theta_2_in, theta_01_in, theta_02_in, prec_theta1_in, y)

  # Test 2: Verify reproducibility
  .Call("_pdm_reset_adaptation_cache")
  set.seed(602)
  result2 <- test_C(theta_1_in, theta_2_in, theta_01_in, theta_02_in, prec_theta1_in, y)
  expect_equal(result1, result2)

  # Test 3: Check basic properties (rates must be positive)
  expect_true(all(result$alpha > 0))
})


#==============================================================================
# ALPHA GENERATION WRAPPERS - LOG-POISSON
#==============================================================================

test_that("generate_alpha_log_poisson_locallevel runs and is reproducible", {

  # R wrapper for the C test function
  test_C <- function(theta_1_in, theta_01_in, prec_theta1_in, y) {
    .Call("_pdm_test_generate_alpha_log_poisson_locallevel",
          theta_1_in, theta_01_in, prec_theta1_in, y)
  }

  # Define inputs
  n <- 5
  theta_1_in <- c(0.5, 0.6, 0.4, 0.7, 0.5)
  theta_01_in <- 0.5
  prec_theta1_in <- 50.0
  y <- c(2, 3, 1, 4, 2)  # Poisson counts

  # Test 1: Check output structure and types
  set.seed(603)
  result <- test_C(theta_1_in, theta_01_in, prec_theta1_in, y)

  expect_true(is.list(result))
  expect_equal(names(result), c("theta_1", "alpha"))
  expect_true(is.numeric(result$theta_1))
  expect_true(is.numeric(result$alpha))
  expect_equal(length(result$theta_1), n)

  # Test 2: Verify reproducibility
  set.seed(603)
  result1 <- test_C(theta_1_in, theta_01_in, prec_theta1_in, y)
  set.seed(603)
  result2 <- test_C(theta_1_in, theta_01_in, prec_theta1_in, y)
  expect_equal(result1, result2)

  # Test 3: Check basic properties
  expect_true(all(result$alpha > 0))
})


test_that("generate_alpha_log_poisson (local trend) runs and is reproducible", {

  # R wrapper
  test_C <- function(theta_1_in, theta_2_in, theta_01_in, theta_02_in, prec_theta1_in, y) {
    .Call("_pdm_test_generate_alpha_log_poisson",
          theta_1_in, theta_2_in, theta_01_in, theta_02_in, prec_theta1_in, y)
  }

  # Define inputs
  n <- 5
  theta_1_in <- c(0.5, 0.6, 0.4, 0.7, 0.5)
  theta_2_in <- c(0.05, 0.06, -0.02, 0.04, 0.05)
  theta_01_in <- 0.5
  theta_02_in <- 0.05
  prec_theta1_in <- 50.0
  y <- c(2, 3, 1, 4, 2)  # Poisson counts

  # Test 1: Check output structure and types
  set.seed(604)
  result <- test_C(theta_1_in, theta_2_in, theta_01_in, theta_02_in, prec_theta1_in, y)

  expect_true(is.list(result))
  expect_equal(names(result), c("theta_1", "alpha"))
  expect_true(is.numeric(result$theta_1))
  expect_true(is.numeric(result$alpha))
  expect_equal(length(result$theta_1), n)

  # Test 2: Verify reproducibility
  set.seed(604)
  result1 <- test_C(theta_1_in, theta_2_in, theta_01_in, theta_02_in, prec_theta1_in, y)
  set.seed(604)
  result2 <- test_C(theta_1_in, theta_2_in, theta_01_in, theta_02_in, prec_theta1_in, y)
  expect_equal(result1, result2)

  # Test 3: Check basic properties
  expect_true(all(result$alpha > 0))
})


#==============================================================================
# COMPLETE MCMC SAMPLER WITH PARAMETER FIXING
#==============================================================================

test_that("mcmc_log_poisson_locallevel sampler is conditionally correct", {

  skip_on_cran()

  # --- 1. Simulation Setup ---
  set.seed(701)
  n <- 200

  # True parameters for simulation
  theta_01_true <- 0.7
  prec_theta1_true <- 80.0

  # Simulate Poisson data with log link
  u1 <- rnorm(n, sd = sqrt(1 / prec_theta1_true))
  theta_1_true <- cumsum(c(theta_01_true, u1))[-1]
  alpha_true <- exp(theta_1_true)  # log link: lambda = exp(theta)
  y <- rpois(n, lambda = alpha_true)  # Poisson counts

  # --- 2. R Wrapper for the Test Sampler ---
  test_sampler <- function(y, burnin, n_chain,
                           theta_1_true = NULL, theta_01_true = NULL, prec_theta1_true = NULL,
                           prior_theta01_mean = 0.0, prior_theta01_prec = 1.0,
                           prior_prec1_shape = 1.0, prior_prec1_rate = 1.0,
                           lag_update = 50L, max_step_size = 0.1,
                           base_adaptation_rate = 1.0, decay_exponent = 0.5,
                           target_acceptance = 0.44) {

    .Call("_pdm_test_mcmc_log_poisson_locallevel_fixed_params",
           as.numeric(y),
           as.integer(burnin),
           as.integer(1L),
           as.integer(n_chain),
           if(is.null(theta_1_true)) NULL else as.numeric(theta_1_true),
           if(is.null(theta_01_true)) NULL else as.numeric(theta_01_true),
           if(is.null(prec_theta1_true)) NULL else as.numeric(prec_theta1_true),
           as.numeric(prior_theta01_mean),
           as.numeric(prior_theta01_prec),
           as.numeric(prior_prec1_shape),
           as.numeric(prior_prec1_rate),
           as.integer(lag_update),
           as.numeric(max_step_size),
           as.numeric(base_adaptation_rate),
           as.numeric(decay_exponent),
           as.numeric(target_acceptance),
           as.logical(FALSE),
           as.logical(FALSE))
  }

  # --- 3. Run Tests for Each Conditional ---

  # Test A: Sample theta_01, fixing theta_1 and prec_theta1
  set.seed(702)
  mcmc_out_A <- test_sampler(
    y,
    burnin = 1000,
    n_chain = 2000,
    theta_1_true = theta_1_true,
    prec_theta1_true = prec_theta1_true,
    prior_theta01_mean = 0.0,
    prior_theta01_prec = 1.0
  )

  # Check if the posterior mean of theta_01 is close to the true value
  posterior_mean_A <- mean(mcmc_out_A$theta_01)
  expect_lt(abs(posterior_mean_A - theta_01_true), 0.15,
            label = "Posterior mean for theta_01 should be close to true value.")

  # Test B: Sample prec_theta1, fixing theta_1 and theta_01
  set.seed(703)
  mcmc_out_B <- test_sampler(
    y,
    burnin = 1000,
    n_chain = 2000,
    theta_1_true = theta_1_true,
    theta_01_true = theta_01_true,
    prior_prec1_shape = 80.0,
    prior_prec1_rate = 1.0
  )

  # Check if the posterior mean of prec_theta1 is close to the true value
  posterior_mean_B <- mean(mcmc_out_B$prec_theta1)
  expect_lt(abs(posterior_mean_B - prec_theta1_true), 15,
            label = "Posterior mean for prec_theta1 should be close to true value.")

  # Test C: Sample theta_1, fixing theta_01 and prec_theta1
  set.seed(704)
  mcmc_out_C <- test_sampler(
    y,
    burnin = 1000,
    n_chain = 1000,
    theta_01_true = theta_01_true,
    prec_theta1_true = prec_theta1_true
  )

  # Check if the posterior estimates for theta_1 are reasonable
  theta_1_median <- apply(mcmc_out_C$theta_1, 2, median)
  rmse <- sqrt(mean((theta_1_median - theta_1_true)^2))
  expect_lt(rmse, 0.5,
            label = "RMSE for theta_1 should be reasonably small.")
})
