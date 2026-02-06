library(testthat)

# This file contains unit tests for the core CWMH sampling functions
# defined in 'src/cwmh_binomial.c'.

test_that("cwmh_alpha_logit_binomial_locallevel runs and is reproducible", {

  # R wrapper for the C test function
  test_C <- function(theta_1_in, theta_01_in, prec_theta1_in, y, n_trials) {
    .Call("_bdm_test_cwmh_alpha_logit_binomial_locallevel",
          theta_1_in, theta_01_in, prec_theta1_in, y, n_trials,
          rep(log(0.1), length(y)))
  }

  # Define inputs
  n <- 5
  theta_1_in <- c(0.1, 0.2, 0.1, 0.3, 0.2)
  theta_01_in <- 0.05
  prec_theta1_in <- 100.0
  y <- c(12, 15, 14, 16, 15)
  n_trials <- 20

  # Test 1: Check output structure and types
  set.seed(301)
  result <- test_C(theta_1_in, theta_01_in, prec_theta1_in, y, n_trials)

  expect_true(is.list(result))
  expect_equal(names(result), c("theta_1", "alpha"))
  expect_true(is.numeric(result$theta_1))
  expect_true(is.numeric(result$alpha))
  expect_equal(length(result$theta_1), n)

  # Test 2: Verify reproducibility
  set.seed(301)
  result1 <- test_C(theta_1_in, theta_01_in, prec_theta1_in, y, n_trials)
  set.seed(301)
  result2 <- test_C(theta_1_in, theta_01_in, prec_theta1_in, y, n_trials)
  expect_equal(result1, result2)

  # Test 3: Check basic properties
  expect_true(all(result$alpha >= 0 & result$alpha <= 1))
})


test_that("cwmh_alpha_logit_binomial (local trend) runs and is reproducible", {

  # R wrapper
  test_C <- function(theta_1_in, theta_2_in, theta_01_in, theta_02_in, prec_theta1_in, y, n_trials) {
    .Call("_bdm_test_cwmh_alpha_logit_binomial",
          theta_1_in, theta_2_in, theta_01_in, theta_02_in, prec_theta1_in, y, n_trials)
  }

  # Define inputs
  n <- 5
  theta_1_in <- c(0.1, 0.2, 0.1, 0.3, 0.2)
  theta_2_in <- c(0.01, 0.02, -0.01, 0.01, 0.02)
  theta_01_in <- 0.05
  theta_02_in <- 0.01
  prec_theta1_in <- 100.0
  y <- c(12, 15, 14, 16, 15)
  n_trials <- 20

  # Test 1: Check output structure and types
  set.seed(302)
  result <- test_C(theta_1_in, theta_2_in, theta_01_in, theta_02_in, prec_theta1_in, y, n_trials)

  expect_true(is.list(result))
  expect_equal(names(result), c("theta_1", "alpha"))
  expect_true(is.numeric(result$theta_1))
  expect_true(is.numeric(result$alpha))
  expect_equal(length(result$theta_1), n)

  # Test 2: Verify reproducibility
  set.seed(302)
  result1 <- test_C(theta_1_in, theta_2_in, theta_01_in, theta_02_in, prec_theta1_in, y, n_trials)
  set.seed(302)
  result2 <- test_C(theta_1_in, theta_2_in, theta_01_in, theta_02_in, prec_theta1_in, y, n_trials)
  expect_equal(result1, result2)

  # Test 3: Check basic properties
  expect_true(all(result$alpha >= 0 & result$alpha <= 1))
})

