library(testthat)

# This file contains unit tests for the binomial state sampling functions
# defined in 'src/generate_alpha_binomial.c'. These are integration tests
# as they also involve the adaptive layer.

test_that("generate_alpha_logit_binomial_locallevel runs and is reproducible", {

  # R wrapper for the C test function
  test_C <- function(theta_1_in, theta_01_in, prec_theta1_in, y, n_trials) {
    .Call("_pdm_test_generate_alpha_logit_binomial_locallevel",
          theta_1_in, theta_01_in, prec_theta1_in, y, n_trials)
  }

  # Define inputs
  n <- 5
  theta_1_in <- c(0.1, 0.2, 0.1, 0.3, 0.2)
  theta_01_in <- 0.05
  prec_theta1_in <- 100.0
  y <- c(12, 15, 14, 16, 15)
  n_trials <- 20

  # Test 1: Check output structure and types
  set.seed(201)
  result <- test_C(theta_1_in, theta_01_in, prec_theta1_in, y, n_trials)

  expect_true(is.list(result))
  expect_equal(names(result), c("theta_1", "alpha"))
  expect_true(is.numeric(result$theta_1))
  expect_true(is.numeric(result$alpha))
  expect_equal(length(result$theta_1), n)

  # Test 2: Verify reproducibility
  set.seed(201)
  result1 <- test_C(theta_1_in, theta_01_in, prec_theta1_in, y, n_trials)
  set.seed(201)
  result2 <- test_C(theta_1_in, theta_01_in, prec_theta1_in, y, n_trials)
  expect_equal(result1, result2)

  # Test 3: Check basic properties
  expect_true(all(result$alpha >= 0 & result$alpha <= 1))
})

test_that("generate_alpha_logit_binomial (local trend) runs and is reproducible", {

  # R wrapper
  test_C <- function(theta_1_in, theta_2_in, theta_01_in, theta_02_in, prec_theta1_in, y, n_trials) {
    .Call("_pdm_test_generate_alpha_logit_binomial",
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
  set.seed(202)
  result <- test_C(theta_1_in, theta_2_in, theta_01_in, theta_02_in, prec_theta1_in, y, n_trials)

  expect_true(is.list(result))
  expect_equal(names(result), c("theta_1", "alpha"))
  expect_true(is.numeric(result$theta_1))
  expect_true(is.numeric(result$alpha))
  expect_equal(length(result$theta_1), n)

  # Test 2: Verify reproducibility
  set.seed(202)
  result1 <- test_C(theta_1_in, theta_2_in, theta_01_in, theta_02_in, prec_theta1_in, y, n_trials)
  set.seed(202)
  result2 <- test_C(theta_1_in, theta_2_in, theta_01_in, theta_02_in, prec_theta1_in, y, n_trials)
  expect_equal(result1, result2)

  # Test 3: Check basic properties
  expect_true(all(result$alpha >= 0 & result$alpha <= 1))
})



test_that("logit CWMH state update clamps a runaway latent state", {

  # Regression test for the logit saturation guard (clamp_logit_state /
  # LOGIT_THETA_CLAMP in src/cwmh_binomial.c), the symmetric counterpart of the
  # probit guard. From a saturated state (theta ~ 50, all-success data) the
  # likelihood is flat, so without the guard theta_1 could random-walk into the
  # tail; the guard must keep it inside the [-36, 36] band and alpha in (0, 1).
  test_C <- function(theta_1_in, theta_01_in, prec_theta1_in, y, n_trials) {
    .Call("_pdm_test_generate_alpha_logit_binomial_locallevel",
          theta_1_in, theta_01_in, prec_theta1_in, y, n_trials)
  }

  n <- 6
  theta_1_in <- c(50, 55, 60, -50, -55, -60)
  n_trials <- 10
  y <- c(10, 10, 10, 0, 0, 0)   # data agrees with the saturated states

  set.seed(921)
  res <- test_C(theta_1_in, 0.0, 5.0, y, n_trials)

  expect_true(all(is.finite(res$theta_1)))
  expect_true(all(abs(res$theta_1) <= 36 + 1e-9),
              info = "latent state must be clamped to the [-36, 36] band")
  expect_true(all(res$alpha > 0 & res$alpha < 1),
              info = "alpha must be strictly inside (0, 1)")
  expect_equal(res$alpha, plogis(res$theta_1), tolerance = 1e-12)
})
