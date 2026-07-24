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



test_that("logit CWMH alpha stays inside the guarded band for a saturated state", {

  # Regression test for the logit probability guard (clamp_link_alpha /
  # ilogit_guarded in src/cwmh_binomial.c), the counterpart of the probit guard.
  # From a saturated state (theta ~ 50, all-success data) the likelihood is flat
  # and the latent state is left unclamped, so theta_1 may stay large; the guard
  # constrains only the probability alpha = ilogit(theta_1) to stay strictly
  # inside (0, 1), within [2e-16, 1 - 2.3e-16], and equal to the clamped
  # plogis(theta_1). It also keeps ilogit finite inside the likelihood, so the
  # accept/reject step never sees an exact 0 or 1 (which would give dbinom = -Inf).
  link_alpha_min <- 2e-16
  link_alpha_max <- 1 - 2.3e-16
  clamp_link_alpha <- function(p) pmin(pmax(p, link_alpha_min), link_alpha_max)

  test_C <- function(theta_1_in, theta_01_in, prec_theta1_in, y, n_trials) {
    .Call("_pdm_test_generate_alpha_logit_binomial_locallevel",
          theta_1_in, theta_01_in, prec_theta1_in, y, n_trials)
  }

  theta_1_in <- c(50, 55, 60, -50, -55, -60)
  n_trials <- 10
  y <- c(10, 10, 10, 0, 0, 0)   # data agrees with the saturated states

  set.seed(921)
  res <- test_C(theta_1_in, 0.0, 5.0, y, n_trials)

  expect_true(all(is.finite(res$theta_1)))
  expect_true(all(res$alpha > 0 & res$alpha < 1),
              info = "alpha must be strictly inside (0, 1)")
  expect_true(all(res$alpha >= link_alpha_min & res$alpha <= link_alpha_max),
              info = "alpha must stay within [2e-16, 1 - 2.3e-16]")
  expect_equal(res$alpha, clamp_link_alpha(plogis(res$theta_1)), tolerance = 1e-12)
})
