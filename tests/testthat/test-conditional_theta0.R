library(testthat)

# This file contains unit tests for the initial state sampling functions
# defined in the 'src/conditional_theta0.c' source file.

# --- Test for generate_theta_01_locallevel ---
test_that("generate_theta_01_locallevel is reproducible", {
  # R wrapper
  test_C <- function(theta_1, prec_theta_1, mean_theta_01, prec_theta_01) {
    .Call("_pdm_test_generate_theta_01_locallevel", theta_1, prec_theta_1, mean_theta_01, prec_theta_01)
  }

  # Inputs
  theta_1 <- c(10.5, 11.0, 10.8)
  prec_theta_1 <- 5.0
  mean_theta_01 <- 10.0
  prec_theta_01 <- 1.0

  # Check reproducibility
  set.seed(101)
  res1 <- test_C(theta_1, prec_theta_1, mean_theta_01, prec_theta_01)
  set.seed(101)
  res2 <- test_C(theta_1, prec_theta_1, mean_theta_01, prec_theta_01)
  expect_equal(res1, res2)
  expect_true(is.numeric(res1))
})

# --- Test for generate_theta_01 ---
test_that("generate_theta_01 is reproducible", {
  # R wrapper
  test_C <- function(theta_1, theta_02, prec_theta_1, mean_theta_01, prec_theta_01) {
    .Call("_pdm_test_generate_theta_01", theta_1, theta_02, prec_theta_1, mean_theta_01, prec_theta_01)
  }

  # Inputs
  theta_1 <- c(10.5, 11.0, 10.8)
  theta_02 <- 0.5
  prec_theta_1 <- 5.0
  mean_theta_01 <- 10.0
  prec_theta_01 <- 1.0

  # Check reproducibility
  set.seed(102)
  res1 <- test_C(theta_1, theta_02, prec_theta_1, mean_theta_01, prec_theta_01)
  set.seed(102)
  res2 <- test_C(theta_1, theta_02, prec_theta_1, mean_theta_01, prec_theta_01)
  expect_equal(res1, res2)
  expect_true(is.numeric(res1))
})

# --- Test for generate_theta_0k ---
test_that("generate_theta_0k is reproducible", {
  # R wrapper
  test_C <- function(theta_km1, theta_k, theta_0km1, theta_0kp1, prec_km1, prec_k, mean_0k, prec_0k) {
    .Call("_pdm_test_generate_theta_0k", theta_km1, theta_k, theta_0km1, theta_0kp1, prec_km1, prec_k, mean_0k, prec_0k)
  }

  # Inputs
  theta_km1 <- c(10.5, 11.0, 10.8)
  theta_k <- c(0.5, 0.4, 0.6)
  theta_0km1 <- 10.0
  theta_0kp1 <- 0.1
  prec_km1 <- 5.0
  prec_k <- 10.0
  mean_0k <- 0.5
  prec_0k <- 2.0

  # Check reproducibility
  set.seed(103)
  res1 <- test_C(theta_km1, theta_k, theta_0km1, theta_0kp1, prec_km1, prec_k, mean_0k, prec_0k)
  set.seed(103)
  res2 <- test_C(theta_km1, theta_k, theta_0km1, theta_0kp1, prec_km1, prec_k, mean_0k, prec_0k)
  expect_equal(res1, res2)
  expect_true(is.numeric(res1))
})

# --- Test for generate_theta_0p ---
test_that("generate_theta_0p is reproducible", {
  # R wrapper
  test_C <- function(theta_pm1, theta_p, theta_0pm1, prec_pm1, prec_p, mean_0p, prec_0p) {
    .Call("_pdm_test_generate_theta_0p", theta_pm1, theta_p, theta_0pm1, prec_pm1, prec_p, mean_0p, prec_0p)
  }

  # Inputs
  theta_pm1 <- c(0.5, 0.4, 0.6)
  theta_p <- c(0.1, 0.0, -0.1)
  theta_0pm1 <- 0.5
  prec_pm1 <- 10.0
  prec_p <- 20.0
  mean_0p <- 0.1
  prec_0p <- 5.0

  # Check reproducibility
  set.seed(104)
  res1 <- test_C(theta_pm1, theta_p, theta_0pm1, prec_pm1, prec_p, mean_0p, prec_0p)
  set.seed(104)
  res2 <- test_C(theta_pm1, theta_p, theta_0pm1, prec_pm1, prec_p, mean_0p, prec_0p)
  expect_equal(res1, res2)
  expect_true(is.numeric(res1))
})

