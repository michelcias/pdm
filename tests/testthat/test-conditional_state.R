library(testthat)

# This file contains unit tests for the state sampling functions
# defined in the 'src/conditional_state.c' source file.

# --- Test for generate_theta_1_locallevel ---
test_that("generate_theta_1_locallevel runs and is reproducible", {

  # R wrapper for the C test function
  test_C_theta_1_locallevel <- function(data, prec_data, prec_theta_1, theta_01) {
    .Call("_bdm_test_generate_theta_1_locallevel",
          data, prec_data, prec_theta_1, theta_01)
  }

  # Define inputs
  data <- c(10.1, 12.3, 11.8, 13.0, 12.5)
  prec_data <- 1.0
  prec_theta_1 <- 5.0
  theta_01 <- 10.0

  # Test 1: Check output type and length
  set.seed(1)
  result <- test_C_theta_1_locallevel(data, prec_data, prec_theta_1, theta_01)
  expect_true(is.numeric(result))
  expect_equal(length(result), length(data))
  expect_false(any(is.na(result)))

  # Test 2: Verify reproducibility
  set.seed(1)
  result1 <- test_C_theta_1_locallevel(data, prec_data, prec_theta_1, theta_01)
  set.seed(1)
  result2 <- test_C_theta_1_locallevel(data, prec_data, prec_theta_1, theta_01)
  expect_equal(result1, result2)
})

# --- Test for generate_theta_1 ---
test_that("generate_theta_1 runs and is reproducible", {

  # R wrapper
  test_C_theta_1 <- function(data, theta_2, prec_data, prec_theta_1, theta_01, theta_02) {
    .Call("_bdm_test_generate_theta_1",
          data, theta_2, prec_data, prec_theta_1, theta_01, theta_02)
  }

  # Define inputs
  data <- c(10.1, 12.3, 11.8, 13.0, 12.5)
  theta_2 <- c(0.5, 0.4, 0.6, 0.5, 0.5)
  prec_data <- 1.0
  prec_theta_1 <- 5.0
  theta_01 <- 10.0
  theta_02 <- 0.5

  # Test 1: Check output type and length
  set.seed(2)
  result <- test_C_theta_1(data, theta_2, prec_data, prec_theta_1, theta_01, theta_02)
  expect_true(is.numeric(result))
  expect_equal(length(result), length(data))
  expect_false(any(is.na(result)))

  # Test 2: Verify reproducibility
  set.seed(2)
  result1 <- test_C_theta_1(data, theta_2, prec_data, prec_theta_1, theta_01, theta_02)
  set.seed(2)
  result2 <- test_C_theta_1(data, theta_2, prec_data, prec_theta_1, theta_01, theta_02)
  expect_equal(result1, result2)
})

# --- Test for generate_theta_k ---
test_that("generate_theta_k runs and is reproducible", {

  # R wrapper
  test_C_theta_k <- function(theta_km1, theta_kp1, prec_km1, prec_k, theta_0k, theta_0kp1) {
    .Call("_bdm_test_generate_theta_k",
          theta_km1, theta_kp1, prec_km1, prec_k, theta_0k, theta_0kp1)
  }

  # Define inputs
  theta_km1 <- c(10.5, 12.0, 11.5, 12.8, 12.2)
  theta_kp1 <- c(0.1, 0.0, -0.1, 0.0, 0.1)
  prec_km1 <- 5.0
  prec_k <- 10.0
  theta_0k <- 0.5
  theta_0kp1 <- 0.1

  # Test 1: Check output type and length
  set.seed(3)
  result <- test_C_theta_k(theta_km1, theta_kp1, prec_km1, prec_k, theta_0k, theta_0kp1)
  expect_true(is.numeric(result))
  expect_equal(length(result), length(theta_km1))
  expect_false(any(is.na(result)))

  # Test 2: Verify reproducibility
  set.seed(3)
  result1 <- test_C_theta_k(theta_km1, theta_kp1, prec_km1, prec_k, theta_0k, theta_0kp1)
  set.seed(3)
  result2 <- test_C_theta_k(theta_km1, theta_kp1, prec_km1, prec_k, theta_0k, theta_0kp1)
  expect_equal(result1, result2)
})

# --- Test for generate_theta_p ---
test_that("generate_theta_p runs and is reproducible", {

  # R wrapper
  test_C_theta_p <- function(theta_pm1, prec_pm1, prec_p, theta_0p) {
    .Call("_bdm_test_generate_theta_p",
          theta_pm1, prec_pm1, prec_p, theta_0p)
  }

  # Define inputs
  theta_pm1 <- c(0.5, 0.4, 0.6, 0.5, 0.5)
  prec_pm1 <- 10.0
  prec_p <- 20.0
  theta_0p <- 0.1

  # Test 1: Check output type and length
  set.seed(4)
  result <- test_C_theta_p(theta_pm1, prec_pm1, prec_p, theta_0p)
  expect_true(is.numeric(result))
  expect_equal(length(result), length(theta_pm1))
  expect_false(any(is.na(result)))

  # Test 2: Verify reproducibility
  set.seed(4)
  result1 <- test_C_theta_p(theta_pm1, prec_pm1, prec_p, theta_0p)
  set.seed(4)
  result2 <- test_C_theta_p(theta_pm1, prec_pm1, prec_p, theta_0p)
  expect_equal(result1, result2)
})
