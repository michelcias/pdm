library(testthat)

# This file contains unit tests for the precision sampling functions
# defined in the 'src/conditional_precision.c' source file.

# --- Test for generate_precision_data ---
test_that("generate_precision_data samples from the correct posterior", {

  # R wrapper for the C test function
  test_C_precision_data <- function(y, theta_1, nu_y, eta_y) {
    .Call("_pdm_test_generate_precision_data", y, theta_1, nu_y, eta_y)
  }

  # Define inputs
  y <- c(10.1, 12.3, 11.8, 13.0, 12.5)
  theta_1 <- c(10.5, 12.0, 11.5, 12.8, 12.2)
  nu_y <- 1.0
  eta_y <- 1.0
  n <- length(y)

  # Manually calculate posterior parameters in R
  ss_y <- sum((y - theta_1)^2)
  nu_y_post <- nu_y + n / 2.0
  eta_y_post <- eta_y + ss_y / 2.0

  # Compare C and R random draws with the same seed
  set.seed(123)
  result_C <- test_C_precision_data(y, theta_1, nu_y, eta_y)

  set.seed(123)
  # Note: R's rgamma uses rate, while C's rgamma uses scale = 1/rate.
  # The C function generate_precision_data already handles this conversion.
  result_R <- rgamma(1, shape = nu_y_post, rate = eta_y_post)

  expect_equal(result_C, result_R)
})

# --- Test for generate_precision_theta_k ---
test_that("generate_precision_theta_k samples from the correct posterior", {

  # R wrapper for the C test function
  test_C_precision_theta_k <- function(theta_0k, theta_0kp1, theta_k, theta_kp1, nu_0k, eta_0k) {
    .Call("_pdm_test_generate_precision_theta_k",
          theta_0k, theta_0kp1, theta_k, theta_kp1, nu_0k, eta_0k)
  }

  # Define inputs
  theta_0k <- 2.0
  theta_0kp1 <- 0.1
  theta_k <- c(2.2, 2.4, 2.5, 2.8, 3.0)
  theta_kp1 <- c(0.12, 0.15, 0.16, 0.18, 0.20)
  nu_0k <- 2.0
  eta_0k <- 2.0
  n <- length(theta_k)

  # Manually calculate posterior parameters in R
  u_sq <- numeric(n)
  u_sq[1] <- (theta_k[1] - theta_0k - theta_0kp1)^2
  for(j in 2:n) {
    u_sq[j] <- (theta_k[j] - theta_k[j-1] - theta_kp1[j-1])^2
  }
  ss_theta <- sum(u_sq)
  nu_0k_post <- nu_0k + n / 2.0
  eta_0k_post <- eta_0k + ss_theta / 2.0

  # Compare C and R random draws with the same seed
  set.seed(456)
  result_C <- test_C_precision_theta_k(theta_0k, theta_0kp1, theta_k, theta_kp1, nu_0k, eta_0k)

  set.seed(456)
  result_R <- rgamma(1, shape = nu_0k_post, rate = eta_0k_post)

  expect_equal(result_C, result_R)
})

# --- Test for generate_precision_theta_p ---
test_that("generate_precision_theta_p samples from the correct posterior", {

  # R wrapper for the C test function
  test_C_precision_theta_p <- function(theta_0p, theta_p, nu_0p, eta_0p) {
    .Call("_pdm_test_generate_precision_theta_p",
          theta_0p, theta_p, nu_0p, eta_0p)
  }

  # Define inputs
  theta_0p <- 0.1
  theta_p <- c(0.12, 0.15, 0.16, 0.18, 0.20)
  nu_0p <- 3.0
  eta_0p <- 3.0
  n <- length(theta_p)

  # Manually calculate posterior parameters in R
  u_sq <- numeric(n)
  u_sq[1] <- (theta_p[1] - theta_0p)^2
  for(j in 2:n) {
    u_sq[j] <- (theta_p[j] - theta_p[j-1])^2
  }
  ss_theta <- sum(u_sq)
  nu_0p_post <- nu_0p + n / 2.0
  eta_0p_post <- eta_0p + ss_theta / 2.0

  # Compare C and R random draws with the same seed
  set.seed(789)
  result_C <- test_C_precision_theta_p(theta_0p, theta_p, nu_0p, eta_0p)

  set.seed(789)
  result_R <- rgamma(1, shape = nu_0p_post, rate = eta_0p_post)

  expect_equal(result_C, result_R)
})

# --- Test for generate_halft_aux (Half-t / Half-Cauchy auxiliary variable) ---
test_that("generate_halft_aux samples the auxiliary from the correct Gamma", {

  # R wrapper for the C test function
  test_C_halft_aux <- function(prec, hc_scale, df) {
    .Call("_pdm_test_generate_halft_aux", prec, hc_scale, df)
  }

  prec     <- 2.5
  hc_scale <- 3.0
  df       <- 4.0

  # Full conditional of b = 1/a: Gamma((df + 1)/2, rate = df * prec + 1/A^2).
  shape_post <- (df + 1) / 2
  rate_post  <- df * prec + 1 / hc_scale^2

  set.seed(2024)
  result_C <- test_C_halft_aux(prec, hc_scale, df)

  set.seed(2024)
  result_R <- rgamma(1, shape = shape_post, rate = rate_post)

  expect_equal(result_C, result_R)
})

test_that("generate_halft_aux reduces to the Half-Cauchy case at df = 1", {

  test_C_halft_aux <- function(prec, hc_scale, df) {
    .Call("_pdm_test_generate_halft_aux", prec, hc_scale, df)
  }

  prec     <- 1.7
  hc_scale <- 2.0

  # df = 1: b ~ Gamma(1, rate = prec + 1/A^2) (equivalently Exponential(rate)).
  rate_post <- prec + 1 / hc_scale^2

  set.seed(99)
  result_C <- test_C_halft_aux(prec, hc_scale, 1.0)

  set.seed(99)
  result_R <- rgamma(1, shape = 1, rate = rate_post)

  expect_equal(result_C, result_R)
})
