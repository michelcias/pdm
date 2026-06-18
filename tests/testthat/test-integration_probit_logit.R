library(testthat)

# This file contains integration tests comparing probit and logit link functions
# for Bernoulli models. The goal is to verify that both implementations give
# sensible and consistent results, especially when theta values are near zero
# where probit and logit links should behave similarly.

test_that("probit and logit give similar results for theta near zero", {

  skip_on_cran()
  
  # --- Setup: Generate data with theta values close to zero ---
  set.seed(701)
  n <- 80
  
  # Create theta trajectory oscillating around zero
  theta_true <- 0.2 * sin(2 * pi * (1:n) / 20) + rnorm(n, 0, 0.1)
  
  # Generate Bernoulli data using probit link (for reference)
  alpha_probit <- pnorm(theta_true)
  y <- as.numeric(rbinom(n, 1, alpha_probit))
  
  # --- Run both samplers ---
  
  # Probit sampler
  set.seed(702)
  result_probit <- .Call("_pdm_C_MCMC_probit_bernoulli_locallevel",
                         y, 200L, 2L, 500L,
                         0.0, 1.0, 10.0, 1.0,
                         FALSE, 60L)
  
  # Logit sampler (with n_trials = 1 for Bernoulli)
  set.seed(702)
  result_logit <- .Call("_pdm_C_MCMC_logit_binomial_locallevel",
                        y, 1.0, 200L, 2L, 500L,
                        0.0, 1.0, 10.0, 1.0,
                        50L, 0.1, 1.0, 0.5, 0.44, 0.01,
                        FALSE, FALSE,
                        FALSE, 60L)
  
  # --- Compare posterior means ---
  
  # Initial state comparison
  probit_theta01_mean <- mean(result_probit$theta_01)
  logit_theta01_mean <- mean(result_logit$theta_01)
  expect_true(abs(probit_theta01_mean - logit_theta01_mean) < 0.3,
              info = "Probit and logit should give similar theta_01 posterior means.")
  
  # State trajectory comparison (using posterior means)
  probit_theta1_mean <- colMeans(result_probit$theta_1)
  logit_theta1_mean <- colMeans(result_logit$theta_1)
  
  # Calculate correlation between posterior mean trajectories
  trajectory_correlation <- cor(probit_theta1_mean, logit_theta1_mean)
  expect_true(trajectory_correlation > 0.7,
              info = "Probit and logit theta_1 trajectories should be highly correlated.")
  
  # --- Compare alpha transformations for small theta values ---
  
  # Find time points where |theta| < 1 (where links should be most similar)
  small_theta_idx <- which(abs(probit_theta1_mean) < 1)
  if (length(small_theta_idx) > 5) {
    
    probit_alpha_small <- colMeans(result_probit$alpha)[small_theta_idx]
    logit_alpha_small <- colMeans(result_logit$alpha)[small_theta_idx]
    
    mean_abs_diff_alpha <- mean(abs(probit_alpha_small - logit_alpha_small))
    expect_true(mean_abs_diff_alpha < 0.1,
                info = "For small theta, probit and logit alpha should be similar.")
  }
})

test_that("probit sampler is more efficient than logit (acceptance rate)", {

  skip_on_cran()
  
  # --- Generate test data ---
  set.seed(703)
  n <- 60
  y <- as.numeric(rbinom(n, 1, 0.3))
  
  # --- Run logit sampler with acceptance tracking ---
  set.seed(704)
  result_logit <- .Call("_pdm_C_MCMC_logit_binomial_locallevel",
                        y, 1.0, 100L, 1L, 300L,
                        0.0, 1.0, 5.0, 1.0,
                        30L, 0.1, 1.0, 0.5, 0.44, 0.01,
                        FALSE, TRUE,
                        FALSE, 60L)  # return_accept_prop = TRUE
  
  # --- Run probit sampler (Gibbs has 100% acceptance) ---
  set.seed(704)
  result_probit <- .Call("_pdm_C_MCMC_probit_bernoulli_locallevel",
                         y, 100L, 1L, 300L,
                         0.0, 1.0, 5.0, 1.0,
                         FALSE, 60L)
  
  # --- Compare efficiency ---
  
  # Logit acceptance rate should be < 1.0
  logit_accept_rate <- mean(result_logit$accept_prop)
  expect_lt(logit_accept_rate, 0.9,
            label = "Logit sampler acceptance rate")
  
  # Probit effectively has 100% acceptance (Gibbs sampling)
  # We can't directly test this, but we can verify that probit produces
  # valid output without rejection sampling
  expect_true(all(is.finite(result_probit$theta_1)),
              info = "Probit sampler should produce all finite values (no rejections).")
  
  # Both should produce similar posterior standard deviations
  probit_theta01_sd <- sd(result_probit$theta_01)
  logit_theta01_sd <- sd(result_logit$theta_01)
  expect_true(abs(probit_theta01_sd - logit_theta01_sd) / logit_theta01_sd < 0.5,
              info = "Probit and logit should have similar posterior variability.")
})

test_that("link function transformations are mathematically correct", {
  
  # --- Test the transformation functions directly ---
  theta_test <- seq(-5, 5, by = 0.5)
  
  # Probit: alpha = Phi(theta)
  alpha_probit <- pnorm(theta_test)
  
  # Logit: alpha = 1 / (1 + exp(-theta))
  alpha_logit <- plogis(theta_test)
  
  # --- Verify bounds ---
  expect_true(all(alpha_probit >= 0 & alpha_probit <= 1),
              info = "Probit transformation should map to [0,1].")
  expect_true(all(alpha_logit >= 0 & alpha_logit <= 1),
              info = "Logit transformation should map to [0,1].")
  
  # --- Verify monotonicity ---
  expect_true(all(diff(alpha_probit) > 0),
              info = "Probit transformation should be monotonic increasing.")
  expect_true(all(diff(alpha_logit) > 0),
              info = "Logit transformation should be monotonic increasing.")
  
  # --- Verify convergence to limits ---
  expect_true(alpha_probit[1] < 0.01,
              info = "Probit should approach 0 for large negative theta.")
  expect_true(alpha_probit[length(alpha_probit)] > 0.99,
              info = "Probit should approach 1 for large positive theta.")
  
  expect_true(alpha_logit[1] < 0.01,
              info = "Logit should approach 0 for large negative theta.")
  expect_true(alpha_logit[length(alpha_logit)] > 0.99,
              info = "Logit should approach 1 for large positive theta.")
  
  # --- Verify similarity near zero ---
  zero_idx <- which(abs(theta_test) < 0.5)
  if (length(zero_idx) > 0) {
    max_diff_near_zero <- max(abs(alpha_probit[zero_idx] - alpha_logit[zero_idx]))
    expect_true(max_diff_near_zero < 0.05,
                info = "Probit and logit should be similar for theta near zero.")
  }
})
