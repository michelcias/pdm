library(testthat)

# This file validates the statistical correctness of the MCMC samplers.
# The strategy is to simulate data from known "true" parameters, then run
# the sampler while fixing some parameters to their true values. We then
# check if the posterior distribution of the unfixed parameter is centered
# around its true value.

test_that("mcmc_binomial_locallevel sampler is conditionally correct", {

  # --- 1. Simulation Setup ---
  set.seed(404)
  n <- 200
  n_trials <- 20

  # True parameters for simulation
  theta_01_true <- 0.5
  prec_theta1_true <- 100.0

  # Simulate data
  u1 <- rnorm(n, sd = sqrt(1 / prec_theta1_true))
  theta_1_true <- cumsum(c(theta_01_true, u1))[-1]
  alpha_true <- plogis(theta_1_true)
  y <- as.numeric(rbinom(n, size = n_trials, prob = alpha_true))

  # --- 2. R Wrapper for the Test Sampler ---
  test_sampler <- function(y, n_trials, burnin, n_chain,
                           theta_1_true = NULL, theta_01_true = NULL, prec_theta1_true = NULL,
                           prior_theta01_mean = 0.0, prior_theta01_prec = 1.0,
                           prior_prec1_shape = 1.0, prior_prec1_rate = 1.0,
                           lag_update = 50L, max_step_size = 0.1,
                           base_adaptation_rate = 1.0, decay_exponent = 0.5,
                           target_acceptance = 0.44) {

    .Call("_pdm_test_mcmc_binomial_locallevel_fixed_params",
          y, n_trials, as.integer(burnin), 1L, as.integer(n_chain),
          theta_1_true, theta_01_true, prec_theta1_true,
          prior_theta01_mean, prior_theta01_prec,
          prior_prec1_shape, prior_prec1_rate,
          lag_update, max_step_size, base_adaptation_rate,
          decay_exponent, target_acceptance, FALSE, FALSE)
  }

  # --- 3. Run Tests for Each Conditional ---

  # Test A: Sample theta_01, fixing theta_1 and prec_theta1
  set.seed(405)
  mcmc_out_A <- test_sampler(y, n_trials, burnin = 500, n_chain = 2000,
                             theta_1_true = theta_1_true,
                             prec_theta1_true = prec_theta1_true,
                             prior_theta01_mean = 0, # Prior for theta_01
                             prior_theta01_prec = 1.0)

  # Check if the posterior mean of theta_01 is close to the true value
  posterior_mean_A <- mean(mcmc_out_A$theta_01)
  expect_true(abs(posterior_mean_A - theta_01_true) < 0.1,
              info = "Posterior mean for theta_01 should be close to true value.")

  # Test B: Sample prec_theta1, fixing theta_1 and theta_01
  set.seed(406)
  mcmc_out_B <- test_sampler(y, n_trials, burnin = 500, n_chain = 2000,
                             theta_1_true = theta_1_true,
                             theta_01_true = theta_01_true,
                             prior_prec1_shape = 100, # Prior for prec_theta1
                             prior_prec1_rate = 1)

  # Check if the posterior mean of prec_theta1 is close to the true value
  posterior_mean_B <- mean(mcmc_out_B$prec_theta1)
  expect_true(abs(posterior_mean_B - prec_theta1_true) < (0.25 * prec_theta1_true),
              info = "Posterior mean for prec_theta1 should be close to true value.")

  # Test C: Sample theta_1, fixing theta_01 and prec_theta1
  set.seed(407)
  mcmc_out_C <- test_sampler(y, n_trials, burnin = 500, n_chain = 1000,
                             theta_01_true = theta_01_true,
                             prec_theta1_true = prec_theta1_true)

  # Check if the posterior mean of theta_1 is close to the true value
  posterior_mean_C <- colMeans(mcmc_out_C$theta_1)
  # Check the average absolute difference
  mean_abs_diff <- mean(abs(posterior_mean_C - theta_1_true))
  expect_true(mean_abs_diff < 0.1,
              info = "Posterior mean for theta_1 trajectory should be close to true trajectory.")
})

