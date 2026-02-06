library(testthat)

# This file validates the statistical correctness of the probit Bernoulli MCMC
# sampler for local-level models using Albert-Chib data augmentation.
# The strategy is to simulate Bernoulli data from known "true" parameters, then
# run the Gibbs sampler while fixing some parameters to their true values. We then
# check if the posterior distribution of the unfixed parameter is centered
# around its true value.

test_that("mcmc_probit_bernoulli_locallevel sampler is conditionally correct", {

  skip_on_cran()

  # --- 1. Simulation Setup ---
  set.seed(501)
  n <- 1500

  # True parameters for simulation
  theta_01_true <- -0.2
  prec_theta1_true <- 80.0

  # Simulate Bernoulli data with probit link
  u1 <- rnorm(n, sd = sqrt(1 / prec_theta1_true))
  theta_1_true <- cumsum(c(theta_01_true, u1))[-1]
  alpha_true <- pnorm(theta_1_true)  # probit link: Phi(theta)
  y <- rbinom(n, size = 1, prob = alpha_true)  # Bernoulli outcomes

  # --- 2. R Wrapper for the Test Sampler ---
  test_sampler <- function(y, burnin, n_chain,
                           theta_1_true = NULL, theta_01_true = NULL, prec_theta1_true = NULL,
                           prior_theta01_mean = 0.0, prior_theta01_prec = 1.0,
                           prior_prec1_shape = 1.0, prior_prec1_rate = 1.0) {

    .Call("_bdm_test_mcmc_probit_bernoulli_locallevel_fixed_params",
          y, burnin, 1L, n_chain,
          theta_1_true, theta_01_true, prec_theta1_true,
          prior_theta01_mean, prior_theta01_prec,
          prior_prec1_shape, prior_prec1_rate)
  }

  # --- 3. Run Tests for Each Conditional ---

  # Test A: Sample theta_01, fixing theta_1 and prec_theta1
  set.seed(502)
  mcmc_out_A <- test_sampler(y, burnin = 4000, n_chain = 1500,
                             theta_1_true = theta_1_true,
                             prec_theta1_true = prec_theta1_true,
                             prior_theta01_mean = 0,    # Prior for theta_01
                             prior_theta01_prec = 1)

  # Check if the posterior mean of theta_01 is close to the true value
  posterior_mean_A <- mean(mcmc_out_A$theta_01)
  expect_lt(abs(posterior_mean_A - theta_01_true), 0.15,
            label = "Posterior mean for theta_01 should be close to true value.")

  # Test B: Sample prec_theta1, fixing theta_1 and theta_01
  set.seed(503)
  mcmc_out_B <- test_sampler(y, burnin = 4000, n_chain = 1500,
                             theta_1_true = theta_1_true,
                             theta_01_true = theta_01_true,
                             prior_prec1_shape = 80,   # Prior for prec_theta1
                             prior_prec1_rate = 1)

  # Check if the posterior mean of prec_theta1 is close to the true value
  posterior_mean_B <- mean(mcmc_out_B$prec_theta1)
  expect_lt(abs(posterior_mean_B - prec_theta1_true), 0.3 * prec_theta1_true,
            label = "Posterior mean for prec_theta1 should be close to true value.")

  # Test C: Sample theta_1, fixing theta_01 and prec_theta1
  set.seed(504)
  mcmc_out_C <- test_sampler(y, burnin = 4000, n_chain = 1500,
                             theta_01_true = theta_01_true,
                             prec_theta1_true = prec_theta1_true)

  # Check if the posterior mean of theta_1 is close to the true value
  posterior_mean_C <- colMeans(mcmc_out_C$theta_1)
  # Check the average absolute difference
  mean_abs_diff <- mean(abs(posterior_mean_C - theta_1_true))
  expect_lt(mean_abs_diff, 0.25,
            label = "Posterior mean for theta_1 trajectory should be close to true trajectory.")

  # Test D: Verify alpha transformation is correct
  expect_true(all(mcmc_out_C$alpha >= 0 & mcmc_out_C$alpha <= 1),
              label = "All alpha values should be in [0,1] range.")

  # Check probit transformation: alpha = Phi(theta_1)
  expect_equal(mcmc_out_C$alpha, pnorm(mcmc_out_C$theta_1), tolerance = 1e-12,
               label = "Alpha should equal Phi(theta_1) exactly.")
})

test_that("mcmc_probit_bernoulli_locallevel handles edge cases correctly", {

  # Test with extreme probability cases
  set.seed(505)
  n_small <- 20

  # Case 1: All zeros (should handle gracefully)
  y_zeros <- rep(0, n_small)
  expect_no_error({
    result_zeros <- .Call("_bdm_C_MCMC_probit_bernoulli_locallevel",
                          y_zeros, 50L, 1L, 100L,
                          0.0, 1.0, 1.0, 1.0,
                          FALSE, 60L)
  })

  # Case 2: All ones (should handle gracefully)
  y_ones <- rep(1, n_small)
  expect_no_error({
    result_ones <- .Call("_bdm_C_MCMC_probit_bernoulli_locallevel",
                         y_ones, 50L, 1L, 100L,
                         0.0, 1.0, 1.0, 1.0,
                         FALSE, 60L)
  })

  # Verify output structure for edge cases
  expect_true(is.list(result_zeros))
  expect_true(is.list(result_ones))
  expect_true(all(c("theta_1", "theta_01", "prec_theta1", "alpha") %in% names(result_zeros)))
})
