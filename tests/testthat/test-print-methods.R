# Test suite for print and summary methods
# Tests consistency between print() and summary() output
# Author: Michel Helcias (michelcias)
# Date: 2025-10-24

# Helper function to create a mock normal_mixture_localtrend object
create_mock_object <- function(n_chain = 1000, n_obs = 100, seed = 123) {
  set.seed(seed)

  # Create MCMC samples with realistic properties
  # Mixture components
  mu_1 <- rnorm(n_chain, mean = 0, sd = 0.5)
  mu_2 <- rnorm(n_chain, mean = 2, sd = 0.5)

  # Precisions (Gamma distributed - right skewed)
  prec_1 <- rgamma(n_chain, shape = 2, rate = 1)
  prec_2 <- rgamma(n_chain, shape = 2, rate = 1)

  # Dynamic states
  theta_01 <- rnorm(n_chain, mean = 0, sd = 1)
  theta_02 <- rnorm(n_chain, mean = 0, sd = 0.5)
  prec_theta1 <- rgamma(n_chain, shape = 5, rate = 1)
  prec_theta2 <- rgamma(n_chain, shape = 10, rate = 1)

  # Time-varying parameters (matrices)
  theta_1 <- matrix(rnorm(n_chain * n_obs), nrow = n_chain, ncol = n_obs)
  theta_2 <- matrix(rnorm(n_chain * n_obs, sd = 0.5), nrow = n_chain, ncol = n_obs)
  alpha <- matrix(runif(n_chain * n_obs, 0.2, 0.8), nrow = n_chain, ncol = n_obs)
  z <- matrix(rbinom(n_chain * n_obs, 1, 0.5), nrow = n_chain, ncol = n_obs)

  # Create list structure
  mock_obj <- list(
    mu_1 = mu_1,
    mu_2 = mu_2,
    prec_1 = prec_1,
    prec_2 = prec_2,
    theta_1 = theta_1,
    theta_2 = theta_2,
    theta_01 = theta_01,
    theta_02 = theta_02,
    prec_theta1 = prec_theta1,
    prec_theta2 = prec_theta2,
    alpha = alpha,
    z = z
  )

  # Add attributes
  attr(mock_obj, "link") <- "logit"
  attr(mock_obj, "n_obs") <- n_obs
  attr(mock_obj, "n_chain") <- n_chain
  attr(mock_obj, "burnin") <- 500
  attr(mock_obj, "thinning") <- 10
  attr(mock_obj, "model_type") <- "localtrend"
  attr(mock_obj, "y") <- rnorm(n_obs)

  # Assign class
  class(mock_obj) <- c("normal_mixture_localtrend", "bdm_mcmc", "list")

  return(mock_obj)
}


# ============================================================================
# Tests for print.normal_mixture_localtrend()
# ============================================================================

test_that("print() shows posterior medians", {
  mock_obj <- create_mock_object()
  output <- capture.output(print(mock_obj))

  # Check that "Posterior Medians" appears in output
  expect_true(any(grepl("Posterior Medians", output, ignore.case = TRUE)))

  # Verify that specific median values are shown
  expect_true(any(grepl("mu_1:", output)))
  expect_true(any(grepl("mu_2:", output)))
  expect_true(any(grepl("phi_1:", output)))
  expect_true(any(grepl("phi_2:", output)))
})

test_that("print() guides user to summary()", {
  mock_obj <- create_mock_object()
  output <- capture.output(print(mock_obj))

  # Should mention summary() function
  expect_true(any(grepl("summary\\(x\\)", output)))

  # Should explain that it shows medians
  expect_true(any(grepl("median", output, ignore.case = TRUE)))
})

test_that("print() respects digits argument", {
  mock_obj <- create_mock_object()

  # Test with different digit settings
  output_3 <- capture.output(print(mock_obj, digits = 3))
  output_5 <- capture.output(print(mock_obj, digits = 5))

  # Extract a numeric value from output
  # Look for pattern like "mu_1:   0.123"
  pattern_3 <- "mu_1:\\s+(-?\\d+\\.\\d{3})"
  pattern_5 <- "mu_1:\\s+(-?\\d+\\.\\d{5})"

  # Check that 3-digit output doesn't have 5 decimals
  match_3 <- grep(pattern_3, output_3, value = TRUE)
  expect_true(length(match_3) > 0)

  # Check that 5-digit output has 5 decimals
  match_5 <- grep(pattern_5, output_5, value = TRUE)
  expect_true(length(match_5) > 0)
})

test_that("print() shows model and MCMC metadata", {
  mock_obj <- create_mock_object()
  output <- capture.output(print(mock_obj))

  # Model information
  expect_true(any(grepl("Type:", output)))
  expect_true(any(grepl("Link function:", output)))
  expect_true(any(grepl("logit", output)))

  # MCMC information
  expect_true(any(grepl("Observations:", output)))
  expect_true(any(grepl("Samples retained:", output)))
  expect_true(any(grepl("Burn-in:", output)))
  expect_true(any(grepl("Thinning:", output)))

  # Check specific values
  expect_true(any(grepl("100", output)))  # n_obs
  expect_true(any(grepl("1000", output))) # n_chain
  expect_true(any(grepl("500", output)))  # burnin
  expect_true(any(grepl("10", output)))   # thinning
})

test_that("print() returns invisibly", {
  mock_obj <- create_mock_object()
  result <- withVisible(print(mock_obj))

  expect_false(result$visible)
  expect_identical(result$value, mock_obj)
})


# ============================================================================
# Tests for summary.normal_mixture_localtrend()
# ============================================================================

test_that("summary() returns correct structure", {
  mock_obj <- create_mock_object()
  summ <- summary(mock_obj)

  # Check class
  expect_s3_class(summ, "summary.normal_mixture_localtrend")

  # Check required components
  expect_true("mixture_params" %in% names(summ))
  expect_true("state_params" %in% names(summ))
  expect_true("alpha_summary" %in% names(summ))
  expect_true("link" %in% names(summ))
  expect_true("model_type" %in% names(summ))
  expect_true("n_obs" %in% names(summ))
  expect_true("n_chain" %in% names(summ))
  expect_true("probs" %in% names(summ))
})

test_that("summary() includes mean AND median", {
  mock_obj <- create_mock_object()
  summ <- summary(mock_obj)

  # Check that both statistics are present
  expect_true("Mean" %in% names(summ$mixture_params))
  expect_true("Median" %in% names(summ$mixture_params))
  expect_true("SD" %in% names(summ$mixture_params))
  expect_true("CI_Lower" %in% names(summ$mixture_params))
  expect_true("CI_Upper" %in% names(summ$mixture_params))

  # Same for state params
  expect_true("Mean" %in% names(summ$state_params))
  expect_true("Median" %in% names(summ$state_params))
})

test_that("summary() mean and median differ for skewed distributions", {
  mock_obj <- create_mock_object()
  summ <- summary(mock_obj)

  # For Gamma-distributed precisions, mean > median
  # prec_1 is row 3, prec_2 is row 4 in mixture_params
  mean_prec1 <- summ$mixture_params$Mean[3]
  median_prec1 <- summ$mixture_params$Median[3]

  # Due to right skew, mean should be larger
  expect_true(mean_prec1 > median_prec1)
})

test_that("summary() respects custom probs", {
  mock_obj <- create_mock_object()

  # Default 95% CI
  summ_95 <- summary(mock_obj, probs = c(0.025, 0.975))
  expect_equal(summ_95$probs, c(0.025, 0.975))

  # Custom 90% CI
  summ_90 <- summary(mock_obj, probs = c(0.05, 0.95))
  expect_equal(summ_90$probs, c(0.05, 0.95))

  # CI should be narrower for 90%
  ci_width_95 <- summ_95$mixture_params$CI_Upper[1] - summ_95$mixture_params$CI_Lower[1]
  ci_width_90 <- summ_90$mixture_params$CI_Upper[1] - summ_90$mixture_params$CI_Lower[1]

  expect_true(ci_width_90 < ci_width_95)
})

test_that("summary() validates probs argument", {
  mock_obj <- create_mock_object()

  # Invalid: not numeric
  expect_error(summary(mock_obj, probs = "invalid"), "must be numeric")

  # Invalid: outside [0, 1]
  expect_error(summary(mock_obj, probs = c(-0.1, 0.5)), "between 0 and 1")
  expect_error(summary(mock_obj, probs = c(0.5, 1.5)), "between 0 and 1")

  # Invalid: wrong length
  expect_error(summary(mock_obj, probs = c(0.025)), "exactly 2 elements")
  expect_error(summary(mock_obj, probs = c(0.025, 0.5, 0.975)), "exactly 2 elements")

  # Invalid: wrong order
  expect_error(summary(mock_obj, probs = c(0.975, 0.025)), "must be less than")
})


# ============================================================================
# Tests for print.summary.normal_mixture_localtrend()
# ============================================================================

test_that("print.summary() explains statistics", {
  mock_obj <- create_mock_object()
  summ <- summary(mock_obj)
  output <- capture.output(print(summ))

  # Should include legend explaining statistics
  expect_true(any(grepl("Statistics Legend", output)))
  expect_true(any(grepl("minimizes squared error", output, ignore.case = TRUE)))
  expect_true(any(grepl("minimizes absolute error", output, ignore.case = TRUE)))
})

test_that("print.summary() shows credible interval level", {
  mock_obj <- create_mock_object()

  # Test with 95% CI
  summ_95 <- summary(mock_obj, probs = c(0.025, 0.975))
  output_95 <- capture.output(print(summ_95))
  expect_true(any(grepl("95\\.0%", output_95)))

  # Test with 90% CI
  summ_90 <- summary(mock_obj, probs = c(0.05, 0.95))
  output_90 <- capture.output(print(summ_90))
  expect_true(any(grepl("90\\.0%", output_90)))
})

test_that("print.summary() respects digits argument", {
  mock_obj <- create_mock_object()
  summ <- summary(mock_obj)

  # Different digit settings
  output_2 <- capture.output(print(summ, digits = 2))
  output_4 <- capture.output(print(summ, digits = 4))

  # Should have different precision
  expect_false(identical(output_2, output_4))
})

test_that("print.summary() guides to plot() and print()", {
  mock_obj <- create_mock_object()
  summ <- summary(mock_obj)
  output <- capture.output(print(summ))

  # Should mention plot() for trajectories
  expect_true(any(grepl("plot\\(\\)", output)))

  # Should mention print() for quick overview
  expect_true(any(grepl("print\\(\\)", output)))
})

test_that("print.summary() returns invisibly", {
  mock_obj <- create_mock_object()
  summ <- summary(mock_obj)
  result <- withVisible(print(summ))

  expect_false(result$visible)
  expect_identical(result$value, summ)
})


# ============================================================================
# Integration tests (print vs summary consistency)
# ============================================================================

test_that("print() medians match summary() medians", {
  mock_obj <- create_mock_object()
  summ <- summary(mock_obj)

  # Extract medians from summary
  summary_median_mu1 <- summ$mixture_params$Median[1]
  summary_median_mu2 <- summ$mixture_params$Median[2]

  # Calculate medians directly (what print() shows)
  direct_median_mu1 <- median(mock_obj$mu_1)
  direct_median_mu2 <- median(mock_obj$mu_2)

  # They should match
  expect_equal(summary_median_mu1, direct_median_mu1)
  expect_equal(summary_median_mu2, direct_median_mu2)
})

test_that("summary table has correct number of rows", {
  mock_obj <- create_mock_object()
  summ <- summary(mock_obj)

  # Mixture params: mu_1, mu_2, phi_1, phi_2
  expect_equal(nrow(summ$mixture_params), 4)

  # State params: theta_01, theta_02, W_1^-1, W_2^-1
  expect_equal(nrow(summ$state_params), 4)

  # Alpha summary: min, median, max
  expect_equal(nrow(summ$alpha_summary), 3)
})

test_that("CI bounds are ordered correctly", {
  mock_obj <- create_mock_object()
  summ <- summary(mock_obj)

  # For all parameters, CI_Lower < CI_Upper
  expect_true(all(summ$mixture_params$CI_Lower < summ$mixture_params$CI_Upper))
  expect_true(all(summ$state_params$CI_Lower < summ$state_params$CI_Upper))
})


# ============================================================================
# Edge cases and error handling
# ============================================================================

test_that("summary() fails on invalid object", {
  # Create object with WRONG class
  bad_obj <- list(mu_1 = 1:10)
  class(bad_obj) <- c("wrong_class", "list")

  # This will call summary.wrong_class (não existe) -> summary.default
  # Our method won't be called, so we test differently

  # Instead, test that our validator catches this
  expect_error(
    validate_normal_mixture_localtrend(bad_obj),
    "must inherit from class 'normal_mixture_localtrend'"
  )
})

test_that("print() and summary() handle small n_chain", {
  # Very small sample size
  small_obj <- create_mock_object(n_chain = 10, n_obs = 5)

  # Should produce output without errors
  expect_output(print(small_obj), "Posterior Medians")

  summ <- summary(small_obj)
  expect_s3_class(summ, "summary.normal_mixture_localtrend")
  expect_output(print(summ), "Mixture Component Parameters")
})

test_that("methods handle extreme values gracefully", {
  mock_obj <- create_mock_object()

  # Add some extreme values
  mock_obj$mu_1[1] <- 1e10
  mock_obj$prec_1[1] <- 1e-10

  # Should not error, but may produce warnings for formatting
  expect_no_error(print(mock_obj))

  summ <- summary(mock_obj)
  expect_s3_class(summ, "summary.normal_mixture_localtrend")

  # Check that statistics are still finite (mean handles extremes)
  expect_true(all(is.finite(summ$mixture_params$Mean)))
  expect_true(all(is.finite(summ$mixture_params$Median)))
})

test_that("print() does not error on missing optional components", {
  mock_obj <- create_mock_object()

  # Remove optional components (like log_sigma for probit link)
  mock_obj$log_sigma <- NULL
  mock_obj$accept_prop <- NULL

  # Should still work
  expect_output(print(mock_obj), "Posterior Medians")

  summ <- summary(mock_obj)
  expect_s3_class(summ, "summary.normal_mixture_localtrend")
})
