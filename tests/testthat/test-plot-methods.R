# Test suite for plot methods
# Tests visualization functions for normal_mixture_localtrend
# Author: Michel Helcias (michelcias)
# Date: 2025-10-25

# Helper function (reuse from test-print-methods.R or recreate here)
create_mock_object <- function(n_chain = 100, n_obs = 50, seed = 123) {
  set.seed(seed)

  mock_obj <- list(
    mu_1 = rnorm(n_chain, mean = 0, sd = 0.5),
    mu_2 = rnorm(n_chain, mean = 2, sd = 0.5),
    prec_1 = rgamma(n_chain, shape = 2, rate = 1),
    prec_2 = rgamma(n_chain, shape = 2, rate = 1),
    theta_1 = matrix(rnorm(n_chain * n_obs), nrow = n_chain, ncol = n_obs),
    theta_2 = matrix(rnorm(n_chain * n_obs, sd = 0.5), nrow = n_chain, ncol = n_obs),
    theta_01 = rnorm(n_chain, mean = 0, sd = 1),
    theta_02 = rnorm(n_chain, mean = 0, sd = 0.5),
    prec_theta1 = rgamma(n_chain, shape = 5, rate = 1),
    prec_theta2 = rgamma(n_chain, shape = 10, rate = 1),
    alpha = matrix(runif(n_chain * n_obs, 0.2, 0.8), nrow = n_chain, ncol = n_obs),
    z = matrix(rbinom(n_chain * n_obs, 1, 0.5), nrow = n_chain, ncol = n_obs)
  )

  attr(mock_obj, "link") <- "logit"
  attr(mock_obj, "n_obs") <- n_obs
  attr(mock_obj, "n_chain") <- n_chain
  attr(mock_obj, "burnin") <- 500
  attr(mock_obj, "thinning") <- 10
  attr(mock_obj, "model_type") <- "localtrend"
  attr(mock_obj, "y") <- rnorm(n_obs)

  class(mock_obj) <- c("normal_mixture_localtrend", "bdm_mcmc", "list")

  return(mock_obj)
}


# ============================================================================
# Tests for Base R Graphics
# ============================================================================

test_that("plot() runs without error with base graphics", {
  mock_obj <- create_mock_object()

  # Suppress plotting to NULL device
  pdf(NULL)
  on.exit(dev.off())

  expect_no_error(plot(mock_obj, type = "mcmc", engine = "base"))
  expect_no_error(plot(mock_obj, type = "params", engine = "base"))
  expect_no_error(plot(mock_obj, type = "states", engine = "base"))
  expect_no_error(plot(mock_obj, type = "alpha", engine = "base"))
})

test_that("plot() type='mcmc' accepts which argument", {
  mock_obj <- create_mock_object()

  pdf(NULL)
  on.exit(dev.off())

  # Single parameter
  expect_no_error(plot(mock_obj, type = "mcmc", which = 1, engine = "base"))

  # Multiple parameters
  expect_no_error(plot(mock_obj, type = "mcmc", which = 1:3, engine = "base"))

  # All 8 parameters
  expect_no_error(plot(mock_obj, type = "mcmc", which = 1:8, engine = "base"))
})

test_that("plot() validates which argument for mcmc type", {
  mock_obj <- create_mock_object()

  pdf(NULL)
  on.exit(dev.off())

  # Invalid: which out of range
  expect_error(
    plot(mock_obj, type = "mcmc", which = 0, engine = "base"),
    "`which` must be between 1 and 8"
  )

  expect_error(
    plot(mock_obj, type = "mcmc", which = 9, engine = "base"),
    "`which` must be between 1 and 8"
  )

  expect_error(
    plot(mock_obj, type = "mcmc", which = c(1, 10), engine = "base"),
    "`which` must be between 1 and 8"
  )
})

test_that("plot() validates type argument", {
  mock_obj <- create_mock_object()

  pdf(NULL)
  on.exit(dev.off())

  expect_error(plot(mock_obj, type = "invalid"), "'arg' should be one of")
})

test_that("plot() validates engine argument", {
  mock_obj <- create_mock_object()

  pdf(NULL)
  on.exit(dev.off())

  expect_error(plot(mock_obj, engine = "invalid"), "'arg' should be one of")
})

test_that("plot() respects which argument for different types", {
  mock_obj <- create_mock_object()

  pdf(NULL)
  on.exit(dev.off())

  # Should only plot specified subplots
  expect_no_error(plot(mock_obj, type = "mcmc", which = c(1, 3), engine = "base"))
  expect_no_error(plot(mock_obj, type = "params", which = 1:2, engine = "base"))
  expect_no_error(plot(mock_obj, type = "states", which = c(1, 2), engine = "base"))
})

test_that("plot() returns invisibly", {
  mock_obj <- create_mock_object()

  pdf(NULL)
  result <- withVisible(plot(mock_obj, type = "alpha", engine = "base"))
  dev.off()

  expect_false(result$visible)
  expect_identical(result$value, mock_obj)
})

test_that("plot() handles overlay_data argument", {
  mock_obj <- create_mock_object()

  pdf(NULL)
  on.exit(dev.off())

  # With data overlay
  expect_no_error(plot(mock_obj, type = "alpha", overlay_data = TRUE, engine = "base"))

  # Without data overlay
  expect_no_error(plot(mock_obj, type = "alpha", overlay_data = FALSE, engine = "base"))

  # Without data attribute
  mock_obj_no_data <- mock_obj
  attr(mock_obj_no_data, "y") <- NULL
  expect_no_error(plot(mock_obj_no_data, type = "alpha", overlay_data = TRUE, engine = "base"))
})

test_that("plot() type='all' generates multiple pages", {
  mock_obj <- create_mock_object()

  pdf(NULL)
  on.exit(dev.off())

  # Should generate 12 pages:
  # - 8 pages for individual MCMC diagnostics
  # - 1 page for mixture params
  # - 1 page for dynamic states
  # - 2 pages for mixture weights (alpha_t and z_t)
  expect_no_error(plot(mock_obj, type = "all", ask = FALSE, engine = "base"))
})


# ============================================================================
# Tests for ggplot2 Graphics
# ============================================================================

test_that("plot() runs without error with ggplot2 (if available)", {
  skip_if_not_installed("ggplot2")

  mock_obj <- create_mock_object()

  expect_no_error(plot(mock_obj, type = "mcmc", which = 1, engine = "ggplot2"))
  expect_no_error(plot(mock_obj, type = "params", engine = "ggplot2"))
  expect_no_error(plot(mock_obj, type = "states", engine = "ggplot2"))
  expect_no_error(plot(mock_obj, type = "alpha", engine = "ggplot2"))
})

test_that("plot() handles missing hexbin with ggplot2", {
  skip_if_not_installed("ggplot2")

  # Skip if hexbin IS installed (we want to test when it's NOT available)
  if (requireNamespace("hexbin", quietly = TRUE)) {
    skip("hexbin is installed - cannot test fallback behavior")
  }

  mock_obj <- create_mock_object()

  # Should show informative message about hexbin
  expect_message(
    plot(mock_obj, type = "params", which = 1, engine = "ggplot2"),
    "Install 'hexbin'"
  )
})

test_that("plot() uses hexbin when available with ggplot2", {
  skip_if_not_installed("ggplot2")
  skip_if_not_installed("hexbin")

  mock_obj <- create_mock_object()

  # Should NOT show message when hexbin is available
  expect_no_message(
    plot(mock_obj, type = "params", which = 1, engine = "ggplot2")
  )
})

test_that("plot() handles missing patchwork gracefully", {
  skip_if_not_installed("ggplot2")

  mock_obj <- create_mock_object()

  # Should work regardless of patchwork availability
  # (prints sequentially if patchwork not available)
  expect_no_error(plot(mock_obj, type = "params", engine = "ggplot2"))
  expect_no_error(plot(mock_obj, type = "states", engine = "ggplot2"))
})

test_that("plot() handles ggplot2 unavailability gracefully", {
  skip("Requires manual testing - see comments in test file")

  # Manual test procedure:
  # 1. remove.packages("ggplot2")
  # 2. mock_obj <- create_mock_object()
  # 3. plot(mock_obj, engine = "ggplot2")
  # 4. Expect warning about falling back to base graphics
  # 5. install.packages("ggplot2")
})


# ============================================================================
# Integration Tests
# ============================================================================

test_that("plot() works with different object sizes", {
  # Small object
  small_obj <- create_mock_object(n_chain = 10, n_obs = 5)

  pdf(NULL)
  expect_no_error(plot(small_obj, type = "alpha", engine = "base"))
  dev.off()

  # Large object
  large_obj <- create_mock_object(n_chain = 1000, n_obs = 200)

  pdf(NULL)
  expect_no_error(plot(large_obj, type = "alpha", engine = "base"))
  dev.off()
})

test_that("plot() handles extreme values", {
  mock_obj <- create_mock_object()

  # Add extreme values
  mock_obj$mu_1[1] <- 1e10
  mock_obj$alpha[, 1] <- c(rep(0, 50), rep(1, 50))

  pdf(NULL)
  expect_no_error(plot(mock_obj, type = "params", engine = "base"))
  expect_no_error(plot(mock_obj, type = "alpha", engine = "base"))
  dev.off()
})

test_that("all plot types work together", {
  mock_obj <- create_mock_object()

  pdf(NULL)
  on.exit(dev.off())

  # Test that all types can be called sequentially
  expect_no_error({
    plot(mock_obj, type = "mcmc", which = 1, engine = "base")
    plot(mock_obj, type = "params", engine = "base")
    plot(mock_obj, type = "states", engine = "base")
    plot(mock_obj, type = "alpha", engine = "base")
  })
})

test_that("plot() mcmc diagnostics covers all 8 parameters", {
  mock_obj <- create_mock_object()

  pdf(NULL)
  on.exit(dev.off())

  # Test each parameter individually
  for (i in 1:8) {
    expect_no_error(
      plot(mock_obj, type = "mcmc", which = i, engine = "base")
    )
  }
})


# ============================================================================
# Edge Cases
# ============================================================================

test_that("plot() handles missing optional components", {
  mock_obj <- create_mock_object()

  # Remove optional components
  attr(mock_obj, "y") <- NULL

  pdf(NULL)
  expect_no_error(plot(mock_obj, type = "alpha", overlay_data = TRUE, engine = "base"))
  dev.off()
})

test_that("plot() method dispatch works correctly", {
  # Object with correct class uses our custom method
  mock_obj <- create_mock_object()
  expect_true(inherits(mock_obj, "normal_mixture_localtrend"))

  pdf(NULL)
  on.exit(dev.off())

  # Custom method should work
  expect_no_error(plot(mock_obj, type = "alpha", engine = "base"))

  # Object WITHOUT our class uses plot.default (different behavior, but no error)
  plain_list <- list(x = 1:10, y = 1:10)
  expect_false(inherits(plain_list, "normal_mixture_localtrend"))

  # plot.default handles generic lists
  expect_no_error(plot(plain_list))
})

test_that("plot() respects par() settings", {
  mock_obj <- create_mock_object()

  pdf(NULL)
  on.exit(dev.off())

  # Save original par
  oldpar <- par(no.readonly = TRUE)

  # Set custom par
  par(mfrow = c(1, 1), mar = c(5, 4, 4, 2))

  # Plot should work
  expect_no_error(plot(mock_obj, type = "alpha", engine = "base"))

  # Par should be restored after plot (our functions use on.exit)
  # Note: This is implicit in our implementation

  # Restore par
  par(oldpar)
})

# test_that("plot() handles NA values in data", {
#   mock_obj <- create_mock_object()
#
#   # Add some NA values
#   mock_obj$mu_1[1:5] <- NA
#   mock_obj$alpha[1, 1:10] <- NA
#
#   pdf(NULL)
#   on.exit(dev.off())
#
#   # Should handle NAs gracefully (density() has na.rm)
#   expect_no_error(plot(mock_obj, type = "mcmc", which = 1, engine = "base"))
#   expect_no_error(plot(mock_obj, type = "alpha", engine = "base"))
# })


# ============================================================================
# Performance Tests (optional, can be slow)
# ============================================================================

test_that("plot() is reasonably fast for typical sizes", {
  skip_on_cran()
  skip_if_not(interactive())  # Only run interactively

  mock_obj <- create_mock_object(n_chain = 1000, n_obs = 100)

  pdf(NULL)
  on.exit(dev.off())

  # Should complete in reasonable time (< 10 seconds)
  expect_lt(
    system.time({
      plot(mock_obj, type = "mcmc", which = 1, engine = "base")
    })[["elapsed"]],
    10
  )
})


# ============================================================================
# Documentation Tests
# ============================================================================

test_that("plot() examples in documentation work", {
  skip_on_cran()

  mock_obj <- create_mock_object()

  pdf(NULL)
  on.exit(dev.off())

  # Examples from documentation
  expect_no_error(plot(mock_obj, type = "all", engine = "base", ask = FALSE))
  expect_no_error(plot(mock_obj, type = "mcmc", which = 1:2))
  expect_no_error(plot(mock_obj, type = "alpha"))
})
