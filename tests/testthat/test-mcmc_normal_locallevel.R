library(testthat)

# Unit tests for the innovation/observation precision prior options
# (Gamma vs Half-t / Half-Cauchy) exposed by `mcmc_normal_locallevel()`.

# Small, fast dataset shared across tests.
make_y <- function(n = 40, seed = 1) {
  set.seed(seed)
  cumsum(rnorm(n)) + rnorm(n, sd = 0.3)
}

run_ll <- function(y, ...) {
  mcmc_normal_locallevel(
    y,
    burnin   = 30,
    thinning = 1,
    n_draws  = 60,
    prior_theta01_mean = y[1],
    prior_theta01_prec = 1 / var(y),
    seed = 123,
    ...
  )
}

test_that("Gamma prior (default) keeps the original interface and output", {
  y <- make_y()

  # Backward-compatible call: explicit Gamma shape/rate, no new arguments.
  fit <- run_ll(
    y,
    prior_prec1_shape  = 1e-2,
    prior_prec1_rate   = 1e-2,
    prior_prec_y_shape = 1e-2,
    prior_prec_y_rate  = 1e-2
  )

  expect_s3_class(fit, "normal_locallevel")
  expect_named(fit, c("theta_1", "theta_01", "prec_theta1", "prec_y"))
  expect_length(fit$prec_theta1, 60)
  expect_true(all(fit$prec_theta1 > 0))
  expect_true(all(fit$prec_y > 0))
  expect_equal(attr(fit, "prior_prec_theta1")$type, "gamma")
  expect_equal(attr(fit, "prior_prec_y")$type, "gamma")
})

test_that("Half-Cauchy prior on both precisions runs and is recorded", {
  y <- make_y()

  fit <- run_ll(
    y,
    prior_prec1_type   = "halfcauchy",
    prior_prec1_scale  = 5,
    prior_prec_y_type  = "halfcauchy",
    prior_prec_y_scale = 5
  )

  expect_s3_class(fit, "normal_locallevel")
  expect_true(all(is.finite(fit$prec_theta1)) && all(fit$prec_theta1 > 0))
  expect_true(all(is.finite(fit$prec_y)) && all(fit$prec_y > 0))
  expect_equal(attr(fit, "prior_prec_theta1")$type, "halfcauchy")
  expect_equal(attr(fit, "prior_prec_theta1")$df, 1)
  expect_equal(attr(fit, "prior_prec_theta1")$scale, 5)
})

test_that("Half-t prior with df > 1 and a mixed Gamma/Half-t specification run", {
  y <- make_y()

  fit <- run_ll(
    y,
    prior_prec1_type   = "halft",
    prior_prec1_scale  = 5,
    prior_prec1_df     = 3,
    prior_prec_y_shape = 1e-2,   # observation precision keeps the Gamma prior
    prior_prec_y_rate  = 1e-2
  )

  expect_s3_class(fit, "normal_locallevel")
  expect_equal(attr(fit, "prior_prec_theta1")$type, "halft")
  expect_equal(attr(fit, "prior_prec_theta1")$df, 3)
  expect_equal(attr(fit, "prior_prec_y")$type, "gamma")
})

test_that("invalid prior specifications are rejected with informative errors", {
  y <- make_y()

  # A Half-t without a scale is no longer an error: the scale is derived from
  # the data (see R/innovation_priors.R). This is the 0.4-0 contract.
  fit <- run_ll(y, prior_prec1_type = "halfcauchy",
                prior_prec_y_shape = 1e-2, prior_prec_y_rate = 1e-2)
  expect_equal(attr(fit, "prior_prec_theta1")$scale,
               sd(diff(y)) / (2 * sqrt(2)))

  # "halfcauchy" contradicted by df != 1.
  expect_error(
    run_ll(y, prior_prec1_type = "halfcauchy", prior_prec1_scale = 5,
           prior_prec1_df = 2,
           prior_prec_y_shape = 1e-2, prior_prec_y_rate = 1e-2),
    "df = 1"
  )

  # Gamma asked for by name, but shape/rate omitted. Naming the type is what
  # makes this an error now -- a bare shape/rate pair is read as Gamma instead.
  expect_error(
    run_ll(y, prior_prec1_type = "gamma",
           prior_prec_y_shape = 1e-2, prior_prec_y_rate = 1e-2),
    "gamma"
  )
})
