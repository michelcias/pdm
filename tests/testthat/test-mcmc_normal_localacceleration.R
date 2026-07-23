library(testthat)

# Unit tests for the innovation/observation precision prior options
# (Gamma vs Half-t / Half-Cauchy) exposed by `mcmc_normal_localacceleration()`.

# Small, fast dataset shared across tests.
make_y <- function(n = 40, seed = 1) {
  set.seed(seed)
  cumsum(cumsum(cumsum(rnorm(n, sd = 0.1)))) + rnorm(n, sd = 0.3)
}

run_la <- function(y, ...) {
  mcmc_normal_localacceleration(
    y,
    burnin   = 30,
    thinning = 1,
    n_chain  = 60,
    prior_theta01_mean = y[1],
    prior_theta01_prec = 1 / var(y),
    prior_theta02_mean = 0,
    prior_theta02_prec = 1 / var(y),
    prior_theta03_mean = 0,
    prior_theta03_prec = 1 / var(y),
    seed = 123,
    ...
  )
}

test_that("Gamma prior (default) keeps the original interface and output", {
  y <- make_y()

  fit <- run_la(
    y,
    prior_prec1_shape  = 1e-2, prior_prec1_rate  = 1e-2,
    prior_prec2_shape  = 1e-2, prior_prec2_rate  = 1e-2,
    prior_prec3_shape  = 1e-2, prior_prec3_rate  = 1e-2,
    prior_prec_y_shape = 1e-2, prior_prec_y_rate = 1e-2
  )

  expect_s3_class(fit, "normal_localacceleration")
  expect_named(fit, c("theta_1", "theta_2", "theta_3",
                      "theta_01", "theta_02", "theta_03",
                      "prec_theta1", "prec_theta2", "prec_theta3", "prec_y"))
  expect_length(fit$prec_theta1, 60)
  expect_true(all(fit$prec_theta1 > 0) && all(fit$prec_theta2 > 0) &&
                all(fit$prec_theta3 > 0) && all(fit$prec_y > 0))
  expect_equal(attr(fit, "prior_prec_theta1")$type, "gamma")
  expect_equal(attr(fit, "prior_prec_theta3")$type, "gamma")
  expect_equal(attr(fit, "prior_prec_y")$type, "gamma")
})

test_that("Half-Cauchy prior on all four precisions runs and is recorded", {
  y <- make_y()

  fit <- run_la(
    y,
    prior_prec1_type  = "halfcauchy", prior_prec1_scale  = 5,
    prior_prec2_type  = "halfcauchy", prior_prec2_scale  = 5,
    prior_prec3_type  = "halfcauchy", prior_prec3_scale  = 5,
    prior_prec_y_type = "halfcauchy", prior_prec_y_scale = 5
  )

  expect_s3_class(fit, "normal_localacceleration")
  expect_true(all(is.finite(fit$prec_theta1)) && all(fit$prec_theta1 > 0))
  expect_true(all(is.finite(fit$prec_theta3)) && all(fit$prec_theta3 > 0))
  expect_true(all(is.finite(fit$prec_y)) && all(fit$prec_y > 0))
  expect_equal(attr(fit, "prior_prec_theta2")$type, "halfcauchy")
  expect_equal(attr(fit, "prior_prec_theta3")$df, 1)
  expect_equal(attr(fit, "prior_prec_y")$scale, 5)
})

test_that("Mixed Gamma/Half-t specification (df > 1) runs and is recorded", {
  y <- make_y()

  fit <- run_la(
    y,
    prior_prec1_type   = "halft", prior_prec1_scale = 5, prior_prec1_df = 4,
    prior_prec2_shape  = 1e-2,    prior_prec2_rate  = 1e-2,   # Gamma on W_2
    prior_prec3_type   = "halfcauchy", prior_prec3_scale = 2,
    prior_prec_y_shape = 1e-2,    prior_prec_y_rate = 1e-2    # Gamma on V
  )

  expect_s3_class(fit, "normal_localacceleration")
  expect_equal(attr(fit, "prior_prec_theta1")$type, "halft")
  expect_equal(attr(fit, "prior_prec_theta1")$df, 4)
  expect_equal(attr(fit, "prior_prec_theta2")$type, "gamma")
  expect_equal(attr(fit, "prior_prec_theta3")$type, "halfcauchy")
  expect_equal(attr(fit, "prior_prec_y")$type, "gamma")
})

test_that("invalid prior specifications are rejected with informative errors", {
  y <- make_y()

  gamma_rest <- list(
    prior_prec2_shape = 1e-2, prior_prec2_rate = 1e-2,
    prior_prec3_shape = 1e-2, prior_prec3_rate = 1e-2,
    prior_prec_y_shape = 1e-2, prior_prec_y_rate = 1e-2
  )

  # Half-Cauchy without a scale (on W_1).
  expect_error(
    do.call(run_la, c(list(y, prior_prec1_type = "halfcauchy"), gamma_rest)),
    "scale"
  )

  # "halfcauchy" contradicted by df != 1 (on W_1).
  expect_error(
    do.call(run_la, c(list(y, prior_prec1_type = "halfcauchy",
                           prior_prec1_scale = 5, prior_prec1_df = 2), gamma_rest)),
    "df = 1"
  )

  # Gamma prior but shape/rate omitted (on W_1).
  expect_error(
    do.call(run_la, c(list(y), gamma_rest)),
    "gamma"
  )
})
