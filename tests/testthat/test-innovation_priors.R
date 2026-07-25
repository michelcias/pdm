library(testthat)

# The Gaussian precision priors default to a data-scaled Half-Cauchy since
# 0.4-0. Two things matter and are tested here: that the resulting fit does not
# depend on the units of `y`, and that a call written against the old Gamma
# default still gets a Gamma.

make_y <- function(n = 120, seed = 5, scale = 1) {
  set.seed(seed)
  (rep(10, n) + rnorm(n)) * scale     # constant state: where the prior decides
}

fit_ll <- function(y, ...) {
  mcmc_normal_locallevel(
    y, burnin = 300, thinning = 2, n_chain = 200,
    prior_theta01_mean = y[1], prior_theta01_prec = 1 / var(y),
    verbose = FALSE, seed = 1, ...
  )
}


test_that("innovation_prior_scale() deflates by the noise amplification", {
  set.seed(1)
  y <- rnorm(500)

  # var(diff^k e) = choose(2k, k) * V for iid noise, so the k-th difference is
  # deflated by sqrt of that to put the orders on a common footing.
  expect_equal(innovation_prior_scale(y, 1), sd(diff(y)) / (2 * sqrt(2)))
  expect_equal(innovation_prior_scale(y, 2),
               sd(diff(y, differences = 2)) / (2 * sqrt(6)))
  expect_equal(innovation_prior_scale(y, 3),
               sd(diff(y, differences = 3)) / (2 * sqrt(20)))

  # Without the deflation the scales would grow with k; with it they do not.
  raw <- vapply(1:3, function(k) sd(diff(y, differences = k)) / 2, numeric(1))
  adj <- vapply(1:3, function(k) innovation_prior_scale(y, k), numeric(1))
  expect_gt(raw[3] / raw[1], 2)
  expect_lt(max(adj) / min(adj), 1.2)
})


test_that("the scales are proportional to the units of y", {
  y <- make_y()
  expect_equal(innovation_prior_scale(100 * y, 1),
               100 * innovation_prior_scale(y, 1))
  expect_equal(observation_prior_scale(100 * y), 100 * observation_prior_scale(y))
})


test_that("the Gaussian samplers default to a data-scaled Half-Cauchy", {
  y   <- make_y()
  fit <- fit_ll(y)

  expect_equal(attr(fit, "prior_prec_theta1")$type, "halfcauchy")
  expect_equal(attr(fit, "prior_prec_theta1")$scale, innovation_prior_scale(y, 1))
  expect_equal(attr(fit, "prior_prec_y")$type, "halfcauchy")
  expect_equal(attr(fit, "prior_prec_y")$scale, observation_prior_scale(y))
})


test_that("higher orders take the matching difference", {
  y <- make_y()

  lt <- mcmc_normal_localtrend(
    y, 300, 2, 200,
    prior_theta01_mean = y[1], prior_theta01_prec = 1 / var(y),
    prior_theta02_mean = 0,    prior_theta02_prec = 1 / var(y),
    verbose = FALSE, seed = 1
  )
  expect_equal(attr(lt, "prior_prec_theta1")$scale, innovation_prior_scale(y, 1))
  expect_equal(attr(lt, "prior_prec_theta2")$scale, innovation_prior_scale(y, 2))

  la <- mcmc_normal_localacceleration(
    y, 300, 2, 200,
    prior_theta01_mean = y[1], prior_theta01_prec = 1 / var(y),
    prior_theta02_mean = 0,    prior_theta02_prec = 1 / var(y),
    prior_theta03_mean = 0,    prior_theta03_prec = 1 / var(y),
    verbose = FALSE, seed = 1
  )
  expect_equal(attr(la, "prior_prec_theta3")$scale, innovation_prior_scale(y, 3))
})


test_that("the fit is invariant to the units of y", {
  # The reason for the change. A fixed Gamma prior is stated in absolute units,
  # so rescaling the data moves the prior relative to the likelihood; a
  # data-scaled prior moves with it.
  est <- function(scale, ...) {
    y <- make_y(scale = scale)
    colMeans(fit_ll(y, ...)$theta_1) / scale
  }

  a <- est(1)
  b <- est(100)
  expect_equal(a, b, tolerance = 1e-8)

  # The old default is not invariant, which is what this replaces.
  ga <- est(1,   prior_prec1_shape = 1e-2, prior_prec1_rate = 1e-2,
                 prior_prec_y_shape = 1e-2, prior_prec_y_rate = 1e-2)
  gb <- est(100, prior_prec1_shape = 1e-2, prior_prec1_rate = 1e-2,
                 prior_prec_y_shape = 1e-2, prior_prec_y_rate = 1e-2)
  expect_false(isTRUE(all.equal(ga, gb, tolerance = 1e-3)))
})


test_that("a bare shape/rate pair is still read as Gamma", {
  # Code written against the pre-0.4-0 default must keep working rather than
  # silently getting a Half-Cauchy with its hyperparameters ignored.
  y <- make_y()

  fit <- fit_ll(y, prior_prec1_shape = 1e-2, prior_prec1_rate = 1e-2,
                prior_prec_y_shape = 1e-2, prior_prec_y_rate = 1e-2)
  expect_equal(attr(fit, "prior_prec_theta1")$type, "gamma")
  expect_equal(attr(fit, "prior_prec_theta1")$shape, 1e-2)
  expect_equal(attr(fit, "prior_prec_y")$type, "gamma")

  # Mixed: one specified as Gamma, the other left at the new default.
  mixed <- fit_ll(y, prior_prec1_shape = 1e-2, prior_prec1_rate = 1e-2)
  expect_equal(attr(mixed, "prior_prec_theta1")$type, "gamma")
  expect_equal(attr(mixed, "prior_prec_y")$type, "halfcauchy")
})


test_that("naming the type explicitly always wins", {
  y <- make_y()

  # Named Half-t with a scale of its own.
  fit <- fit_ll(y, prior_prec1_type = "halft", prior_prec1_scale = 3,
                prior_prec1_df = 4)
  expect_equal(attr(fit, "prior_prec_theta1")$type, "halft")
  expect_equal(attr(fit, "prior_prec_theta1")$scale, 3)
  expect_equal(attr(fit, "prior_prec_theta1")$df, 4)

  # Named Gamma without hyperparameters still errors rather than falling back.
  expect_error(fit_ll(y, prior_prec1_type = "gamma"), "gamma")
})
