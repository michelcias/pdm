library(testthat)

# Half-t / Half-Cauchy prior option on the innovation precisions (W_k) of the
# non-Gaussian drivers (binomial, poisson, probit-bernoulli). These models have
# NO observation precision V, so the Half-t dispatch applies only to the W_k.
# The runs are deliberately tiny: we check the wiring (finiteness, positivity,
# recorded prior attribute), not statistical accuracy.

make_binom <- function(n = 30, size = 10, seed = 1) {
  set.seed(seed)
  rbinom(n, size, plogis(cumsum(rnorm(n, sd = 0.2))))
}
make_pois <- function(n = 30, seed = 1) {
  set.seed(seed)
  rpois(n, exp(0.5 + cumsum(rnorm(n, sd = 0.1))))
}
make_bern <- function(n = 30, seed = 1) {
  set.seed(seed)
  rbinom(n, 1, pnorm(cumsum(rnorm(n, sd = 0.2))))
}

CTRL <- list(burnin = 20, thinning = 1, n_draws = 40)

# ---- Binomial (logit link) -------------------------------------------------

test_that("binomial locallevel accepts a Half-Cauchy prior on W_1", {
  y <- make_binom()
  fit <- mcmc_binomial_locallevel(
    y, n_trials = 10, CTRL$burnin, CTRL$thinning, CTRL$n_draws,
    prior_theta01_mean = 0, prior_theta01_prec = 1,
    prior_prec1_type = "halfcauchy", prior_prec1_scale = 5,
    verbose = FALSE, seed = 1
  )
  expect_s3_class(fit, "binomial_locallevel")
  expect_true(all(is.finite(fit$prec_theta1)) && all(fit$prec_theta1 > 0))
  expect_equal(attr(fit, "prior_prec_theta1")$type, "halfcauchy")
})

test_that("binomial localtrend accepts a mixed Gamma / Half-t specification", {
  y <- make_binom()
  fit <- mcmc_binomial_localtrend(
    y, n_trials = 10, CTRL$burnin, CTRL$thinning, CTRL$n_draws,
    prior_theta01_mean = 0, prior_theta01_prec = 1,
    prior_theta02_mean = 0, prior_theta02_prec = 1,
    prior_prec1_type = "halft", prior_prec1_scale = 5, prior_prec1_df = 3,
    prior_prec2_shape = 1e-2, prior_prec2_rate = 1e-2,   # Gamma on W_2
    verbose = FALSE, seed = 1
  )
  expect_s3_class(fit, "binomial_localtrend")
  expect_true(all(fit$prec_theta1 > 0) && all(fit$prec_theta2 > 0))
  expect_equal(attr(fit, "prior_prec_theta1")$type, "halft")
  expect_equal(attr(fit, "prior_prec_theta1")$df, 3)
  expect_equal(attr(fit, "prior_prec_theta2")$type, "gamma")
})

test_that("binomial localacceleration accepts Half-Cauchy on all three W_k", {
  y <- make_binom()
  fit <- mcmc_binomial_localacceleration(
    y, n_trials = 10, CTRL$burnin, CTRL$thinning, CTRL$n_draws,
    prior_theta01_mean = 0, prior_theta01_prec = 1,
    prior_theta02_mean = 0, prior_theta02_prec = 1,
    prior_theta03_mean = 0, prior_theta03_prec = 1,
    prior_prec1_type = "halfcauchy", prior_prec1_scale = 5,
    prior_prec2_type = "halfcauchy", prior_prec2_scale = 5,
    prior_prec3_type = "halfcauchy", prior_prec3_scale = 5,
    verbose = FALSE, seed = 1
  )
  expect_s3_class(fit, "binomial_localacceleration")
  expect_true(all(fit$prec_theta1 > 0) && all(fit$prec_theta2 > 0) &&
                all(fit$prec_theta3 > 0))
  expect_equal(attr(fit, "prior_prec_theta3")$type, "halfcauchy")
})

# ---- Poisson (log link) ----------------------------------------------------

test_that("poisson locallevel accepts a Half-Cauchy prior on W_1", {
  y <- make_pois()
  fit <- mcmc_poisson_locallevel(
    y, CTRL$burnin, CTRL$thinning, CTRL$n_draws,
    prior_theta01_mean = 0, prior_theta01_prec = 1,
    prior_prec1_type = "halfcauchy", prior_prec1_scale = 5,
    verbose = FALSE, seed = 1
  )
  expect_s3_class(fit, "poisson_locallevel")
  expect_true(all(is.finite(fit$prec_theta1)) && all(fit$prec_theta1 > 0))
  expect_equal(attr(fit, "prior_prec_theta1")$type, "halfcauchy")
})

test_that("poisson localtrend and localacceleration accept Half-t priors", {
  y <- make_pois()
  ft <- mcmc_poisson_localtrend(
    y, CTRL$burnin, CTRL$thinning, CTRL$n_draws,
    prior_theta01_mean = 0, prior_theta01_prec = 1,
    prior_theta02_mean = 0, prior_theta02_prec = 1,
    prior_prec1_type = "halft", prior_prec1_scale = 5, prior_prec1_df = 4,
    prior_prec2_type = "halfcauchy", prior_prec2_scale = 5,
    verbose = FALSE, seed = 1
  )
  expect_equal(attr(ft, "prior_prec_theta1")$df, 4)
  expect_equal(attr(ft, "prior_prec_theta2")$type, "halfcauchy")
  expect_true(all(ft$prec_theta1 > 0) && all(ft$prec_theta2 > 0))

  fa <- mcmc_poisson_localacceleration(
    y, CTRL$burnin, CTRL$thinning, CTRL$n_draws,
    prior_theta01_mean = 0, prior_theta01_prec = 1,
    prior_theta02_mean = 0, prior_theta02_prec = 1,
    prior_theta03_mean = 0, prior_theta03_prec = 1,
    prior_prec1_type = "halfcauchy", prior_prec1_scale = 5,
    prior_prec2_type = "halfcauchy", prior_prec2_scale = 5,
    prior_prec3_type = "halfcauchy", prior_prec3_scale = 5,
    verbose = FALSE, seed = 1
  )
  expect_s3_class(fa, "poisson_localacceleration")
  expect_true(all(fa$prec_theta1 > 0) && all(fa$prec_theta3 > 0))
})

# ---- Probit-Bernoulli ------------------------------------------------------

test_that("probit-bernoulli drivers accept Half-t priors on all W_k", {
  y <- make_bern()

  fl <- mcmc_probit_bernoulli_locallevel(
    y, CTRL$burnin, CTRL$thinning, CTRL$n_draws,
    prior_theta01_mean = 0, prior_theta01_prec = 1,
    prior_prec1_type = "halfcauchy", prior_prec1_scale = 5, seed = 1
  )
  expect_s3_class(fl, "probit_bernoulli_locallevel")
  expect_equal(attr(fl, "prior_prec_theta1")$type, "halfcauchy")
  expect_true(all(fl$prec_theta1 > 0))

  ft <- mcmc_probit_bernoulli_localtrend(
    y, CTRL$burnin, CTRL$thinning, CTRL$n_draws,
    prior_theta01_mean = 0, prior_theta01_prec = 1,
    prior_theta02_mean = 0, prior_theta02_prec = 1,
    prior_prec1_type = "halft", prior_prec1_scale = 5, prior_prec1_df = 2,
    prior_prec2_type = "halfcauchy", prior_prec2_scale = 5, seed = 1
  )
  expect_equal(attr(ft, "prior_prec_theta1")$df, 2)
  expect_true(all(ft$prec_theta1 > 0) && all(ft$prec_theta2 > 0))

  fa <- mcmc_probit_bernoulli_localacceleration(
    y, CTRL$burnin, CTRL$thinning, CTRL$n_draws,
    prior_theta01_mean = 0, prior_theta01_prec = 1,
    prior_theta02_mean = 0, prior_theta02_prec = 1,
    prior_theta03_mean = 0, prior_theta03_prec = 1,
    prior_prec1_type = "halfcauchy", prior_prec1_scale = 5,
    prior_prec2_type = "halfcauchy", prior_prec2_scale = 5,
    prior_prec3_type = "halfcauchy", prior_prec3_scale = 5, seed = 1
  )
  expect_s3_class(fa, "probit_bernoulli_localacceleration")
  expect_true(all(fa$prec_theta1 > 0) && all(fa$prec_theta3 > 0))
})

# ---- Backward compatibility + validation ----------------------------------

test_that("Gamma prior stays the default and remains backward compatible", {
  y <- make_pois()
  fit <- mcmc_poisson_locallevel(
    y, CTRL$burnin, CTRL$thinning, CTRL$n_draws,
    prior_theta01_mean = 0, prior_theta01_prec = 1,
    prior_prec1_shape = 1e-2, prior_prec1_rate = 1e-2,
    verbose = FALSE, seed = 1
  )
  expect_equal(attr(fit, "prior_prec_theta1")$type, "gamma")
  expect_true(all(fit$prec_theta1 > 0))
})

test_that("invalid Half-t specifications are rejected (probit locallevel)", {
  y <- make_bern()

  # A Half-Cauchy without a scale is no longer an error: the link families
  # default to a fixed scale of 2 on the link scale (see 0.5-0).
  fit <- mcmc_probit_bernoulli_locallevel(
    y, CTRL$burnin, CTRL$thinning, CTRL$n_draws,
    prior_theta01_mean = 0, prior_theta01_prec = 1,
    prior_prec1_type = "halfcauchy", seed = 1)
  expect_equal(attr(fit, "prior_prec_theta1")$scale, 2)

  # "halfcauchy" contradicted by df != 1.
  expect_error(
    mcmc_probit_bernoulli_locallevel(
      y, CTRL$burnin, CTRL$thinning, CTRL$n_draws,
      prior_theta01_mean = 0, prior_theta01_prec = 1,
      prior_prec1_type = "halfcauchy", prior_prec1_scale = 5, prior_prec1_df = 2,
      seed = 1),
    "df = 1"
  )

  # Gamma asked for by name, but shape/rate omitted. Naming the type is what
  # makes this an error now: Gamma is no longer the default.
  expect_error(
    mcmc_probit_bernoulli_locallevel(
      y, CTRL$burnin, CTRL$thinning, CTRL$n_draws,
      prior_theta01_mean = 0, prior_theta01_prec = 1,
      prior_prec1_type = "gamma", seed = 1),
    "gamma"
  )
})
