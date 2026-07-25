library(testthat)

# Half-t / Half-Cauchy priors in the Gaussian mixture drivers. These carry BOTH
# state innovation precisions W_k AND the two mixture component precisions phi_k
# (sampled inside conditional_mixture_normal_parameters_k2, with a label-switch
# swap). Tiny runs: we check wiring (finiteness, positivity, recorded prior
# attributes) and backward compatibility, not statistical accuracy. There was no
# prior test coverage for the mixture family, so these also guard it generally.

make_mix <- function(n = 60, seed = 1) {
  set.seed(seed)
  theta <- cumsum(rnorm(n, sd = 0.15))     # latent process driving the weight
  z <- rbinom(n, 1, pnorm(theta))
  ifelse(z == 1, rnorm(n, 2, 0.7), rnorm(n, -2, 0.7))
}

CTRL <- list(burnin = 40, thinning = 1, n_draws = 60)

test_that("mixture locallevel: Gamma default works and stays backward compatible", {
  y <- make_mix()
  fit <- mcmc_normal_mixture_locallevel(
    y, link = "probit", CTRL$burnin, CTRL$thinning, CTRL$n_draws,
    verbose = FALSE, seed = 1
  )
  expect_s3_class(fit, "normal_mixture_locallevel")
  expect_true(all(fit$prec_1 > 0) && all(fit$prec_2 > 0) &&
                all(fit$prec_theta1 > 0))
  expect_equal(attr(fit, "prior_prec_phi1")$type, "gamma")
  expect_equal(attr(fit, "prior_prec_phi2")$type, "gamma")
  # W_1 is the *state* innovation and defaults to a Half-Cauchy since 0.5-0;
  # the component precisions phi_k keep their Gamma, for the opposite reason
  # (see docs/mixture-convergence.md).
  expect_equal(attr(fit, "prior_prec_theta1")$type, "halfcauchy")
  expect_equal(attr(fit, "prior_prec_theta1")$scale, 2)
})

test_that("mixture locallevel: Half-Cauchy on phi_1, phi_2 and W_1 (probit)", {
  y <- make_mix()
  fit <- mcmc_normal_mixture_locallevel(
    y, link = "probit", CTRL$burnin, CTRL$thinning, CTRL$n_draws,
    prior_prec01_type = "halfcauchy", prior_prec01_scale = 2,
    prior_prec02_type = "halfcauchy", prior_prec02_scale = 2,
    prior_prec1_type  = "halfcauchy", prior_prec1_scale  = 2,
    verbose = FALSE, seed = 1
  )
  expect_s3_class(fit, "normal_mixture_locallevel")
  expect_true(all(is.finite(fit$prec_1)) && all(fit$prec_1 > 0))
  expect_true(all(is.finite(fit$prec_2)) && all(fit$prec_2 > 0))
  expect_true(all(is.finite(fit$prec_theta1)) && all(fit$prec_theta1 > 0))
  expect_equal(attr(fit, "prior_prec_phi1")$type, "halfcauchy")
  expect_equal(attr(fit, "prior_prec_phi2")$df, 1)
  expect_equal(attr(fit, "prior_prec_theta1")$scale, 2)
})

test_that("mixture localtrend: mixed Gamma / Half-t on phi and W (logit)", {
  y <- make_mix()
  fit <- mcmc_normal_mixture_localtrend(
    y, link = "logit", CTRL$burnin, CTRL$thinning, CTRL$n_draws,
    prior_prec01_type = "halft", prior_prec01_scale = 2, prior_prec01_df = 3,
    prior_prec02_type = "halfcauchy", prior_prec02_scale = 2,
    prior_prec1_type  = "halfcauchy", prior_prec1_scale  = 1,
    prior_prec2_shape = 1e-2, prior_prec2_rate = 1e-2,   # Gamma on W_2
    verbose = FALSE, seed = 1
  )
  expect_s3_class(fit, "normal_mixture_localtrend")
  expect_true(all(fit$prec_1 > 0) && all(fit$prec_2 > 0))
  expect_true(all(fit$prec_theta1 > 0) && all(fit$prec_theta2 > 0))
  expect_equal(attr(fit, "prior_prec_phi1")$type, "halft")
  expect_equal(attr(fit, "prior_prec_phi1")$df, 3)
  expect_equal(attr(fit, "prior_prec_theta2")$type, "gamma")
})

test_that("mixture localacceleration: Half-Cauchy everywhere (probit)", {
  y <- make_mix()
  fit <- mcmc_normal_mixture_localacceleration(
    y, link = "probit", CTRL$burnin, CTRL$thinning, CTRL$n_draws,
    prior_prec01_type = "halfcauchy", prior_prec01_scale = 2,
    prior_prec02_type = "halfcauchy", prior_prec02_scale = 2,
    prior_prec1_type  = "halfcauchy", prior_prec1_scale  = 1,
    prior_prec2_type  = "halfcauchy", prior_prec2_scale  = 1,
    prior_prec3_type  = "halfcauchy", prior_prec3_scale  = 1,
    verbose = FALSE, seed = 1
  )
  expect_s3_class(fit, "normal_mixture_localacceleration")
  expect_true(all(fit$prec_1 > 0) && all(fit$prec_2 > 0))
  expect_true(all(fit$prec_theta1 > 0) && all(fit$prec_theta2 > 0) &&
                all(fit$prec_theta3 > 0))
  expect_equal(attr(fit, "prior_prec_phi2")$type, "halfcauchy")
  expect_equal(attr(fit, "prior_prec_theta3")$type, "halfcauchy")
})

test_that("mixture: invalid Half-t specifications are rejected", {
  y <- make_mix()
  # Half-Cauchy on phi_1 without a scale.
  expect_error(
    mcmc_normal_mixture_locallevel(
      y, link = "probit", CTRL$burnin, CTRL$thinning, CTRL$n_draws,
      prior_prec01_type = "halfcauchy", verbose = FALSE, seed = 1),
    "scale"
  )
  # "halfcauchy" on W_1 contradicted by df != 1.
  expect_error(
    mcmc_normal_mixture_locallevel(
      y, link = "probit", CTRL$burnin, CTRL$thinning, CTRL$n_draws,
      prior_prec1_type = "halfcauchy", prior_prec1_scale = 1, prior_prec1_df = 2,
      verbose = FALSE, seed = 1),
    "df = 1"
  )
})
