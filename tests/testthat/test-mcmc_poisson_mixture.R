library(testthat)

# The Poisson mixture family: two-component Poisson observations whose mixture
# weight follows a polynomial dynamic model. The dynamic-weight machinery is
# shared with the Gaussian mixture and is covered by that family's tests, so
# these concentrate on what is new -- the conjugate Gamma component step, the
# lambda_1 < lambda_2 identification, the count-specific validation, and the
# wiring of the three orders through to the fitted object.

make_counts <- function(n = 120, seed = 1) {
  set.seed(seed)
  alpha <- (sin(2 * pi * seq_len(n) / n) + 2) / 4
  z <- rbinom(n, 1, alpha)
  list(y = rpois(n, (1 - z) * 2 + z * 10), z = z, alpha = alpha)
}

CTRL <- list(burnin = 60, thinning = 1, n_draws = 80)

fit_level <- function(y, ...) {
  mcmc_poisson_mixture_locallevel(
    y, link = "logit", CTRL$burnin, CTRL$thinning, CTRL$n_draws,
    verbose = FALSE, seed = 1, ...
  )
}


# ---------------------------------------------------------------------------
# Object shape, across the three orders and both links
# ---------------------------------------------------------------------------

test_that("each order returns its own components under both links", {
  y <- make_counts()$y

  expected <- list(
    locallevel = c("lambda_1", "lambda_2", "theta_1", "theta_01",
                   "prec_theta1", "alpha", "z"),
    localtrend = c("lambda_1", "lambda_2", "theta_1", "theta_2",
                   "theta_01", "theta_02", "prec_theta1", "prec_theta2",
                   "alpha", "z"),
    localacceleration = c("lambda_1", "lambda_2", "theta_1", "theta_2",
                          "theta_3", "theta_01", "theta_02", "theta_03",
                          "prec_theta1", "prec_theta2", "prec_theta3",
                          "alpha", "z")
  )

  generators <- list(
    locallevel        = mcmc_poisson_mixture_locallevel,
    localtrend        = mcmc_poisson_mixture_localtrend,
    localacceleration = mcmc_poisson_mixture_localacceleration
  )

  for (order in names(generators)) {
    for (link in c("logit", "probit")) {
      fit <- generators[[order]](
        y, link = link, CTRL$burnin, CTRL$thinning, CTRL$n_draws,
        verbose = FALSE, seed = 2
      )
      expect_s3_class(fit, paste0("poisson_mixture_", order))
      expect_s3_class(fit, "pdm_mcmc")
      expect_identical(names(fit), expected[[order]])
      expect_identical(attr(fit, "model_type"), order)
      expect_identical(attr(fit, "link"), link)
      expect_true(all(is.finite(unlist(fit))))
    }
  }
})


test_that("the diagnostics are returned only for the logit link", {
  y <- make_counts()$y

  logit <- fit_level(y, return_log_sigma = TRUE, return_accept_prop = TRUE)
  expect_true(is.matrix(logit$log_sigma))
  expect_true(is.matrix(logit$accept_prop))
  expect_true(all(logit$accept_prop >= 0 & logit$accept_prop <= 1))

  # The probit sampler is pure Gibbs, so the flags are ignored with a warning.
  expect_warning(
    probit <- mcmc_poisson_mixture_locallevel(
      y, link = "probit", CTRL$burnin, CTRL$thinning, CTRL$n_draws,
      return_accept_prop = TRUE, verbose = FALSE, seed = 1
    ),
    "ignored when link = 'probit'"
  )
  expect_null(probit$accept_prop)
})


# ---------------------------------------------------------------------------
# The component step
# ---------------------------------------------------------------------------

test_that("the component rates are positive and correctly labelled", {
  fit <- fit_level(make_counts()$y)

  expect_true(all(fit$lambda_1 > 0))
  expect_true(all(fit$lambda_2 > 0))
  # The ordering constraint is what identifies the components; it must hold in
  # every draw, not merely on average.
  expect_true(all(fit$lambda_1 <= fit$lambda_2))
})


test_that("the sampler recovers well-separated component rates", {
  d <- make_counts(n = 200, seed = 3)
  fit <- mcmc_poisson_mixture_locallevel(
    d$y, link = "logit", burnin = 1000, thinning = 10, n_draws = 500,
    prior_theta01_prec = 1, prior_prec1_shape = 100, prior_prec1_rate = 1,
    verbose = FALSE, seed = 456
  )

  expect_equal(median(fit$lambda_1), 2, tolerance = 0.3)
  expect_equal(median(fit$lambda_2), 10, tolerance = 0.3)
  # The indicators should agree with the truth on the great majority of points.
  expect_gt(mean((colMeans(fit$z) > 0.5) == d$z), 0.9)
})


test_that("an empty component falls back on its prior rather than failing", {
  # Every count identical: the indicators can leave one component with no
  # observation at all, and the Gamma posterior then collapses to the Gamma
  # prior. Nothing here should be non-finite or non-positive.
  y <- rep(4L, 60)
  fit <- fit_level(y)

  expect_true(all(is.finite(fit$lambda_1)) && all(fit$lambda_1 > 0))
  expect_true(all(is.finite(fit$lambda_2)) && all(fit$lambda_2 > 0))
})


test_that("the indicator step survives counts far from both rates", {
  # Both Poisson densities underflow to zero on the original scale at these
  # magnitudes, which is why the indicator sampler normalises in logs. Without
  # that, the weights would come out 0/0.
  d <- make_counts()
  y <- d$y
  y[1] <- 400L                      # P(Y = 400 | lambda = 10) is ~1e-450

  fit <- fit_level(y)
  expect_true(all(fit$z %in% c(0, 1)))
  expect_false(anyNA(fit$z))
})


# ---------------------------------------------------------------------------
# Priors
# ---------------------------------------------------------------------------

test_that("the component rate defaults are centred on the count quartiles", {
  d <- make_counts()
  fit <- fit_level(d$y)

  targets <- pmax(as.numeric(quantile(d$y, c(0.25, 0.75))), 0.5)
  expect_equal(attr(fit, "prior_lambda_1")$shape, 2)
  expect_equal(attr(fit, "prior_lambda_1")$rate, 2 / targets[1])
  expect_equal(attr(fit, "prior_lambda_2")$shape, 2)
  expect_equal(attr(fit, "prior_lambda_2")$rate, 2 / targets[2])

  # An explicit rate overrides the data-scaled default. The larger rate goes to
  # component 1, which is the low-rate component: a Gamma's mean is shape/rate,
  # so ordering the rates the other way would contradict the constraint (and
  # warn, as the test below checks).
  fixed <- fit_level(d$y, prior_lambda01_rate = 7, prior_lambda02_rate = 3)
  expect_equal(attr(fixed, "prior_lambda_1")$rate, 7)
  expect_equal(attr(fixed, "prior_lambda_2")$rate, 3)

  # A changed shape carries into the default rate, keeping the prior mean.
  shaped <- fit_level(d$y, prior_lambda01_shape = 6)
  expect_equal(attr(shaped, "prior_lambda_1")$rate, 6 / targets[1])
})


test_that("a zero lower quartile is floored rather than dividing by zero", {
  # A count series with many zeros: quantile(y, 0.25) is exactly 0, and an
  # unfloored target would send the Gamma rate to infinity.
  set.seed(4)
  y <- rpois(80, c(rep(0.05, 60), rep(6, 20)))
  expect_equal(as.numeric(quantile(y, 0.25)), 0)

  fit <- fit_level(y)
  expect_true(is.finite(attr(fit, "prior_lambda_1")$rate))
  expect_equal(attr(fit, "prior_lambda_1")$rate, 2 / 0.5)
})


test_that("priors that contradict the ordering constraint warn", {
  y <- make_counts()$y

  # Prior mean of component 1 (10/1) above that of component 2 (2/1).
  expect_warning(
    fit_level(y, prior_lambda01_shape = 10, prior_lambda01_rate = 1,
              prior_lambda02_shape = 2, prior_lambda02_rate = 1),
    "lambda_1 < lambda_2"
  )

  # Equal prior means are the exchangeable specification and are left alone.
  expect_silent(
    fit_level(y, prior_lambda01_shape = 2, prior_lambda01_rate = 1,
              prior_lambda02_shape = 2, prior_lambda02_rate = 1)
  )
})


test_that("the state innovation precision takes a Half-t as well as a Gamma", {
  y <- make_counts()$y

  # Default is the Half-Cauchy, as for every other link family since 0.5-0.
  default <- fit_level(y)
  expect_equal(attr(default, "prior_prec_theta1")$type, "halfcauchy")
  expect_equal(attr(default, "prior_prec_theta1")$scale, 2)

  # A bare shape/rate pair still reads as a Gamma, so older code keeps working.
  gamma_fit <- fit_level(y, prior_prec1_shape = 100, prior_prec1_rate = 1)
  expect_equal(attr(gamma_fit, "prior_prec_theta1")$type, "gamma")
  expect_true(all(gamma_fit$prec_theta1 > 0))

  halft <- fit_level(y, prior_prec1_type = "halft", prior_prec1_scale = 1,
                     prior_prec1_df = 3)
  expect_equal(attr(halft, "prior_prec_theta1")$type, "halft")
  expect_equal(attr(halft, "prior_prec_theta1")$df, 3)
  expect_true(all(halft$prec_theta1 > 0))
})


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

test_that("non-count data is rejected", {
  expect_error(fit_level(c(1, 2, -3, 4)), "non-negative integers")
  expect_error(fit_level(c(1, 2.5, 3, 4)), "non-negative integers")
  expect_error(fit_level(c(1, NA, 3, 4)), "finite")
  expect_error(fit_level(c("a", "b", "c")), "numeric vector")
  expect_error(fit_level(c(1L, 2L)), "at least 3 observations")
})


# ---------------------------------------------------------------------------
# Starting values
# ---------------------------------------------------------------------------

test_that("init pins the rates and records where the chain started", {
  y <- make_counts()$y

  fit <- fit_level(y, init = list(lambda_1 = 2, lambda_2 = 9, theta_01 = 0.5))
  start <- attr(fit, "init")
  expect_equal(start$lambda_1, 2)
  expect_equal(start$lambda_2, 9)
  expect_equal(start$theta_01, 0.5)
  # prec_theta1 was left to the prior, so it is present but not pinned.
  expect_true(is.finite(start$prec_theta1) && start$prec_theta1 > 0)
})


test_that("init refuses starting rates in the wrong order", {
  y <- make_counts()$y

  expect_error(
    fit_level(y, init = list(lambda_1 = 9, lambda_2 = 2)),
    "requires `lambda_1` <= `lambda_2`"
  )
  expect_error(fit_level(y, init = list(lambda_1 = -1)), "positive")
  expect_error(fit_level(y, init = list(mu_1 = 1)), "unknown name")
})


test_that("init reaches every order", {
  y <- make_counts()$y

  trend <- mcmc_poisson_mixture_localtrend(
    y, link = "logit", CTRL$burnin, CTRL$thinning, CTRL$n_draws,
    init = list(theta_02 = 0.1, prec_theta2 = 5), verbose = FALSE, seed = 1
  )
  expect_equal(attr(trend, "init")$theta_02, 0.1)
  expect_equal(attr(trend, "init")$prec_theta2, 5)

  accel <- mcmc_poisson_mixture_localacceleration(
    y, link = "logit", CTRL$burnin, CTRL$thinning, CTRL$n_draws,
    init = list(theta_03 = -0.2, prec_theta3 = 4), verbose = FALSE, seed = 1
  )
  expect_equal(attr(accel, "init")$theta_03, -0.2)
  expect_equal(attr(accel, "init")$prec_theta3, 4)
})


# ---------------------------------------------------------------------------
# Reproducibility and multiple chains
# ---------------------------------------------------------------------------

test_that("a seed reproduces a fit exactly", {
  y <- make_counts()$y
  a <- fit_level(y)
  b <- fit_level(y)
  expect_equal(a$lambda_1, b$lambda_1)
  expect_equal(a$theta_1, b$theta_1)
})


test_that("chains > 1 returns a pdm_mcmc_list the diagnostics accept", {
  y <- make_counts()$y

  fits <- mcmc_poisson_mixture_locallevel(
    y, link = "logit", CTRL$burnin, CTRL$thinning, CTRL$n_draws,
    chains = 3, seed = 99
  )
  expect_s3_class(fits, "pdm_mcmc_list")
  expect_length(fits, 3)

  # The diagnostic reads the family's scalar parameters off get_param_config(),
  # so this is what says the new family is wired into it: four scalars, named
  # for the rates rather than for a Gaussian mixture's means and precisions.
  conv <- mcmc_convergence(fits)
  expect_identical(conv$table$Parameter[seq_len(4)],
                   c("lambda_1", "lambda_2", "theta_01", "W_1^{-1}"))
  expect_true(all(is.finite(conv$table$Rhat)))
})


# ---------------------------------------------------------------------------
# log_lik
# ---------------------------------------------------------------------------

test_that("log_lik marginalises the indicator and matches a direct computation", {
  y <- make_counts()$y
  fit <- fit_level(y)

  ll <- log_lik(fit)
  expect_equal(dim(ll), c(CTRL$n_draws, length(y)))
  expect_true(all(is.finite(ll)))

  # Recompute one entry by hand: the two-component mixture mass at (draw, t).
  s <- 7L
  t <- 11L
  manual <- log(
    (1 - fit$alpha[s, t]) * dpois(y[t], fit$lambda_1[s]) +
      fit$alpha[s, t] * dpois(y[t], fit$lambda_2[s])
  )
  expect_equal(ll[s, t], manual, tolerance = 1e-10)
})
