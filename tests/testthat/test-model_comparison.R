library(testthat)

# Unit tests for the loo-based model-comparison layer: the waic()/loo() methods
# for pdm_mcmc objects, pdm_compare(), and pdm_lfo(). All require the suggested
# 'loo' package and are skipped when it is unavailable.

fit_ll <- function(y, seed = 1, ...) {
  mcmc_normal_locallevel(
    y, burnin = 50, thinning = 1, n_draws = 300,
    prior_theta01_mean = y[1], prior_theta01_prec = 1 / var(y),
    prior_prec1_shape  = 1e-2, prior_prec1_rate  = 1e-2,
    prior_prec_y_shape = 1e-2, prior_prec_y_rate = 1e-2,
    verbose = FALSE, seed = seed, ...
  )
}

make_series <- function(n = 60, seed = 7) {
  set.seed(seed)
  cumsum(rnorm(n)) + rnorm(n, sd = 0.4)
}


test_that("waic()/loo() methods dispatch and agree with the matrix form", {
  skip_if_not_installed("loo")
  fit <- fit_ll(make_series())

  w <- suppressWarnings(loo::waic(fit))
  l <- suppressWarnings(loo::loo(fit))
  expect_s3_class(w, "waic")
  expect_s3_class(l, "loo")

  # The method is exactly loo::waic() on the pointwise log-likelihood matrix.
  w2 <- suppressWarnings(loo::waic(log_lik(fit)))
  expect_equal(w$estimates, w2$estimates, tolerance = 1e-10)
})

test_that("pdm_compare ranks models and honours explicit names", {
  skip_if_not_installed("loo")
  y  <- make_series()
  m1 <- fit_ll(y, seed = 1)
  m2 <- fit_ll(y, seed = 2)

  cmp <- suppressWarnings(pdm_compare(m1, m2))
  expect_s3_class(cmp, "compare.loo")
  expect_equal(nrow(cmp), 2L)
  expect_equal(unname(cmp[1, "elpd_diff"]), 0)          # best model is the reference

  cmp2 <- suppressWarnings(pdm_compare(level = m1, alt = m2))
  expect_setequal(cmp2$model, c("level", "alt"))
})

test_that("pdm_compare validates its inputs", {
  skip_if_not_installed("loo")
  m1 <- fit_ll(make_series())
  expect_error(pdm_compare(m1), "at least two")
  expect_error(pdm_compare(m1, 42), "pdm_mcmc")
})

test_that("pdm_lfo: the last forecast equals PSIS-LOO for the final observation", {
  skip_if_not_installed("loo")
  y   <- make_series(n = 60)
  fit <- fit_ll(y)

  lfo <- suppressWarnings(pdm_lfo(fit, L = 20))
  expect_s3_class(lfo, "pdm_lfo")
  expect_named(lfo$pointwise, c("t", "elpd", "pareto_k"))
  expect_equal(lfo$n_pred, length(y) - 20L)

  # At t = N only y_N is dropped, so the reweighting is identical to leaving out
  # observation N -- i.e. PSIS-LOO for that point (same r_eff = NA).
  ll       <- log_lik(fit)
  loo_full <- suppressWarnings(loo::loo(ll, r_eff = NA))
  lfo_N    <- lfo$pointwise$elpd[lfo$pointwise$t == length(y)]
  expect_equal(lfo_N, unname(loo_full$pointwise[length(y), "elpd_loo"]),
               tolerance = 1e-8)
})

test_that("pdm_lfo validates L and warns on unreliable forecasts", {
  skip_if_not_installed("loo")
  fit <- fit_ll(make_series(n = 50))

  expect_error(pdm_lfo(fit, L = 50), "\\[1, n_obs")
  expect_error(pdm_lfo(fit, L = 0),  "\\[1, n_obs")

  # The no-refit estimator is expected to flag most forecasts -> a warning.
  expect_warning(pdm_lfo(fit, L = 10), "Pareto k")
})
