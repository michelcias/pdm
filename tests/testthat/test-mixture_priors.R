library(testthat)

# The mixture samplers identify their components by enforcing mu_1 < mu_2,
# relabelling after each draw. Priors that ask for the reverse are not an error
# -- the run is still a valid draw from the constrained posterior -- but the
# relabelling then fires on nearly every iteration and nothing in the returned
# object shows it, so the wrapper warns.

make_y <- function(n = 80, seed = 7) {
  set.seed(seed)
  z <- rbinom(n, 1, 0.5)
  ifelse(z == 1, rnorm(n, 3, 0.5), rnorm(n, 0, 0.5))
}

fit_mix <- function(y, ...) {
  mcmc_normal_mixture_locallevel(
    y, link = "logit", 50, 1, 60, verbose = FALSE, seed = 1, ...
  )
}


test_that("rg_component_rate() implements the Richardson-Green scaling", {
  y <- c(0, 10)                       # range 10
  expect_equal(rg_component_rate(y, 2), 2 * 100 / 100)
  expect_equal(rg_component_rate(y, 5), 5 * 100 / 100)

  # Scales with the square of the range, and follows the shape.
  expect_equal(rg_component_rate(c(0, 20), 2), 4 * rg_component_rate(c(0, 10), 2))
  expect_equal(rg_component_rate(y, 4), 2 * rg_component_rate(y, 2))
})


test_that("the component-precision defaults follow Richardson-Green", {
  y <- make_y()
  expected <- 2 * diff(range(y))^2 / 100

  fit <- fit_mix(y)
  # The wrapper records the priors it actually used.
  expect_equal(attr(fit, "prior_prec_phi1")$shape, 2)
  expect_equal(attr(fit, "prior_prec_phi1")$rate, expected)
  expect_equal(attr(fit, "prior_prec_phi2")$shape, 2)
  expect_equal(attr(fit, "prior_prec_phi2")$rate, expected)

  # An explicit rate overrides the data-scaled default.
  fixed <- fit_mix(y, prior_prec01_rate = 3, prior_prec02_rate = 7)
  expect_equal(attr(fixed, "prior_prec_phi1")$rate, 3)
  expect_equal(attr(fixed, "prior_prec_phi2")$rate, 7)

  # A changed shape carries into the default rate, as in R&G's formula.
  shaped <- fit_mix(y, prior_prec01_shape = 4)
  expect_equal(attr(shaped, "prior_prec_phi1")$rate, 4 * diff(range(y))^2 / 100)
})


test_that("the Half-t path ignores the Gamma rate default", {
  y <- make_y()

  # `rate` is meaningless under Half-t; leaving it NULL must not error.
  fit <- fit_mix(y, prior_prec01_type = "halfcauchy", prior_prec01_scale = 1)
  expect_s3_class(fit, "normal_mixture_locallevel")
  expect_equal(attr(fit, "prior_prec_phi1")$type, "halfcauchy")
  # The other component keeps the Gamma default.
  expect_equal(attr(fit, "prior_prec_phi2")$shape, 2)
})


test_that("the new default suppresses the degenerate precision excursions", {
  skip_on_cran()
  # The reason for the change: a component collapsing onto a few observations
  # drives its precision up without limit, and the old Gamma(0.01, 0.01) barely
  # penalised that region.
  set.seed(1)
  n <- 300
  a <- plogis(cumsum(rnorm(n, sd = 0.2)))
  z <- rbinom(n, 1, a)
  y <- ifelse(z == 1, rnorm(n, 3, 1), rnorm(n, 0, 1))

  ctrl <- list(y, link = "logit", 2000, 10, 400, verbose = FALSE, seed = 2024)
  new_default <- do.call(mcmc_normal_mixture_locallevel, ctrl)
  old_default <- do.call(mcmc_normal_mixture_locallevel, c(ctrl, list(
    prior_prec01_shape = 0.01, prior_prec01_rate = 0.01,
    prior_prec02_shape = 0.01, prior_prec02_rate = 0.01
  )))

  expect_lt(max(new_default$prec_2), max(old_default$prec_2))
  expect_lt(max(new_default$prec_2), 100)
})


test_that("check_mixture_mu_priors() warns only when the order is reversed", {
  expect_warning(check_mixture_mu_priors(2, -2), "enforcing mu_1 < mu_2")
  expect_warning(check_mixture_mu_priors(0.5, 0.4), "prior_mu01_mean")

  # Correct order, and the exchangeable case, both pass silently.
  expect_silent(check_mixture_mu_priors(-2, 2))
  expect_silent(check_mixture_mu_priors(0, 0))
})


test_that("the warning names both values so the fix is obvious", {
  w <- tryCatch(check_mixture_mu_priors(2, -2), warning = conditionMessage)
  expect_match(w, "`prior_mu01_mean` \\(2\\)")
  expect_match(w, "`prior_mu02_mean` \\(-2\\)")
  expect_match(w, "Swap the two priors")
})


test_that("the mixture samplers warn on reversed component-mean priors", {
  y <- make_y()

  expect_warning(fit_mix(y, prior_mu01_mean = 2, prior_mu02_mean = -2),
                 "enforcing mu_1 < mu_2")

  # The fit still completes and is usable; this is a warning, not an error.
  fit <- suppressWarnings(
    fit_mix(y, prior_mu01_mean = 2, prior_mu02_mean = -2)
  )
  expect_s3_class(fit, "normal_mixture_locallevel")
  expect_true(all(is.finite(fit$mu_1)) && all(is.finite(fit$mu_2)))
  # The constraint is enforced regardless of what the priors asked for.
  expect_true(all(fit$mu_1 <= fit$mu_2))
})


test_that("well-specified and default priors do not warn", {
  y <- make_y()

  expect_no_warning(fit_mix(y, prior_mu01_mean = 0, prior_mu02_mean = 3))
  expect_no_warning(fit_mix(y, prior_mu01_mean = 1, prior_mu02_mean = 1))
  # Defaults are quantile(y, 0.25) and quantile(y, 0.75), so always ordered.
  expect_no_warning(fit_mix(y))
})


test_that("the check reaches the trend and acceleration samplers too", {
  y <- make_y()
  rev_priors <- list(prior_mu01_mean = 2, prior_mu02_mean = -2)

  expect_warning(
    do.call(mcmc_normal_mixture_localtrend,
            c(list(y, link = "logit", 50, 1, 60, verbose = FALSE, seed = 1),
              rev_priors)),
    "enforcing mu_1 < mu_2"
  )
  expect_warning(
    do.call(mcmc_normal_mixture_localacceleration,
            c(list(y, link = "logit", 50, 1, 60, verbose = FALSE, seed = 1),
              rev_priors)),
    "enforcing mu_1 < mu_2"
  )
})
