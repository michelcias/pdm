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
