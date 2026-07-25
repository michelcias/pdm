library(testthat)

# A fitted object records what produced it: the package version, the seed, and
# every prior after resolution. The point is that a saved .rds can be reproduced
# on a later version even if the defaults have moved again -- which they have,
# four times, between 0.1-0 and 0.5-0.

test_that("a fit records its version and seed", {
  set.seed(1)
  y <- cumsum(c(10, rnorm(80)))[-1] + rnorm(80, sd = 0.4)
  f <- mcmc_normal_locallevel(y, 200, 2, 100,
                              prior_theta01_mean = y[1],
                              prior_theta01_prec = 1 / var(y),
                              verbose = FALSE, seed = 456)

  expect_equal(attr(f, "pdm_version"),
               as.character(utils::packageVersion("pdm")))
  expect_equal(attr(f, "seed"), 456)

  # A NULL seed is recorded as NULL, not dropped or coerced.
  g <- mcmc_normal_locallevel(y, 200, 2, 100,
                              prior_theta01_mean = y[1],
                              prior_theta01_prec = 1 / var(y),
                              verbose = FALSE)
  expect_null(attr(g, "seed"))
})


test_that("the recorded priors are the resolved ones, not the call", {
  set.seed(1)
  y <- cumsum(c(10, rnorm(80)))[-1] + rnorm(80, sd = 0.4)
  f <- mcmc_normal_locallevel(y, 200, 2, 100,
                              prior_theta01_mean = y[1],
                              prior_theta01_prec = 1 / var(y),
                              verbose = FALSE, seed = 1)
  pr <- attr(f, "priors")

  # The scales were never named in the call; they came from the data. Recording
  # the call would have lost them entirely.
  expect_equal(pr$prior_prec1_scale, innovation_prior_scale(y, 1))
  expect_equal(pr$prior_prec_y_scale, observation_prior_scale(y))
  expect_equal(pr$prior_prec1_type, "halfcauchy")

  # Arguments the chosen prior does not use are dropped rather than recorded as
  # NULL, so the list can be spliced back into a call as-is.
  expect_false("prior_prec1_shape" %in% names(pr))
  expect_false("prior_prec1_rate" %in% names(pr))

  # The initial-state priors are included; they were not recorded before.
  expect_equal(pr$prior_theta01_mean, y[1])
  expect_equal(pr$prior_theta01_prec, 1 / var(y))
})


test_that("a fit can be reproduced from its own attributes", {
  # The reason the provenance exists. Splice the recorded priors back into a
  # fresh call and the draws must come out identical.
  roundtrip <- function(fit, sampler, extra = list()) {
    args <- c(list(attr(fit, "y")), extra,
              list(attr(fit, "burnin"), attr(fit, "thinning"),
                   attr(fit, "n_chain")),
              attr(fit, "priors"),
              list(verbose = FALSE, seed = attr(fit, "seed")))
    again <- do.call(sampler, args)
    expect_identical(fit[names(fit)], again[names(again)])
  }

  set.seed(1)
  n <- 80
  p <- plogis(cumsum(rnorm(n, sd = 0.2)))

  roundtrip(
    mcmc_normal_locallevel(
      cumsum(c(10, rnorm(n)))[-1] + rnorm(n, sd = 0.4), 200, 2, 100,
      prior_theta01_mean = 10, prior_theta01_prec = 1, verbose = FALSE, seed = 3),
    mcmc_normal_locallevel)

  roundtrip(
    mcmc_binomial_locallevel(
      rbinom(n, 15, p), n_trials = 15, 200, 2, 100,
      prior_theta01_mean = 0, prior_theta01_prec = 1, verbose = FALSE, seed = 4),
    mcmc_binomial_locallevel, list(n_trials = 15))

  roundtrip(
    mcmc_poisson_locallevel(
      rpois(n, exp(0.5 + cumsum(rnorm(n, sd = 0.1)))), 200, 2, 100,
      prior_theta01_mean = 0, prior_theta01_prec = 1, verbose = FALSE, seed = 5),
    mcmc_poisson_locallevel)

  roundtrip(
    mcmc_normal_mixture_locallevel(
      ifelse(rbinom(n, 1, p) == 1, rnorm(n, 3, 0.5), rnorm(n, 0, 0.5)),
      link = "logit", 200, 2, 100, verbose = FALSE, seed = 6),
    mcmc_normal_mixture_locallevel, list(link = "logit"))
})


test_that("higher orders record every innovation prior", {
  set.seed(1)
  y <- cumsum(c(10, rnorm(80)))[-1] + rnorm(80, sd = 0.4)

  la <- mcmc_normal_localacceleration(
    y, 200, 2, 100,
    prior_theta01_mean = y[1], prior_theta01_prec = 1 / var(y),
    prior_theta02_mean = 0,    prior_theta02_prec = 1 / var(y),
    prior_theta03_mean = 0,    prior_theta03_prec = 1 / var(y),
    verbose = FALSE, seed = 1
  )
  pr <- attr(la, "priors")

  for (k in 1:3) {
    expect_equal(pr[[paste0("prior_prec", k, "_scale")]],
                 innovation_prior_scale(y, k))
    expect_equal(pr[[paste0("prior_theta0", k, "_prec")]], 1 / var(y))
  }
})


test_that("provenance travels through a multi-chain fit", {
  set.seed(1)
  y <- cumsum(c(10, rnorm(80)))[-1] + rnorm(80, sd = 0.4)
  fits <- mcmc_normal_locallevel(y, 200, 2, 100,
                                 prior_theta01_mean = y[1],
                                 prior_theta01_prec = 1 / var(y),
                                 verbose = FALSE, seed = 11, chains = 3)

  # Each chain carries its own seed, which is what makes the set reproducible.
  seeds <- attr(fits, "seeds")
  for (i in seq_along(fits)) {
    expect_equal(attr(fits[[i]], "seed"), seeds[i])
    expect_equal(attr(fits[[i]], "pdm_version"),
                 as.character(utils::packageVersion("pdm")))
  }

  # Pooling keeps the provenance of the chains it was built from.
  pooled <- pool_chains(fits)
  expect_equal(attr(pooled, "pdm_version"),
               as.character(utils::packageVersion("pdm")))
  expect_equal(attr(pooled, "priors"), attr(fits[[1L]], "priors"))
})
