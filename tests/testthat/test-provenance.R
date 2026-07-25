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


test_that("every prior argument of every wrapper reaches the record", {
  # record_provenance() collects by naming convention: `prior_*` variables in
  # the wrapper's frame. A future prior argument named something else would
  # vanish from the record silently, and the round trip would still pass,
  # because it splices back whatever was recorded. This is the check that
  # would fail instead.
  skip_on_cran()

  set.seed(1)
  n  <- 50
  p  <- plogis(cumsum(rnorm(n, sd = 0.2)))
  yg <- cumsum(c(10, rnorm(n)))[-1] + rnorm(n, sd = 0.4)
  ym <- ifelse(rbinom(n, 1, p) == 1, rnorm(n, 3, 0.5), rnorm(n, 0, 0.5))

  # Initial-state priors are required arguments, so they must be supplied; the
  # precision priors are left at their defaults, which is the case under test.
  th <- function(order, mean1, prec1) {
    out <- list(prior_theta01_mean = mean1, prior_theta01_prec = prec1)
    if (order >= 2) out <- c(out, list(prior_theta02_mean = 0, prior_theta02_prec = prec1))
    if (order >= 3) out <- c(out, list(prior_theta03_mean = 0, prior_theta03_prec = prec1))
    out
  }
  ctrl <- list(80, 1, 60, verbose = FALSE, seed = 1)

  cases <- list()
  for (o in 1:3) {
    suf <- c("locallevel", "localtrend", "localacceleration")[o]
    cases[[paste0("normal_", suf)]] <- list(
      fun = get(paste0("mcmc_normal_", suf)),
      args = c(list(yg), ctrl, th(o, yg[1], 1 / var(yg))))
    cases[[paste0("binomial_", suf)]] <- list(
      fun = get(paste0("mcmc_binomial_", suf)),
      args = c(list(rbinom(n, 15, p), n_trials = 15), ctrl, th(o, 0, 1)))
    cases[[paste0("poisson_", suf)]] <- list(
      fun = get(paste0("mcmc_poisson_", suf)),
      args = c(list(rpois(n, exp(0.5 + cumsum(rnorm(n, sd = 0.1))))), ctrl, th(o, 0, 1)))
    cases[[paste0("probit_", suf)]] <- list(
      fun = get(paste0("mcmc_probit_bernoulli_", suf)),
      args = c(list(rbinom(n, 1, p)), ctrl, th(o, 0, 1)))
    cases[[paste0("mixture_", suf)]] <- list(
      fun = get(paste0("mcmc_normal_mixture_", suf)),
      args = c(list(ym, link = "logit"), ctrl))
  }

  expect_equal(length(cases), 15L)

  for (nm in names(cases)) {
    fit      <- do.call(cases[[nm]]$fun, cases[[nm]]$args)
    recorded <- names(attr(fit, "priors"))
    formals_ <- grep("^prior_", names(formals(cases[[nm]]$fun)), value = TRUE)

    # Nothing recorded that is not an argument of this wrapper.
    expect_true(all(recorded %in% formals_),
                info = paste(nm, ": stray", toString(setdiff(recorded, formals_))))

    # Everything missing must be an argument the resolved type does not use:
    # shape/rate under a Half-t, scale/df under a Gamma.
    for (miss in setdiff(formals_, recorded)) {
      stem <- sub("_(shape|rate|scale|df|type|mean|prec)$", "", miss)
      type <- attr(fit, "priors")[[paste0(stem, "_type")]]
      unused <- if (identical(type, "gamma")) c("scale", "df") else c("shape", "rate")
      expect_true(grepl(paste0("_(", paste(unused, collapse = "|"), ")$"), miss),
                  info = paste(nm, ": unexplained missing", miss))
    }
  }
})
