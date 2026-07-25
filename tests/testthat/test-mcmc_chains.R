library(testthat)

# Tests for the multi-chain interface (`chains` argument on the Gaussian
# samplers) and for the `mcmc_convergence()` method it enables.

make_y <- function(n = 120, seed = 5) {
  set.seed(seed)
  cumsum(c(10, rnorm(n)))[-1] + rnorm(n, sd = sqrt(1 / 5))
}

# Priors shared by every fit below; the sampler arguments that vary are passed
# through `...`.
run_ll <- function(y, ...) {
  mcmc_normal_locallevel(
    y,
    prior_theta01_mean = y[1],
    prior_theta01_prec = 1 / var(y),
    prior_prec1_shape  = 1e-2,
    prior_prec1_rate   = 1e-2,
    prior_prec_y_shape = 1e-2,
    prior_prec_y_rate  = 1e-2,
    ...
  )
}


test_that("chains = 1 is identical to omitting the argument", {
  y <- make_y()

  without <- run_ll(y, burnin = 200, thinning = 2, n_chain = 200, seed = 7)
  with_1  <- run_ll(y, burnin = 200, thinning = 2, n_chain = 200, seed = 7,
                    chains = 1)

  expect_identical(without, with_1)
  expect_s3_class(without, "normal_locallevel")
  expect_false(inherits(without, "pdm_mcmc_list"))
})


test_that("chains > 1 returns a validated pdm_mcmc_list", {
  y <- make_y()
  fits <- run_ll(y, burnin = 200, thinning = 2, n_chain = 200, seed = 7,
                 chains = 3)

  expect_s3_class(fits, "pdm_mcmc_list")
  expect_length(fits, 3L)
  expect_equal(attr(fits, "chains"), 3L)
  expect_length(attr(fits, "seeds"), 3L)

  # Every element is an ordinary fit that the single-chain methods still accept.
  expect_true(all(vapply(fits, inherits, logical(1L), what = "normal_locallevel")))
  expect_s3_class(mcmc_convergence(fits[[1L]]), "pdm_convergence")

  # Metadata is lifted from the chains.
  expect_equal(attr(fits, "n_chain"), 200L)
  expect_equal(attr(fits, "burnin"), 200L)
  expect_equal(attr(fits, "n_obs"), length(y))
  expect_equal(attr(fits, "model_type"), "locallevel")
})


test_that("the chains actually differ from one another", {
  y <- make_y()
  fits <- run_ll(y, burnin = 200, thinning = 2, n_chain = 200, seed = 7,
                 chains = 3)

  expect_false(identical(fits[[1L]]$prec_y, fits[[2L]]$prec_y))
  expect_false(identical(fits[[1L]]$theta_01, fits[[2L]]$theta_01))
  expect_equal(length(unique(attr(fits, "seeds"))), 3L)
})


test_that("a master seed reproduces the whole set, in parallel or not", {
  y <- make_y()
  args <- list(y = y, burnin = 100, thinning = 1, n_chain = 150, seed = 42,
               chains = 2)

  a <- do.call(run_ll, args)
  b <- do.call(run_ll, args)
  expect_identical(a, b)

  # Each chain seeds itself, so the execution mode cannot change the result.
  skip_on_os("windows")
  p <- do.call(run_ll, c(args, parallel = TRUE))
  expect_identical(a, p)
})


test_that("invalid chains / parallel arguments are rejected", {
  y <- make_y(n = 40)

  expect_error(run_ll(y, burnin = 10, thinning = 1, n_chain = 20, chains = 0),
               "positive integer")
  expect_error(run_ll(y, burnin = 10, thinning = 1, n_chain = 20, chains = 2.5),
               "positive integer")
  expect_error(run_ll(y, burnin = 10, thinning = 1, n_chain = 20, chains = c(2, 3)),
               "positive integer")
  expect_error(run_ll(y, burnin = 10, thinning = 1, n_chain = 20, chains = 2,
                      parallel = "yes"),
               "single logical")
})


test_that("mcmc_convergence() on a pdm_mcmc_list reports R-hat and ESS", {
  y <- make_y()
  fits <- run_ll(y, burnin = 500, thinning = 2, n_chain = 400, seed = 11,
                 chains = 4)

  conv <- mcmc_convergence(fits)

  expect_s3_class(conv, "pdm_convergence_multi")
  expect_true(all(c("Parameter", "Rhat", "ESS_bulk", "ESS_tail", "Overall")
                  %in% names(conv$table)))
  expect_equal(conv$chains, 4L)
  expect_true(all(is.finite(conv$table$Rhat)))
  expect_true(all(conv$table$Rhat > 0.9))

  # Scalar parameters plus three time points of the single state trajectory.
  expect_true(all(c("theta_01", "W_1^{-1}", "V^{-1}") %in% conv$table$Parameter))
  expect_equal(sum(grepl("^theta_1\\[t=", conv$table$Parameter)), 3L)

  # theta_timepoints = NULL keeps the scalars only.
  scalars <- mcmc_convergence(fits, theta_timepoints = NULL)
  expect_equal(nrow(scalars$table), 3L)

  # show_* flags drop their columns.
  bare <- mcmc_convergence(fits, show_ess = FALSE, show_overall = FALSE)
  expect_equal(names(bare$table), c("Parameter", "Rhat"))
})


test_that("mcmc_convergence() rejects invalid thresholds", {
  y <- make_y(n = 40)
  fits <- run_ll(y, burnin = 50, thinning = 1, n_chain = 100, seed = 3,
                 chains = 2)

  expect_error(mcmc_convergence(fits, rhat_threshold = 0.9), "greater than 1")
  expect_error(mcmc_convergence(fits, ess_threshold = -1), "positive")
  expect_error(mcmc_convergence(fits, theta_timepoints = c(0, 0.5)), "in \\(0, 1\\)")
})


test_that("R-hat separates an under-burned run from a converged one", {
  # The point of the diagnostic is not that it runs but that it discriminates.
  # Both fits use the same data, priors and number of chains; only the burn-in
  # differs, so a flag on the first and none on the second is attributable to
  # the burn-in alone.
  y <- make_y()

  under <- mcmc_convergence(
    run_ll(y, burnin = 10, thinning = 1, n_chain = 200, seed = 99, chains = 4),
    theta_timepoints = NULL
  )
  # The innovation and observation precisions are the slowest-mixing parameters
  # of this model and need thinning of this order to clear the 1.01 cut-off;
  # burn-in alone is not enough.
  converged <- mcmc_convergence(
    run_ll(y, burnin = 5000, thinning = 30, n_chain = 2000, seed = 99, chains = 4),
    theta_timepoints = NULL
  )

  expect_gt(max(under$table$Rhat), 1.05)
  expect_true(any(under$table$Overall == "POOR"))

  expect_lt(max(converged$table$Rhat), 1.01)
  expect_true(all(converged$table$Overall == "GOOD"))
})


test_that("print methods run without error", {
  y <- make_y(n = 40)
  fits <- run_ll(y, burnin = 50, thinning = 1, n_chain = 100, seed = 3,
                 chains = 2)

  expect_output(print(fits), "Multi-chain pdm fit")
  expect_output(print(fits), "Chains:        2")
  expect_output(print(mcmc_convergence(fits)), "multi-chain")
})


test_that("chains works for every non-Gaussian family", {
  # Structural coverage of the phase-2 wiring: one model per family, tiny runs.
  set.seed(3)
  n <- 40
  p <- plogis(cumsum(rnorm(n, sd = 0.2)))

  gamma_w1 <- list(prior_prec1_shape = 1e-2, prior_prec1_rate = 1e-2)

  binom <- do.call(mcmc_binomial_locallevel, c(
    list(rbinom(n, 10, p), n_trials = 10, 20, 1, 40,
         prior_theta01_mean = 0, prior_theta01_prec = 1,
         verbose = FALSE, seed = 1, chains = 2), gamma_w1
  ))
  pois <- do.call(mcmc_poisson_locallevel, c(
    list(rpois(n, exp(0.5 + cumsum(rnorm(n, sd = 0.1)))), 20, 1, 40,
         prior_theta01_mean = 0, prior_theta01_prec = 1,
         verbose = FALSE, seed = 1, chains = 2), gamma_w1
  ))
  probit <- do.call(mcmc_probit_bernoulli_locallevel, c(
    list(rbinom(n, 1, p), 20, 1, 40,
         prior_theta01_mean = 0, prior_theta01_prec = 1,
         verbose = FALSE, seed = 1, chains = 2), gamma_w1
  ))
  mixture <- mcmc_normal_mixture_locallevel(
    ifelse(rbinom(n, 1, p) == 1, rnorm(n, 3, 0.5), rnorm(n, 0, 0.5)),
    link = "logit", 20, 1, 40, verbose = FALSE, seed = 1, chains = 2
  )

  for (fit in list(binom, pois, probit, mixture)) {
    expect_s3_class(fit, "pdm_mcmc_list")
    expect_length(fit, 2L)
    expect_false(identical(fit[[1L]]$theta_1, fit[[2L]]$theta_1))
    expect_s3_class(mcmc_convergence(fit, theta_timepoints = 0.5),
                    "pdm_convergence_multi")
  }
})


test_that("R-hat has power on a non-Gaussian family too (phase-2 init)", {
  # Until the phase-2 fix these samplers started every chain's trajectory at a
  # hard-coded theta = 0, which suppressed the between-chain variance R-hat
  # depends on. With the fix an under-burned run must be flagged, exactly as it
  # already was for the Gaussian family.
  set.seed(7)
  n <- 60
  y <- rbinom(n, 10, plogis(cumsum(rnorm(n, sd = 0.2))))

  under <- mcmc_convergence(
    mcmc_binomial_locallevel(
      y, n_trials = 10, burnin = 50, thinning = 1, n_chain = 300,
      prior_theta01_mean = 0, prior_theta01_prec = 1,
      prior_prec1_shape = 1e-2, prior_prec1_rate = 1e-2,
      verbose = FALSE, seed = 2024, chains = 4
    ),
    theta_timepoints = NULL
  )

  expect_gt(max(under$table$Rhat), 1.05)
  expect_true(any(under$table$Overall == "POOR"))
})


test_that("chains works for the trend and acceleration samplers too", {
  set.seed(9)
  n  <- 100
  th2 <- cumsum(rnorm(n, sd = 0.1))
  th1 <- cumsum(th2) + 5
  y   <- th1 + rnorm(n, sd = 0.5)

  trend <- mcmc_normal_localtrend(
    y,
    burnin             = 200,
    thinning           = 2,
    n_chain            = 200,
    prior_theta01_mean = y[1],
    prior_theta01_prec = 1 / var(y),
    prior_theta02_mean = 0,
    prior_theta02_prec = 1 / var(y),
    prior_prec1_shape  = 1e-2,
    prior_prec1_rate   = 1e-2,
    prior_prec2_shape  = 1e-2,
    prior_prec2_rate   = 1e-2,
    prior_prec_y_shape = 1e-2,
    prior_prec_y_rate  = 1e-2,
    chains             = 2,
    seed               = 4
  )

  expect_s3_class(trend, "pdm_mcmc_list")
  conv <- mcmc_convergence(trend, theta_timepoints = 0.5)
  # Two states x one time point, plus the five scalars.
  expect_equal(sum(grepl("^theta_[12]\\[t=", conv$table$Parameter)), 2L)

  accel <- mcmc_normal_localacceleration(
    y,
    burnin             = 200,
    thinning           = 2,
    n_chain            = 200,
    prior_theta01_mean = y[1],
    prior_theta01_prec = 1 / var(y),
    prior_theta02_mean = 0,
    prior_theta02_prec = 1 / var(y),
    prior_theta03_mean = 0,
    prior_theta03_prec = 1 / var(y),
    prior_prec1_shape  = 1e-2,
    prior_prec1_rate   = 1e-2,
    prior_prec2_shape  = 1e-2,
    prior_prec2_rate   = 1e-2,
    prior_prec3_shape  = 1e-2,
    prior_prec3_rate   = 1e-2,
    prior_prec_y_shape = 1e-2,
    prior_prec_y_rate  = 1e-2,
    chains             = 2,
    seed               = 4
  )

  expect_s3_class(accel, "pdm_mcmc_list")
  expect_equal(sum(grepl("^theta_[123]\\[t=",
                         mcmc_convergence(accel, theta_timepoints = 0.5)$table$Parameter)),
               3L)
})


test_that("worker_count() never asks mclapply for a bad core count", {
  # parallel::detectCores() is documented as unsuitable for mc.cores directly:
  # it may return NA, and it counts the machine's CPUs rather than the ones
  # this process is allowed to use. mc.cores = NA is an error, so the NA path
  # is the one that would have taken a real run down.
  old <- options(mc.cores = NULL)
  on.exit(options(old), add = TRUE)

  local_mocked_bindings(detectCores = function(...) NA_integer_,
                        .package = "parallel")
  expect_equal(worker_count(4L), 2L)

  local_mocked_bindings(detectCores = function(...) 16L, .package = "parallel")
  expect_equal(worker_count(4L), 4L)   # never more workers than chains
  expect_equal(worker_count(64L), 16L) # nor more than the machine has

  # An explicit mc.cores wins over detectCores(): this is how a job script or
  # R CMD check caps a package, and how CRAN's two-core limit is honoured.
  options(mc.cores = 2L)
  expect_equal(worker_count(8L), 2L)
  expect_equal(worker_count(1L), 1L)

  # Nothing can drive it below one worker.
  options(mc.cores = 0L)
  expect_equal(worker_count(4L), 1L)
})
