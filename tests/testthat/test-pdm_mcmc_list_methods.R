library(testthat)

# Methods that make a multi-chain fit usable wherever a single-chain fit is:
# pooling, summary, log_lik, waic/loo, pdm_compare and the plot methods.

make_y <- function(n = 80, seed = 5) {
  set.seed(seed)
  cumsum(c(10, rnorm(n)))[-1] + rnorm(n, sd = sqrt(1 / 5))
}

fit_ll <- function(y, ...) {
  mcmc_normal_locallevel(
    y,
    burnin             = 200,
    thinning           = 1,
    n_draws            = 150,
    prior_theta01_mean = y[1],
    prior_theta01_prec = 1 / var(y),
    prior_prec1_shape  = 1e-2,
    prior_prec1_rate   = 1e-2,
    prior_prec_y_shape = 1e-2,
    prior_prec_y_rate  = 1e-2,
    ...
  )
}


test_that("pool_chains() stacks the draws in chain order", {
  y     <- make_y()
  fits  <- fit_ll(y, seed = 1, chains = 3)
  pool  <- pool_chains(fits)

  expect_s3_class(pool, "normal_locallevel")
  expect_s3_class(pool, "pdm_mcmc")
  expect_false(inherits(pool, "pdm_mcmc_list"))

  # Vectors concatenate, matrices rbind, chain 1 first.
  expect_length(pool$prec_y, 450L)
  expect_equal(dim(pool$theta_1), c(450L, length(y)))
  expect_identical(pool$prec_y[1:150], fits[[1L]]$prec_y)
  expect_identical(pool$prec_y[151:300], fits[[2L]]$prec_y)
  expect_identical(pool$theta_1[1:150, ], fits[[1L]]$theta_1)

  # Attributes carry over, with the draw count corrected.
  expect_equal(attr(pool, "n_draws"), 450L)
  expect_equal(attr(pool, "n_obs"), attr(fits[[1L]], "n_obs"))
  expect_equal(attr(pool, "model_type"), "locallevel")
  expect_identical(attr(pool, "y"), y)
})


test_that("summary() and log_lik() delegate to the pooled fit", {
  y    <- make_y()
  fits <- fit_ll(y, seed = 1, chains = 3)

  s <- suppressWarnings(summary(fits))
  expect_s3_class(s, "summary.pdm_mcmc_list")
  expect_s3_class(s, "summary.normal_locallevel")
  expect_equal(s$n_draws, 450L)

  # The tables themselves are exactly the single-chain method's, on the pooled
  # draws; only the convergence fields and the class are added on top.
  pooled <- summary(pool_chains(fits))
  expect_identical(s$scalar_params, pooled$scalar_params)
  expect_identical(s$theta_summary, pooled$theta_summary)

  ll <- log_lik(fits)
  expect_equal(dim(ll), c(450L, length(y)))
  expect_true(all(is.finite(ll)))
  expect_identical(ll, log_lik(pool_chains(fits)))
})


test_that("summary() carries R-hat for every scalar parameter", {
  y    <- make_y()
  fits <- fit_ll(y, seed = 1, chains = 3)
  s    <- suppressWarnings(summary(fits))

  expect_equal(s$chains, 3L)
  expect_equal(s$n_draws_each, 150L)
  expect_named(s$rhat, c("V^{-1}", "theta_01", "W_1^{-1}"))
  expect_true(all(s$rhat > 0.9))

  # The same numbers mcmc_convergence() reports for those parameters.
  conv <- mcmc_convergence(fits, theta_timepoints = NULL)
  expect_equal(unname(round(s$rhat[conv$table$Parameter], 4)),
               conv$table$Rhat, tolerance = 1e-6)
})


test_that("summary() warns when the chains have not converged", {
  y <- make_y()

  # Ten iterations of burn-in: the chains cannot have agreed yet. Called
  # directly because fit_ll() fixes the MCMC controls.
  under <- mcmc_normal_locallevel(
    y, burnin = 10, thinning = 1, n_draws = 200,
    prior_theta01_mean = y[1], prior_theta01_prec = 1 / var(y),
    prior_prec1_shape = 1e-2, prior_prec1_rate = 1e-2,
    prior_prec_y_shape = 1e-2, prior_prec_y_rate = 1e-2,
    seed = 99, chains = 4
  )
  expect_warning(summary(under), "chains have not converged")
  expect_warning(summary(under), "max R-hat")

  s <- suppressWarnings(summary(under))
  expect_gt(max(s$rhat, na.rm = TRUE), 1.01)
  expect_output(print(s), "WARNING: max R-hat")
  expect_output(print(s), "no single posterior")

  # Raising the threshold above the observed value silences it.
  expect_no_warning(summary(under, rhat_threshold = 100))
})


test_that("summary() is silent and reassuring on a converged fit", {
  y <- make_y()
  ok <- mcmc_normal_locallevel(
    y, burnin = 5000, thinning = 30, n_draws = 400,
    prior_theta01_mean = y[1], prior_theta01_prec = 1 / var(y),
    prior_prec1_shape = 1e-2, prior_prec1_rate = 1e-2,
    prior_prec_y_shape = 1e-2, prior_prec_y_rate = 1e-2,
    seed = 99, chains = 4
  )

  expect_no_warning(summary(ok))
  s <- summary(ok)
  expect_lt(max(s$rhat, na.rm = TRUE), 1.01)
  expect_output(print(s), "pooling the chains is justified")
  expect_output(print(s), "Scalar R-hat")
})


test_that("summary() rejects an invalid rhat_threshold", {
  fits <- fit_ll(make_y(), seed = 1, chains = 2)
  expect_error(summary(fits, rhat_threshold = 0.9), "greater than 1")
  expect_error(summary(fits, rhat_threshold = c(1.01, 1.05)), "single numeric")
})


test_that("waic() and loo() work on a multi-chain fit", {
  skip_if_not_installed("loo")
  y    <- make_y()
  fits <- fit_ll(y, seed = 1, chains = 3)

  w <- suppressWarnings(loo::waic(fits))
  expect_s3_class(w, "waic")
  expect_true(is.finite(w$estimates["waic", "Estimate"]))

  l <- suppressWarnings(loo::loo(fits))
  expect_s3_class(l, "loo")
  expect_true(is.finite(l$estimates["looic", "Estimate"]))
})


test_that("waic() and loo() warn when the chains have not converged", {
  skip_if_not_installed("loo")
  y <- make_y()

  # The asymmetry this closes: summary() said so, waic() and loo() did not, and
  # a "loo" object gives no hint whether the draws behind it agreed. Model
  # comparison is where that does the most damage.
  under <- mcmc_normal_locallevel(
    y, burnin = 10, thinning = 1, n_draws = 200,
    prior_theta01_mean = y[1], prior_theta01_prec = 1 / var(y),
    prior_prec1_shape = 1e-2, prior_prec1_rate = 1e-2,
    prior_prec_y_shape = 1e-2, prior_prec_y_rate = 1e-2,
    seed = 99, chains = 4
  )

  # Filtered, because the loo package raises Pareto-k warnings of its own.
  ours <- function(expr) {
    hit <- FALSE
    withCallingHandlers(force(expr), warning = function(w) {
      if (grepl("have not converged", conditionMessage(w))) hit <<- TRUE
      invokeRestart("muffleWarning")
    })
    hit
  }

  expect_true(ours(loo::waic(under)))
  expect_true(ours(loo::loo(under)))

  # Raising the threshold above the observed value silences ours specifically.
  expect_false(ours(loo::waic(under, rhat_threshold = 100)))
  expect_false(ours(loo::loo(under, rhat_threshold = 100)))
})


test_that("loo() passes the real chain identifiers, not a single block", {
  skip_if_not_installed("loo")
  y    <- make_y()
  fits <- fit_ll(y, seed = 1, chains = 3)

  ll <- log_lik(fits)

  # What the method computes internally, against what the single-chain method
  # would have had to assume. Treating 3 chains as 1 changes the relative
  # efficiency, which is what the PSIS smoothing depends on.
  r_eff_multi  <- loo::relative_eff(exp(ll),
                                    chain_id = rep(1:3, each = 150L))
  r_eff_single <- loo::relative_eff(exp(ll),
                                    chain_id = rep(1L, nrow(ll)))
  expect_false(isTRUE(all.equal(r_eff_multi, r_eff_single)))

  l <- suppressWarnings(loo::loo(fits))
  expect_equal(l$diagnostics$r_eff, r_eff_multi, tolerance = 1e-8)
})


test_that("pdm_compare() accepts multi-chain fits", {
  skip_if_not_installed("loo")
  y     <- make_y()
  multi <- fit_ll(y, seed = 1, chains = 3)
  one   <- fit_ll(y, seed = 2)

  cmp <- suppressWarnings(pdm_compare(multi = multi, single = one))
  expect_true(is.matrix(cmp) || inherits(cmp, "compare.loo"))
  expect_equal(nrow(cmp), 2L)

  # Both multi-chain, and the waic criterion path.
  multi2 <- fit_ll(y, seed = 9, chains = 3)
  expect_silent(suppressWarnings(pdm_compare(a = multi, b = multi2)))
  expect_equal(nrow(suppressWarnings(
    pdm_compare(a = multi, b = multi2, criterion = "waic"))), 2L)
})


test_that("pdm_compare() does not mistake one multi-chain fit for a model list", {
  skip_if_not_installed("loo")
  # A pdm_mcmc_list is itself a list, so the "single list of models" branch
  # would otherwise treat its chains as competing models and compare a fit
  # against itself.
  fits <- fit_ll(make_y(), seed = 1, chains = 3)
  expect_error(pdm_compare(fits), "at least two fitted models")
})


test_that("true_value_for() maps display names to true_values keys", {
  tv <- list(prec_y = 5, prec_theta1 = 2, theta_01 = 10, prec_1 = 1, mu_2 = 3)

  expect_equal(true_value_for("V^{-1}", tv), 5)
  expect_equal(true_value_for("W_1^{-1}", tv), 2)
  expect_equal(true_value_for("theta_01", tv), 10)
  expect_equal(true_value_for("phi_1", tv), 1)
  expect_equal(true_value_for("mu_2", tv), 3)

  # Absent from the list, and unknown to the table.
  expect_null(true_value_for("W_3^{-1}", tv))
  expect_null(true_value_for("not_a_parameter", tv))
  expect_null(true_value_for("V^{-1}", NULL))
})


test_that("the single-chain plot still resolves true_values after the refactor", {
  # `true_value_for()` replaced an if-chain inside
  # plot_mcmc_diagnostics_generic(). No test covered that path, so this pins the
  # single-chain behaviour the refactor had to preserve, for every display name
  # the table knows.
  y   <- make_y()
  fit <- fit_ll(y, seed = 1)
  cfg <- get_param_config(fit)
  tv  <- list(prec_y = 5, prec_theta1 = 2, theta_01 = 10)

  expect_equal(
    vapply(cfg, function(p) true_value_for(p$name_str, tv), numeric(1L)),
    c(V_inv = 5, theta_01 = 10, W1_inv = 2)
  )

  pdf(NULL)
  on.exit(dev.off(), add = TRUE)
  expect_silent(plot(fit, type = "mcmc", which = 1, true_values = tv))
  expect_silent(plot(fit, type = "mcmc", true_values = tv))
  # An unrelated name in the list must be ignored, not matched by position.
  expect_silent(plot(fit, type = "mcmc", which = 1,
                     true_values = list(mu_1 = 0, prec_y = 5)))
})


test_that("plot() runs for every type the family accepts", {
  y    <- make_y()
  fits <- fit_ll(y, seed = 1, chains = 3)

  pdf(NULL)
  on.exit(dev.off(), add = TRUE)

  expect_silent(plot(fits, type = "mcmc", which = 1))
  expect_silent(plot(fits, type = "mcmc"))
  expect_silent(plot(fits, type = "states", ask = FALSE))
  expect_silent(plot(fits, type = "all", ask = FALSE))
  expect_silent(plot(fits, type = "mcmc", which = 1,
                     true_values = list(prec_y = 5)))

  # The return value is the input, invisibly.
  expect_identical(suppressWarnings(plot(fits, type = "mcmc", which = 1)), fits)
})


test_that("plot() rejects a type the family does not offer", {
  fits <- fit_ll(make_y(), seed = 1, chains = 2)

  pdf(NULL)
  on.exit(dev.off(), add = TRUE)

  # "alpha" exists for binomial/poisson but not for the Gaussian family.
  expect_error(plot(fits, type = "alpha"), "must be one of")
  expect_error(plot(fits, type = "nonsense"), "must be one of")
  expect_error(plot(fits, type = "mcmc", which = 99), "must index")
})


test_that("plot() covers the non-Gaussian families and their extra types", {
  skip_on_cran()
  set.seed(3)
  n <- 50
  p <- plogis(cumsum(rnorm(n, sd = 0.2)))

  pdf(NULL)
  on.exit(dev.off(), add = TRUE)

  bin <- mcmc_binomial_locallevel(
    rbinom(n, 10, p), n_trials = 10, 100, 1, 80,
    prior_theta01_mean = 0, prior_theta01_prec = 1,
    prior_prec1_shape = 1e-2, prior_prec1_rate = 1e-2,
    verbose = FALSE, seed = 1, chains = 2, return_accept_prop = TRUE
  )
  for (ty in c("mcmc", "states", "alpha", "acceptance", "all")) {
    expect_silent(plot(bin, type = ty, ask = FALSE))
  }

  # Without the acceptance diagnostic, type = "all" must skip it rather than
  # fail, matching the single-chain behaviour.
  bin_no_acc <- mcmc_binomial_locallevel(
    rbinom(n, 10, p), n_trials = 10, 100, 1, 80,
    prior_theta01_mean = 0, prior_theta01_prec = 1,
    prior_prec1_shape = 1e-2, prior_prec1_rate = 1e-2,
    verbose = FALSE, seed = 1, chains = 2
  )
  expect_silent(plot(bin_no_acc, type = "all", ask = FALSE))
  expect_error(plot(bin_no_acc, type = "acceptance"))

  mix <- mcmc_normal_mixture_locallevel(
    ifelse(rbinom(n, 1, p) == 1, rnorm(n, 3, 0.5), rnorm(n, 0, 0.5)),
    link = "logit", 100, 1, 80, verbose = FALSE, seed = 1, chains = 2
  )
  # "params" is the mixture-only page; the alpha page emits a rescaling note.
  expect_silent(plot(mix, type = "mcmc"))
  expect_silent(plot(mix, type = "params", ask = FALSE))
  # An 80-draw mixture fit has not converged, and summary() now says so; that
  # is checked elsewhere, so it is silenced here.
  expect_s3_class(suppressWarnings(summary(mix)),
                  "summary.normal_mixture_locallevel")
})


test_that("summary() screens the latent states, not only the scalars", {
  y    <- make_y()
  fits <- fit_ll(y, seed = 1, chains = 3)
  s    <- suppressWarnings(summary(fits))

  # Twenty time points along the single trajectory of a local-level model.
  expect_length(s$rhat_states, 20L)
  expect_true(all(grepl("^theta_1\\[t=[0-9]+\\]$", names(s$rhat_states))))
  expect_true(all(s$rhat_states > 0.9))

  # The same numbers mcmc_convergence() reports for those time points.
  conv <- mcmc_convergence(fits, theta_timepoints = seq(0.05, 0.95,
                                                        length.out = 20L))
  from_conv <- conv$table$Rhat[match(names(s$rhat_states),
                                     conv$table$Parameter)]
  expect_equal(unname(round(s$rhat_states, 4)), from_conv, tolerance = 1e-6)

  expect_output(print(s), "State R-hat")
})


test_that("an unconverged state warns even when every scalar looks fine", {
  # The hole this closes. summary() used to check only the scalar parameters,
  # documented as the slowest-mixing part of these models -- true of the
  # Gaussian family, false of the link families, where the states are slower.
  # A scalar-only screen therefore reported all-clear on fits whose
  # trajectories had not agreed.
  y    <- make_y()
  fits <- fit_ll(y, seed = 1, chains = 3)

  clean_scalars <- structure(c(`V^{-1}` = 1.001, theta_01 = 1.002), class = NULL)
  bad_state     <- c(`theta_1[t=40]` = 1.35)

  local_mocked_bindings(scalar_rhats = function(...) clean_scalars,
                        state_rhats  = function(...) bad_state)

  expect_warning(summary(fits), "chains have not converged")
  expect_warning(summary(fits), "1.35")

  s <- suppressWarnings(summary(fits))
  expect_output(print(s), "WARNING: max R-hat")

  # And the scalar table alone would have said nothing is wrong.
  expect_lt(max(s$rhat), 1.01)
})


test_that("waic() and loo() screen the states too", {
  skip_if_not_installed("loo")
  y    <- make_y()
  fits <- fit_ll(y, seed = 1, chains = 3)

  local_mocked_bindings(scalar_rhats = function(...) c(`V^{-1}` = 1.001),
                        state_rhats  = function(...) c(`theta_1[t=40]` = 1.35))

  ours <- function(expr) {
    hit <- FALSE
    withCallingHandlers(force(expr), warning = function(w) {
      if (grepl("have not converged", conditionMessage(w))) hit <<- TRUE
      invokeRestart("muffleWarning")
    })
    hit
  }

  expect_true(ours(loo::waic(fits)))
  expect_true(ours(loo::loo(fits)))
})


test_that("state_rhats() is empty for a model with no trajectory matrices", {
  # Guards the unlist()/names() path against a zero-length result, and pins
  # that theta_01 -- a vector, covered by scalar_rhats() -- is not mistaken
  # for a trajectory.
  fits <- fit_ll(make_y(), seed = 1, chains = 2)
  stripped <- lapply(fits, function(ch) {
    ch$theta_1 <- NULL
    ch
  })
  attributes(stripped) <- attributes(fits)

  expect_length(state_rhats(stripped), 0L)
  expect_type(state_rhats(stripped), "double")
})
